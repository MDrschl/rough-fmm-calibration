#!/usr/bin/env python3
"""
Incremental Calibration of Mapped Rough SABR FMM
=================================================

Follows the two-step calibration procedure of Adachi et al. (2025) §6.2:

  Step 1 — First-step calibration (1Y-tenor smiles):
    For each H in a grid {0.05, 0.10, ..., 0.50}:
      a) Optimize κ and ρ_{0,i} for i = 2, 4, 6, 11 (smile tenors)
      b) Interpolate remaining ρ_{0,i} linearly with flat extrapolation
      c) Pin α_i via ATM root-finding: model ATM IV = market ATM IV
      d) Iterate (a)-(c) until convergence
    Select H* that minimizes total smile RMSE across all 1Y tenors.

  Step 2 — Second-step calibration (correlation matrix):
    Fix H*, κ, ρ₀, α from Step 1.
    Calibrate the (N+1)×(N+1) correlation matrix incrementally:
      For i = 2, 3, ..., N:
        Calibrate row i of B (Rapisarda angles ω_{i,1}, ..., ω_{i,i-1})
        using i-Y co-terminal ATM swaptions.

Usage:
    python calibration_experiment.py

Requires:
    - amcc_mapped_rough_sabr_fmm.py (in same directory or on PYTHONPATH)
    - usd_swaption_data.pkl (preprocessed market data)
    - PyTorch, NumPy, SciPy, matplotlib
"""

import sys
import time
import copy
import numpy as np
import torch
import torch.nn as nn
from typing import Optional

# Import the main module
from amcc_mapped_rough_sabr_fmm import (
    MappedRoughSABRParams,
    MarketData,
    SwaptionData,
    load_market_data,
    compute_effective_params,
    compute_vbar,
    rough_sabr_iv,
    rough_sabr_prices,
    match_alpha_atm,
    match_all_alphas,
    simulate_swaption,
    compute_swaption_prices,
    mc_prices_to_black_iv,
    compute_loss_vegaweighted,
    compute_loss_ivspace,
    compute_total_loss,
    calibrate,
    print_market_summary,
    print_smile,
    print_calibration_report,
    print_smile_comparison,
    generate_smile_plot_data,
    compute_v_curve,
    black_price_torch,
    black_price_np,
)


# =============================================================================
# Utility: ρ₀ interpolation (Adachi §6.2)
# =============================================================================
#
# "Given κ, ρ_{0i} for i = 2, 4, 6, 11, the remaining ρ_{0i} are determined
# by linear interpolation and flat extrapolation."

def interpolate_rho0(
    rho0_anchor: dict,
    N: int,
) -> torch.Tensor:
    """
    Linearly interpolate ρ_{0,i} from anchor points with flat extrapolation.

    Args:
        rho0_anchor: dict {i: rho_0i_value} for anchor tenors
                     e.g. {2: -0.615, 4: -0.438, 6: -0.532, 11: -0.553}
        N:           total number of forward rates

    Returns:
        shape (N,) tensor with ρ_{0,1}, ..., ρ_{0,N}  (0-indexed: rho0[j-1] = ρ_{0,j})
    """
    rho0 = torch.zeros(N, dtype=torch.float64)
    anchors = sorted(rho0_anchor.keys())

    for j in range(1, N + 1):
        if j <= anchors[0]:
            # Flat extrapolation left
            rho0[j - 1] = rho0_anchor[anchors[0]]
        elif j >= anchors[-1]:
            # Flat extrapolation right
            rho0[j - 1] = rho0_anchor[anchors[-1]]
        elif j in rho0_anchor:
            rho0[j - 1] = rho0_anchor[j]
        else:
            # Linear interpolation
            i_lo = max(a for a in anchors if a < j)
            i_hi = min(a for a in anchors if a > j)
            frac = (j - i_lo) / (i_hi - i_lo)
            rho0[j - 1] = (1 - frac) * rho0_anchor[i_lo] + frac * rho0_anchor[i_hi]

    return rho0


# =============================================================================
# Step 1: First-step calibration (1Y-tenor smiles)
# =============================================================================

def first_step_calibration(
    mkt: MarketData,
    H_grid: list = None,
    smile_anchor_indices: list = None,
    n_outer: int = 5,
    n_adam_steps: int = 100,
    lr: float = 5e-3,
    N_paths: int = 5000,
    M: int = 30,
    use_formula: bool = True,
    verbose: bool = True,
) -> dict:
    """
    First-step calibration: scan H, optimize κ and ρ₀ on 1Y-tenor smiles.

    For each H:
      Outer loop (n_outer iterations):
        1. Match α_i via ATM root-finding
        2. Adam-optimize κ and ρ₀ anchor values on smile loss
        3. Interpolate remaining ρ₀ values

    Args:
        mkt:                    MarketData
        H_grid:                 list of H values to try
        smile_anchor_indices:   which rates have smile data (default [2,4,6,11])
        n_outer:                outer loop iterations
        n_adam_steps:           Adam steps per outer iteration
        lr:                     Adam learning rate
        N_paths:                MC paths (if not using formula)
        M:                      simulation time steps (if not using formula)
        use_formula:            True = rough SABR formula, False = MC
        verbose:                print progress

    Returns:
        dict with:
            best_H:     optimal Hurst exponent
            best_params: calibrated parameter dict
            table:      list of {H, kappa, alphas, rho0_anchors, rmse} for each H
    """
    if H_grid is None:
        H_grid = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]

    if smile_anchor_indices is None:
        smile_anchor_indices = [2, 4, 6, 11]

    # Identify 1Y-tenor swaption keys (first-step smiles)
    smile_keys = sorted([k for k in mkt.swaptions.keys() if k[1] == 1])
    if verbose:
        print(f"First-step smiles: {smile_keys}")
        print(f"Anchor indices: {smile_anchor_indices}")
        print(f"H grid: {H_grid}")

    table = []
    best_rmse = float('inf')
    best_result = None

    for H_val in H_grid:
        if verbose:
            print(f"\n{'='*60}")
            print(f"H = {H_val:.2f}")
            print(f"{'='*60}")

        # Initialize parameters
        params = MappedRoughSABRParams(N=mkt.N, device=mkt.device)
        params.set_H(H_val)
        params.fix_H()  # H is fixed in first step

        # We only optimize κ and the anchor ρ₀ values.
        # α is pinned by ATM matching.
        # The remaining ρ₀ are interpolated.

        # Outer loop: alternate between α matching and κ/ρ₀ optimization
        for outer in range(n_outer):
            p = params()

            # (a) Match α via ATM root-finding
            alpha_matched = match_all_alphas(
                mkt, p["H"], p["kappa"], p["rho0"], p["rho"],
                alpha_init=p["alpha"],
                smile_keys=smile_keys,
                variance_curve_mode="simplified",
                method="formula",
            )
            # Write back matched α
            with torch.no_grad():
                for key in smile_keys:
                    swn = mkt.swaptions[key]
                    if swn.J - swn.I == 1:
                        j = swn.I
                        target_alpha = alpha_matched[j].item()
                        # Invert softplus
                        params.alpha_tilde.data[j] = np.log(np.exp(target_alpha) - 1.0)

            # (b) Adam-optimize κ and ω (which encodes ρ₀) on smile loss
            # Only optimize kappa_tilde and the first column of omega_tilde
            params.alpha_tilde.requires_grad_(False)  # freeze α
            optimizer = torch.optim.Adam([
                {"params": [params.kappa_tilde], "lr": lr},
                {"params": [params.omega_tilde], "lr": lr * 0.5},
            ])

            for step in range(n_adam_steps):
                optimizer.zero_grad()

                # Compute loss over smile swaptions
                p_step = params()

                total_loss = torch.tensor(0.0, dtype=torch.float64)
                for key in smile_keys:
                    swn = mkt.swaptions[key]
                    eff = compute_effective_params(
                        p_step["alpha"], p_step["rho0"], p_step["rho"], swn,
                    )

                    if use_formula:
                        vbar = compute_vbar(
                            swn.expiry_years, eff["v0"], p_step["H"],
                            p_step["kappa"], mode="simplified",
                        )
                        model_ivs = rough_sabr_iv(
                            swn.strikes, swn.S0, swn.expiry_years,
                            eff["v0"], vbar, eff["rho_eff"],
                            p_step["H"], p_step["kappa"],
                        )
                        # IV-space loss (differentiable since formula is analytic)
                        iv_loss = ((model_ivs - swn.ivs_black) ** 2).sum()
                        total_loss = total_loss + iv_loss
                    else:
                        torch.manual_seed(42 + hash(key) % (2**31))
                        S_T = simulate_swaption(
                            params, swn, mkt,
                            N_paths=N_paths, M=M,
                            use_exact=False,
                            variance_curve_mode="simplified",
                        )
                        mc_prices = compute_swaption_prices(S_T, swn)
                        total_loss = total_loss + compute_loss_vegaweighted(mc_prices, swn)

                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(params.parameters(), max_norm=5.0)
                optimizer.step()

            params.alpha_tilde.requires_grad_(True)  # unfreeze for next outer

            if verbose and (outer == n_outer - 1 or outer == 0):
                with torch.no_grad():
                    p_report = params()
                    print(f"  Outer {outer}: κ={p_report['kappa'].item():.4f}, "
                          f"ρ₀ anchors: ", end="")
                    for ai in smile_anchor_indices:
                        if ai - 1 < mkt.N:
                            print(f"ρ₀,{ai}={p_report['rho0'][ai-1].item():.3f} ", end="")
                    print()

        # (c) Final α matching after optimization converges
        with torch.no_grad():
            p_final = params()
            alpha_final = match_all_alphas(
                mkt, p_final["H"], p_final["kappa"],
                p_final["rho0"], p_final["rho"],
                alpha_init=p_final["alpha"],
                smile_keys=smile_keys,
                variance_curve_mode="simplified",
                method="formula",
            )
            for key in smile_keys:
                swn = mkt.swaptions[key]
                if swn.J - swn.I == 1:
                    j = swn.I
                    params.alpha_tilde.data[j] = np.log(
                        np.exp(alpha_final[j].item()) - 1.0
                    )

        # Compute RMSE
        with torch.no_grad():
            p_eval = params()
            total_sq = 0.0
            n_total = 0
            alpha_report = {}
            for key in smile_keys:
                swn = mkt.swaptions[key]
                eff = compute_effective_params(
                    p_eval["alpha"], p_eval["rho0"], p_eval["rho"], swn,
                )
                vbar = compute_vbar(
                    swn.expiry_years, eff["v0"], p_eval["H"],
                    p_eval["kappa"], mode="simplified",
                )
                model_ivs = rough_sabr_iv(
                    swn.strikes, swn.S0, swn.expiry_years,
                    eff["v0"], vbar, eff["rho_eff"],
                    p_eval["H"], p_eval["kappa"],
                )
                sq = ((model_ivs - swn.ivs_black) ** 2).sum().item()
                total_sq += sq
                n_total += swn.n_strikes
                if swn.J - swn.I == 1:
                    alpha_report[swn.I + 1] = p_eval["alpha"][swn.I].item()

            rmse = np.sqrt(total_sq / max(1, n_total)) * 100

        rho0_anchors = {}
        for ai in smile_anchor_indices:
            if ai - 1 < mkt.N:
                rho0_anchors[ai] = p_eval["rho0"][ai - 1].item()

        row = {
            "H": H_val,
            "kappa": p_eval["kappa"].item(),
            "alphas": alpha_report,
            "rho0_anchors": rho0_anchors,
            "rmse": rmse,
            "state_dict": copy.deepcopy(params.state_dict()),
        }
        table.append(row)

        if verbose:
            alpha_str = " ".join(f"α{k}={v:.3f}" for k, v in sorted(alpha_report.items()))
            rho_str = " ".join(f"ρ₀,{k}={v*100:.1f}%" for k, v in sorted(rho0_anchors.items()))
            print(f"  → κ={row['kappa']:.4f}  {alpha_str}")
            print(f"    {rho_str}")
            print(f"    RMSE = {rmse:.3f}%")

        if rmse < best_rmse:
            best_rmse = rmse
            best_result = row

    # Print summary table
    if verbose:
        print("\n" + "=" * 80)
        print("FIRST-STEP CALIBRATION RESULTS")
        print("=" * 80)
        alpha_keys = sorted(best_result["alphas"].keys())
        rho_keys = sorted(best_result["rho0_anchors"].keys())
        header = (f"{'H':>6s}  {'κ':>8s}  "
                  + "  ".join(f"{'α'+str(k):>6s}" for k in alpha_keys)
                  + "  " + "  ".join(f"{'ρ₀,'+str(k):>8s}" for k in rho_keys)
                  + f"  {'RMSE(%)':>8s}")
        print(header)
        print("-" * len(header))
        for row in table:
            vals = (f"{row['H']:6.2f}  {row['kappa']:8.4f}  "
                    + "  ".join(f"{row['alphas'].get(k, 0):6.3f}" for k in alpha_keys)
                    + "  " + "  ".join(f"{row['rho0_anchors'].get(k, 0)*100:7.1f}%" for k in rho_keys)
                    + f"  {row['rmse']:8.3f}")
            marker = " ← best" if row["H"] == best_result["H"] else ""
            print(vals + marker)

    return {
        "best_H": best_result["H"],
        "best_result": best_result,
        "table": table,
    }


# =============================================================================
# Step 2: Second-step calibration (correlation matrix)
# =============================================================================
#
# From Adachi §6.2:
# "We treat the i-th row of b_{i,j} as the search domain and calibrate them
# by minimizing the sum of squared differences between the model IV and the
# market IV for the i-Y co-terminal swaptions – i.e., the set of swaptions
# for which the sum of the maturity and the underlying tenor equals i years."

def get_coterminal_keys(mkt: MarketData, terminal_year: int) -> list:
    """
    Get co-terminal swaption keys: expiry + tenor = terminal_year.

    E.g., for terminal_year=5: (1,4), (2,3), (3,2), (4,1).
    Only returns keys that exist in mkt.swaptions.
    """
    keys = []
    for key in sorted(mkt.swaptions.keys()):
        expiry, tenor = key
        if abs(expiry + tenor - terminal_year) < 0.1:
            keys.append(key)
    return keys


def second_step_calibration(
    mkt: MarketData,
    first_step_result: dict,
    min_tenor: int = 2,
    n_adam_steps: int = 200,
    lr: float = 1e-2,
    use_formula: bool = True,
    N_paths: int = 10000,
    M: int = 50,
    verbose: bool = True,
) -> MappedRoughSABRParams:
    """
    Second-step calibration: incremental row-by-row correlation estimation.

    Fixes H, κ, α, ρ₀ from the first step.
    Calibrates the inter-rate correlation matrix ρ_{ij} for i,j = 1,...,N
    by optimizing Rapisarda angles row by row using co-terminal swaptions.

    Args:
        mkt:                MarketData
        first_step_result:  output of first_step_calibration()
        min_tenor:          minimum underlying tenor for co-terminal swaptions
        n_adam_steps:       Adam steps per row
        lr:                 learning rate
        use_formula:        True = rough SABR, False = MC
        N_paths, M:         MC settings
        verbose:            print progress

    Returns:
        calibrated MappedRoughSABRParams
    """
    best = first_step_result["best_result"]

    # Initialize params from first-step best
    params = MappedRoughSABRParams(N=mkt.N, device=mkt.device)
    params.load_state_dict(best["state_dict"])
    params.fix_H()

    # Freeze everything except omega_tilde (correlation angles)
    params.kappa_tilde.requires_grad_(False)
    params.alpha_tilde.requires_grad_(False)
    # omega_tilde stays free — we'll selectively optimize rows

    if verbose:
        with torch.no_grad():
            p = params()
            print(f"Second-step: H={p['H'].item():.4f}, κ={p['kappa'].item():.4f}")
            print(f"Calibrating correlation matrix rows {min_tenor}..{mkt.N}")

    # Incremental row-by-row calibration
    for row_i in range(min_tenor, mkt.N + 1):
        # Co-terminal swaptions for this row
        coterminal = get_coterminal_keys(mkt, row_i)
        # Filter to those with tenor >= min_tenor
        coterminal = [k for k in coterminal if k[1] >= min_tenor]

        if not coterminal:
            if verbose:
                print(f"  Row {row_i}: no co-terminal swaptions, skipping")
            continue

        if verbose:
            print(f"\n  Row {row_i}: co-terminal swaptions = {coterminal}")

        # Create a mask: only optimize row row_i of omega_tilde
        # Row row_i in the (N+1)×(N+1) matrix corresponds to rate index row_i
        # Columns 1..row_i-1 are the free angles (column 0 is ρ₀, already fixed)

        # Save initial omega_tilde
        omega_save = params.omega_tilde.data.clone()

        # Optimizer: only the relevant row's angles
        # We use a trick: optimize all of omega_tilde but zero out gradients
        # for all rows except row_i
        optimizer = torch.optim.Adam([params.omega_tilde], lr=lr)

        for step in range(n_adam_steps):
            optimizer.zero_grad()
            p_step = params()

            total_loss = torch.tensor(0.0, dtype=torch.float64)
            for key in coterminal:
                swn = mkt.swaptions[key]
                eff = compute_effective_params(
                    p_step["alpha"], p_step["rho0"], p_step["rho"], swn,
                )

                if use_formula:
                    vbar = compute_vbar(
                        swn.expiry_years, eff["v0"], p_step["H"],
                        p_step["kappa"], mode="simplified",
                    )
                    # ATM only for correlation calibration
                    atm_mask = (swn.strikes - swn.S0).abs() < 1e-10
                    if atm_mask.sum() > 0:
                        atm_iv_model = torch.sqrt(vbar + 1e-30)
                        atm_iv_mkt = swn.ivs_black[atm_mask][0]
                        total_loss = total_loss + (atm_iv_model - atm_iv_mkt) ** 2
                    else:
                        # Use full smile
                        model_ivs = rough_sabr_iv(
                            swn.strikes, swn.S0, swn.expiry_years,
                            eff["v0"], vbar, eff["rho_eff"],
                            p_step["H"], p_step["kappa"],
                        )
                        total_loss = total_loss + ((model_ivs - swn.ivs_black) ** 2).sum()
                else:
                    torch.manual_seed(42 + hash(key) % (2**31))
                    S_T = simulate_swaption(
                        params, swn, mkt,
                        N_paths=N_paths, M=M,
                        use_exact=False, variance_curve_mode="simplified",
                    )
                    mc_prices = compute_swaption_prices(S_T, swn)
                    total_loss = total_loss + compute_loss_vegaweighted(mc_prices, swn)

            total_loss.backward()

            # Zero out gradients for all rows except row_i
            if params.omega_tilde.grad is not None:
                mask = torch.zeros_like(params.omega_tilde.grad)
                mask[row_i] = 1.0
                # Also keep column 0 frozen (ρ₀ was calibrated in step 1)
                mask[row_i, 0] = 0.0
                params.omega_tilde.grad *= mask

            optimizer.step()

        if verbose:
            with torch.no_grad():
                loss_val = total_loss.item()
                p_check = params()
                rho_row = p_check["rho"][row_i - 1, :row_i].numpy()
                print(f"    loss={loss_val:.8f}  ρ[{row_i},1:{row_i}] = {np.array2string(rho_row, precision=3)}")

    # Unfreeze everything for potential further use
    params.kappa_tilde.requires_grad_(True)
    params.alpha_tilde.requires_grad_(True)
    params.unfix_H()

    return params


# =============================================================================
# Main: run the full Adachi calibration
# =============================================================================

if __name__ == "__main__":

    t_start = time.time()

    # --- Load data ---
    print("Loading market data...")
    mkt = load_market_data(
        "usd_swaption_data.pkl",
        subset="joint_all_smiles",
        convert_otm_from_bachelier=True,
        device="cpu",
    )
    print_market_summary(mkt)

    # --- Step 1: First-step calibration ---
    print("\n" + "#" * 60)
    print("# STEP 1: First-step calibration (1Y-tenor smiles)")
    print("#" * 60)

    step1 = first_step_calibration(
        mkt,
        H_grid=[0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50],
        smile_anchor_indices=[2, 4, 6, 11],
        n_outer=3,
        n_adam_steps=80,
        lr=5e-3,
        use_formula=True,
        verbose=True,
    )

    print(f"\n→ Best H = {step1['best_H']:.2f}, "
          f"RMSE = {step1['best_result']['rmse']:.3f}%")

    # --- Step 2: Second-step calibration ---
    print("\n" + "#" * 60)
    print("# STEP 2: Correlation matrix calibration")
    print("#" * 60)

    params = second_step_calibration(
        mkt,
        step1,
        min_tenor=2,
        n_adam_steps=150,
        lr=1e-2,
        use_formula=True,
        verbose=True,
    )

    # --- Final diagnostics ---
    print("\n" + "#" * 60)
    print("# FINAL DIAGNOSTICS")
    print("#" * 60)

    with torch.no_grad():
        p = params()
        print(f"\nCalibrated parameters:")
        print(f"  H     = {p['H'].item():.4f}")
        print(f"  κ     = {p['kappa'].item():.4f}")
        print(f"  α     = {p['alpha'].numpy()}")
        print(f"  ρ₀    = {p['rho0'].numpy()}")
        print(f"\n  Correlation matrix ρ_ij:")
        rho = p['rho'].numpy()
        np.set_printoptions(precision=3, linewidth=120)
        print(rho)

    # Report
    print("\n--- Formula-based report ---")
    print_calibration_report(
        params, mkt,
        method="formula",
        variance_curve_mode="simplified",
    )

    # Smile comparisons for selected swaptions
    for key in [(1.0, 1), (3.0, 1), (5.0, 1), (5.0, 5)]:
        if key in mkt.swaptions:
            print_smile_comparison(
                params, mkt.swaptions[key], mkt,
                method="formula",
                variance_curve_mode="simplified",
            )

    elapsed = time.time() - t_start
    print(f"\n\nTotal elapsed time: {elapsed:.1f}s")

    # --- Save results ---
    results = {
        "step1": step1,
        "params_state_dict": params.state_dict(),
        "H": p["H"].item(),
        "kappa": p["kappa"].item(),
        "alpha": p["alpha"].numpy(),
        "rho0": p["rho0"].numpy(),
        "rho": p["rho"].numpy(),
    }
    torch.save(results, "calibration_results.pt")
    print("Results saved to calibration_results.pt")

    # --- Generate plot data for visualization ---
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plot_data = generate_smile_plot_data(
            params, mkt,
            method="formula",
            variance_curve_mode="simplified",
        )

        # Smile plots
        one_y_keys = sorted([k for k in plot_data.keys() if k[1] == 1])
        if one_y_keys:
            fig, axes = plt.subplots(1, min(4, len(one_y_keys)),
                                     figsize=(5 * min(4, len(one_y_keys)), 4))
            if len(one_y_keys) == 1:
                axes = [axes]
            for ax, key in zip(axes, one_y_keys):
                d = plot_data[key]
                ax.plot(d["offsets_bp"], d["market_ivs_pct"], "ko-",
                        label="Market", markersize=4)
                ax.plot(d["offsets_bp"], d["model_ivs_pct"], "r^--",
                        label="Model", markersize=4)
                ax.set_title(f"{key[0]:.0f}Y × {key[1]:.0f}Y")
                ax.set_xlabel("Strike offset (bp)")
                ax.set_ylabel("IV (%)")
                ax.legend(fontsize=8)
                ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig("smile_fit_1y.png", dpi=150)
            print("Smile plot saved to smile_fit_1y.png")

        # H scan plot
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        H_vals = [r["H"] for r in step1["table"]]
        rmse_vals = [r["rmse"] for r in step1["table"]]
        ax2.plot(H_vals, rmse_vals, "bo-")
        ax2.axvline(step1["best_H"], color="r", linestyle="--",
                    label=f"Best H={step1['best_H']:.2f}")
        ax2.set_xlabel("Hurst exponent H")
        ax2.set_ylabel("RMSE (%)")
        ax2.set_title("First-step RMSE vs Hurst exponent")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig("H_scan.png", dpi=150)
        print("H scan plot saved to H_scan.png")

        plt.close("all")

    except ImportError:
        print("matplotlib not available, skipping plots")

    print("\nDone.")