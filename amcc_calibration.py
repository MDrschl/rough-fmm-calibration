"""
AMCC Calibration of the Mapped Rough SABR FMM
==============================================

Joint calibration of the Mapped Rough SABR Forward Market Model (Adachi et al.,
2025) to USD swaption implied volatility smiles using Automatic Monte Carlo
Calibration (Gonon & Stockinger, 2025).

Algorithm overview
------------------
The calibration proceeds in four steps:

  Step 1 — Load market data
      19 swaptions (1Y–10Y expiry × 1Y–10Y tenor), 9 strikes each = 171 IVs.
      OTM implied vols are converted from Bachelier (bps) to Black (lognormal).

  Step 2 — Initialize parameters
      (a) Set H=0.2, κ=1.0, ρ₀=-0.5, Σ≈I  (prior from Adachi §6.2).
      (b) Match α_j analytically for each 1Y-tenor swaption:
              α_j = σ_ATM / (π_j √G)  where  G = ∫₀¹ exp(κ²(Ts)^{2H}/8H) ds.
          This pins 5 anchor rates (indices 1,3,5,7,10).
      (c) Fill the remaining 6 rates by linear interpolation of the anchors.

  Step 3 — Two-stage gradient optimization (AMCC)
      Stage 1: Approximate kernel discretization  (H differentiable).
          W̃^H_{t_{i+1}} ≈ Σ_k ((i+1-k)h)^{H-1/2} ΔW⁰_k
          Optimizes all parameters jointly: H, κ, α_j, ρ₀,j, ρ_{ij}.
          Uses simplified variance curve v(t)/v(0) = exp(κ²t^{2H}/8H).

      Stage 2: Exact Cholesky fBm sampling  (H frozen at Stage 1 value).
          Joint (W̃^H, W⁰) sampled from precomputed 2M×2M covariance.
          Refines α_j, ρ₀,j, ρ_{ij} with full variance curve ξ_j(t).

      Both stages use:
        - Adam optimizer with CosineAnnealingLR (robust to MC noise)
        - Vega-weighted price loss: L = Σ_k ((P_MC - P_mkt)/Vega_k)²
        - Common random numbers (CRN) for gradient variance reduction

  Step 4 — In-sample diagnostics
      Reprice all 171 data points with 100k paths / 100 steps (exact scheme,
      full variance curve).  Reports per-swaption RMSE and smile fits.

Usage:
    python amcc_calibration.py

Requires:
    amcc_mapped_rough_sabr_fmm.py, usd_swaption_data.pkl
    PyTorch >= 2.0, NumPy, SciPy, matplotlib
"""

import sys
import time
import numpy as np
import torch

from amcc_mapped_rough_sabr_fmm import (
    MappedRoughSABRParams,
    load_market_data,
    match_all_alphas,
    calibrate_two_stage,
    print_market_summary,
    print_calibration_report,
    print_smile_comparison,
    generate_smile_plot_data,
    compute_effective_params,
    mc_prices_to_black_iv,
    simulate_swaption,
    compute_swaption_prices,
)


# #############################################################################
#
#   CONFIGURATION
#
# #############################################################################

CONFIG = {
    # --- Data ---
    "data_file": "usd_swaption_data.pkl",
    "subset": "joint_all_smiles",       # 19 swaptions x 9 strikes
    "device": "cpu",

    # --- Step 2: Initialization ---
    "H_init": 0.20,                     # Adachi's optimal for EUR data
    "kappa_init": 1.0,

    # --- Step 3a: Stage 1 (approximate scheme, H differentiable) ---
    #   Gonon: 500 iters x 25k paths for ~4 params.
    #   We have ~80 params -> more iterations, fewer paths (approximate is O(M^2)).
    "stage1": {
        "iterations": 800,
        "lr": 5e-3,
        "N_paths": 10_000,
        "M": 50,
        "scheduler": "cosine",          # cosine annealing: lr decays smoothly
        "min_lr": 5e-4,                 #   from lr -> min_lr over all iterations
        "keys": None,                   # None = all 19 swaptions
    },

    # --- Step 3b: Stage 2 (exact Cholesky, H frozen) ---
    #   More paths for lower MC variance; full xi_j(t) variance curve.
    "stage2": {
        "iterations": 800,
        "lr": 3e-3,
        "N_paths": 30_000,
        "M": 50,
        "variance_mode": "full",
        "scheduler": "cosine",
        "min_lr": 1e-4,
        "keys": None,
    },

    # --- Step 4: In-sample diagnostics ---
    "diag_N_paths": 100_000,
    "diag_M": 100,

    # --- Reproducibility ---
    "crn_seed": 42,
}


# #############################################################################
#
#   STEP 2: INITIALIZATION HELPERS
#
# #############################################################################

def _softplus_inv(a):
    """Numerically stable inverse of softplus: x s.t. log(1+exp(x)) = a."""
    a = float(a)
    if a > 20.0:
        return a
    return a + np.log(-np.expm1(-a))


def _interpolate_alpha(alpha, matched_indices):
    """
    Fill unmatched alpha values by linear interpolation with flat extrapolation.

    Follows the same principle as Adachi sec. 6.2 for rho_0 interpolation.
    Anchor points are the analytically matched 1Y-tenor rates.
    """
    alpha = alpha.clone()
    N = alpha.shape[0]
    anchors = sorted(matched_indices)

    for j in range(N):
        if j in anchors:
            continue
        if j <= anchors[0]:
            alpha[j] = alpha[anchors[0]]
        elif j >= anchors[-1]:
            alpha[j] = alpha[anchors[-1]]
        else:
            lo = max(a for a in anchors if a < j)
            hi = min(a for a in anchors if a > j)
            frac = (j - lo) / (hi - lo)
            alpha[j] = (1 - frac) * alpha[lo] + frac * alpha[hi]

    return alpha


def initialise_params(mkt, H_init=0.20, kappa_init=1.0):
    """
    Step 2: Create parameters and warm-start alpha via ATM matching.

    Returns a MappedRoughSABRParams module with:
      - H, kappa at the specified initial values
      - alpha_j matched so that sqrt(vbar(T)) ~ sigma_ATM^mkt for 1Y-tenor
        swaptions, then interpolated for rates not covered by any 1Y-tenor smile
      - rho_0 ~ -0.5, Sigma ~ I  (default initialization)
    """
    params = MappedRoughSABRParams(N=mkt.N, device=mkt.device)
    params.set_H(H_init)
    params.set_kappa(kappa_init)

    with torch.no_grad():
        p = params()

        # (a) Match 1Y-tenor swaptions analytically (single-rate: exact)
        #     Each 1Y-tenor swaption (expiry, 1) has J-I = 1, so pins one alpha_j.
        smile_keys_1y = sorted([k for k in mkt.swaptions.keys() if k[1] == 1])
        alpha_matched = match_all_alphas(
            mkt, p["H"], p["kappa"], p["rho0"], p["rho"],
            alpha_init=p["alpha"],
            smile_keys=smile_keys_1y,
            variance_curve_mode="simplified",
            method="formula",
        )

        matched_indices = []
        for key in smile_keys_1y:
            swn = mkt.swaptions[key]
            if swn.J - swn.I == 1:
                matched_indices.append(swn.I)

        print(f"\n  (a) ATM matching on 1Y-tenor swaptions:")
        print(f"      Pinned rate indices (0-based): {matched_indices}")
        for i in matched_indices:
            print(f"      alpha[{i}] = {alpha_matched[i].item():.4f}")

        # (b) Interpolate unmatched rates
        alpha_final = _interpolate_alpha(alpha_matched, matched_indices)

        unmatched = [j for j in range(mkt.N) if j not in matched_indices]
        print(f"  (b) Interpolation for unmatched indices: {unmatched}")
        for j in unmatched:
            print(f"      alpha[{j}] = {alpha_final[j].item():.4f}")

        # Write all alpha back to unconstrained params
        for j in range(mkt.N):
            params.alpha_tilde.data[j] = _softplus_inv(alpha_final[j].item())

    # Verify
    with torch.no_grad():
        p = params()
        print(f"\n  Initialized parameters:")
        print(f"    H = {p['H'].item():.4f},  kappa = {p['kappa'].item():.4f}")
        print(f"    alpha = [{', '.join(f'{a:.4f}' for a in p['alpha'].numpy())}]")
        print(f"    rho0  = [{', '.join(f'{r:.3f}' for r in p['rho0'].numpy())}]")

        # ATM check on representative swaptions
        check_keys = [
            (1.0, 1), (1.0, 5), (1.0, 10),
            (3.0, 1), (5.0, 5), (10.0, 1),
        ]
        print(f"\n  ATM verification (sqrt(v(0)) vs market sigma_ATM):")
        for key in check_keys:
            if key not in mkt.swaptions:
                continue
            swn = mkt.swaptions[key]
            eff = compute_effective_params(p["alpha"], p["rho0"], p["rho"], swn)
            model_pct = np.sqrt(eff["v0"].item()) * 100
            mkt_pct = swn.ivs_black[swn.n_strikes // 2].item() * 100
            print(f"    {key[0]:.0f}Y x {key[1]:.0f}Y:  "
                  f"model {model_pct:.2f}%  market {mkt_pct:.2f}%  "
                  f"diff = {model_pct - mkt_pct:+.2f}%")

    return params


# #############################################################################
#
#   STEP 4: IN-SAMPLE DIAGNOSTICS
#
# #############################################################################

def run_diagnostics(params, mkt, N_paths=100_000, M=100, seed=42):
    """
    Step 4: Reprice all swaptions with high-quality MC and report fit quality.

    This is the in-sample test: same 171 data points used for calibration,
    but repriced with 100k paths / 100 steps / exact scheme / full v(t).
    It removes MC noise from training and reveals the true model-vs-market gap.
    """
    print("\n" + "=" * 72)
    print("IN-SAMPLE FIT  (exact scheme, full xi_j(t) variance curve)")
    print(f"  {N_paths:,} paths x {M} time steps per swaption")
    print("=" * 72)

    report = print_calibration_report(
        params, mkt,
        method="mc",
        variance_curve_mode="full",
        N_paths=N_paths,
        M=M,
        seed=seed,
    )

    return report


# #############################################################################
#
#   PLOTTING
#
# #############################################################################

def save_plots(params, mkt, step1_history, step2_history, config):
    """Generate convergence, smile fit, and correlation plots."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping plots")
        return

    # --- 1. Convergence ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    s1_steps = [r["step"] for r in step1_history]
    s1_loss = [r["loss"] for r in step1_history]
    s2_steps = [r["step"] + len(step1_history) for r in step2_history]
    s2_loss = [r["loss"] for r in step2_history]

    ax1.semilogy(s1_steps, s1_loss, "b-", alpha=0.7, label="Stage 1 (approx)")
    ax1.semilogy(s2_steps, s2_loss, "r-", alpha=0.7, label="Stage 2 (exact)")
    ax1.axvline(len(step1_history), color="gray", ls="--", alpha=0.5)
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Vega-weighted loss")
    ax1.set_title("Calibration convergence")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    s1_lr = [r["lr"] for r in step1_history]
    s2_lr = [r["lr"] for r in step2_history]
    ax2.semilogy(s1_steps, s1_lr, "b-", alpha=0.7, label="Stage 1")
    ax2.semilogy(s2_steps, s2_lr, "r-", alpha=0.7, label="Stage 2")
    ax2.axvline(len(step1_history), color="gray", ls="--", alpha=0.5)
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("Learning rate")
    ax2.set_title("Learning rate schedule")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("amcc_convergence.png", dpi=150)
    print("Saved: amcc_convergence.png")

    # --- 2. Smile fits ---
    keys_by_maturity = {}
    for key in sorted(mkt.swaptions.keys()):
        keys_by_maturity.setdefault(key[0], []).append(key)

    maturities = sorted(keys_by_maturity.keys())
    n_mat = len(maturities)
    max_per_row = max(len(v) for v in keys_by_maturity.values())

    fig, axes = plt.subplots(
        n_mat, max_per_row,
        figsize=(4.5 * max_per_row, 3.5 * n_mat),
        squeeze=False,
    )

    cholesky_cache = {}
    with torch.no_grad():
        for row_idx, mat in enumerate(maturities):
            keys = keys_by_maturity[mat]
            for col_idx, key in enumerate(keys):
                ax = axes[row_idx, col_idx]
                swn = mkt.swaptions[key]

                torch.manual_seed(config["crn_seed"])
                S_T = simulate_swaption(
                    params, swn, mkt,
                    N_paths=50_000, M=80,
                    use_exact=True,
                    variance_curve_mode="full",
                    cholesky_cache=cholesky_cache,
                )
                mc_prices = compute_swaption_prices(S_T, swn)
                model_ivs = mc_prices_to_black_iv(mc_prices, swn)

                offsets = ((swn.strikes - swn.S0) * 10000).numpy()
                mkt_ivs = swn.ivs_black.numpy() * 100
                mod_ivs = model_ivs.numpy() * 100

                ax.plot(offsets, mkt_ivs, "ko-", markersize=3, label="Market")
                ax.plot(offsets, mod_ivs, "r^--", markersize=3, label="Model")
                ax.set_title(f"{key[0]:.0f}Y x {key[1]:.0f}Y", fontsize=10)
                ax.set_xlabel("Strike offset (bp)", fontsize=8)
                ax.set_ylabel("IV (%)", fontsize=8)
                ax.tick_params(labelsize=7)
                ax.grid(True, alpha=0.3)
                if row_idx == 0 and col_idx == 0:
                    ax.legend(fontsize=7)

            for col_idx in range(len(keys), max_per_row):
                axes[row_idx, col_idx].set_visible(False)

    plt.tight_layout()
    plt.savefig("amcc_smile_fits.png", dpi=150)
    print("Saved: amcc_smile_fits.png")

    # --- 3. Correlation matrix ---
    with torch.no_grad():
        p = params()
        Sigma = p["Sigma"].numpy()

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(Sigma, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_title("Calibrated correlation matrix Sigma")
    labels = ["W0"] + [f"W{i}" for i in range(1, mkt.N + 1)]
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=7, rotation=45)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=7)
    plt.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()
    plt.savefig("amcc_correlation.png", dpi=150)
    print("Saved: amcc_correlation.png")

    plt.close("all")


# #############################################################################
#
#   MAIN
#
# #############################################################################

if __name__ == "__main__":

    t_start = time.time()
    cfg = CONFIG

    # =================================================================
    # STEP 1: Load market data
    # =================================================================
    print("=" * 60)
    print("STEP 1: Load market data")
    print("=" * 60)

    mkt = load_market_data(
        cfg["data_file"],
        subset=cfg["subset"],
        convert_otm_from_bachelier=True,
        device=cfg["device"],
    )
    print_market_summary(mkt)

    # =================================================================
    # STEP 2: Initialize parameters
    # =================================================================
    print("\n" + "=" * 60)
    print("STEP 2: Initialize parameters")
    print("=" * 60)

    params = initialise_params(
        mkt,
        H_init=cfg["H_init"],
        kappa_init=cfg["kappa_init"],
    )

    # =================================================================
    # STEP 3: Two-stage AMCC calibration
    # =================================================================
    print("\n" + "=" * 60)
    print("STEP 3: Two-stage AMCC calibration")
    print("=" * 60)

    s1 = cfg["stage1"]
    s2 = cfg["stage2"]

    result = calibrate_two_stage(
        params, mkt,
        # Stage 1: approximate scheme
        stage1_iterations=s1["iterations"],
        stage1_lr=s1["lr"],
        stage1_N_paths=s1["N_paths"],
        stage1_M=s1["M"],
        stage1_keys=s1["keys"],
        stage1_scheduler=s1["scheduler"],
        stage1_min_lr=s1["min_lr"],
        # Stage 2: exact Cholesky scheme
        stage2_iterations=s2["iterations"],
        stage2_lr=s2["lr"],
        stage2_N_paths=s2["N_paths"],
        stage2_M=s2["M"],
        stage2_variance_mode=s2["variance_mode"],
        stage2_keys=s2["keys"],
        stage2_scheduler=s2["scheduler"],
        stage2_min_lr=s2["min_lr"],
        # Common
        use_crn=True,
        crn_seed=cfg["crn_seed"],
        log_every=20,
    )

    # Print calibrated parameters
    print("\n" + "-" * 60)
    print("Calibrated parameters")
    print("-" * 60)

    with torch.no_grad():
        p = params()
        print(f"\n  H     = {p['H'].item():.4f}")
        print(f"  kappa = {p['kappa'].item():.4f}")

        alpha = p["alpha"].numpy()
        print(f"\n  Volatility levels alpha_j:")
        for j in range(mkt.N):
            print(f"    alpha_{j+1:2d} = {alpha[j]:.4f}")

        rho0 = p["rho0"].numpy()
        print(f"\n  Spot-vol correlations rho_0,j:")
        for j in range(mkt.N):
            print(f"    rho_0,{j+1:2d} = {rho0[j]:+.4f}")

        print(f"\n  Forward-rate correlation matrix rho_ij:")
        np.set_printoptions(precision=3, linewidth=120)
        print(p["rho"].numpy())

    # =================================================================
    # STEP 4: In-sample diagnostics
    # =================================================================
    print("\n" + "=" * 60)
    print("STEP 4: In-sample diagnostics")
    print("=" * 60)

    report = run_diagnostics(
        params, mkt,
        N_paths=cfg["diag_N_paths"],
        M=cfg["diag_M"],
    )

    # Smile comparisons for selected swaptions
    print("\n" + "-" * 60)
    print("Smile comparisons")
    print("-" * 60)

    representative_keys = [
        (1.0, 1), (1.0, 5), (1.0, 10),
        (3.0, 1), (3.0, 5),
        (5.0, 1), (5.0, 5),
        (7.0, 1), (7.0, 3),
        (10.0, 1),
    ]
    for key in representative_keys:
        if key in mkt.swaptions:
            print_smile_comparison(
                params, mkt.swaptions[key], mkt,
                method="mc",
                variance_curve_mode="full",
                N_paths=cfg["diag_N_paths"],
                M=cfg["diag_M"],
            )

    # =================================================================
    # Save outputs
    # =================================================================
    print("\n" + "-" * 60)
    print("Generating plots and saving results")
    print("-" * 60)

    save_plots(
        params, mkt,
        result["stage1"]["history"],
        result["stage2"]["history"],
        cfg,
    )

    elapsed = time.time() - t_start

    results = {
        "config": cfg,
        "params_state_dict": params.state_dict(),
        "H": p["H"].item(),
        "kappa": p["kappa"].item(),
        "alpha": p["alpha"].numpy(),
        "rho0": p["rho0"].numpy(),
        "rho": p["rho"].numpy(),
        "Sigma": p["Sigma"].numpy(),
        "stage1_history": result["stage1"]["history"],
        "stage2_history": result["stage2"]["history"],
        "H_calibrated": result["H_calibrated"],
        "elapsed_seconds": elapsed,
    }
    torch.save(results, "amcc_calibration_results.pt")
    print(f"\nResults saved to amcc_calibration_results.pt")
    print(f"Total elapsed time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print("\nDone.")