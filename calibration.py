"""
calibration.py
AMCC Calibration of the Mapped Rough SABR FMM
==============================================

This is the main calibration experiment for the thesis. It implements
Automatic Monte Carlo Calibration (AMCC) following Gonon et al. (2025):
all model parameters are optimised *jointly* by backpropagating through
the Monte Carlo simulator, treating each Euler time step as a layer in
a deep network with shared weights.

Two calibration modes are available:

  "hybrid" (recommended):
      Single-stage calibration using the BLP hybrid simulation scheme
      (Bennedsen, Lunde & Pakkanen, Finance & Stochastics 2017).
      ALL parameters — including H — are differentiable throughout.
      The hybrid scheme handles the fBm singularity with exact Wiener
      integrals in a small near-field (κ=2 cells), giving accuracy
      matching full Cholesky with only a tiny (3×3) differentiable
      Cholesky decomposition.

  "two_stage":
      Stage 1: approximate Riemann-sum scheme (H differentiable but
               biased near the singularity)
      Stage 2: exact Cholesky scheme (H frozen, unbiased fBm samples)
      The two-stage split is purely computational: the approximate
      scheme makes H differentiable but has discretisation bias; the
      exact scheme fixes H but gives unbiased fBm samples.

Usage:
    python amcc_calibration_experiment.py

Requires:
    - amcc_mapped_rough_sabr_fmm.py (with verbose=False fix for torch ≥2.2)
    - usd_swaption_data.pkl
    - PyTorch ≥ 2.0, NumPy, SciPy, matplotlib
"""

import sys
import time
import copy
import numpy as np
import torch

from main import (
    MappedRoughSABRParams,
    load_market_data,
    match_all_alphas,
    calibrate,
    calibrate_two_stage,
    print_market_summary,
    print_calibration_report,
    print_smile_comparison,
    generate_smile_plot_data,
    compute_effective_params,
    compute_vbar,
    mc_prices_to_black_iv,
    simulate_swaption,
    compute_swaption_prices,
    compute_model_smile,
    ATM_TOL,
)


# =============================================================================
# Configuration
# =============================================================================

CONFIG = {
    # Data
    "data_file": "usd_swaption_data.pkl",
    "subset": "joint_all_smiles",
    "in_sample_date": "2024-12-09",
    "out_sample_date": "2024-12-10",
    "device": "cpu",

    # Calibration mode: "hybrid" (recommended) or "two_stage" (legacy)
    #   "hybrid"    — single-stage, BLP scheme, H differentiable throughout
    #   "two_stage" — Stage 1 approx (H free) → Stage 2 exact (H frozen)
    "mode": "two_stage",

    # Single-stage hybrid settings (used when mode="hybrid")
    "hybrid": {
        "iterations": 1200,
        "lr": 5e-3,
        "N_paths": 20_000,
        "M": 50,
        "kappa": 2,                    # BLP near-field cells (2 recommended)
        "variance_mode": "full",       # can use full mode from the start
        "keys": None,                  # None = all swaptions
        "scheduler": "cosine",         # cosine annealing with warmup
        "warmup_steps": 50,            # linear warmup before cosine decay
        "early_stop_patience": 200,    # stop if no improvement for 200 steps
    },

    # Two-stage legacy settings (used when mode="two_stage")
    # Stage 1: approximate scheme (H differentiable)
    "stage1": {
        "iterations": 800,
        "lr": 5e-3,
        "N_paths": 10_000,
        "M": 50,
        "keys": None,           # None = all swaptions
    },
    # Stage 2: exact Cholesky scheme (H frozen)
    "stage2": {
        "iterations": 800,
        "lr": 5e-3,
        "N_paths": 30_000,
        "M": 50,
        "variance_mode": "full",
        "keys": None,
        "scheduler": "cosine",         # cosine annealing with warmup
        "warmup_steps": 30,            # linear warmup before cosine decay
        "cosine_power": 0.5,           # power steepening: faster early decay
    },

    # Early stopping (applies to all stages)
    "early_stop_patience": 150,        # stop if no improvement for N steps
    "early_stop_tol": 1e-4,           # relative improvement threshold

    # Diagnostics
    "diag_N_paths": 100_000,
    "diag_M": 100,

    # Reproducibility
    "crn_seed": 42,
}


# =============================================================================
# Initialisation
# =============================================================================

def _softplus_inv(a):
    """Numerically stable inverse of softplus: x = a + log(1 - exp(-a))."""
    a = float(a)
    if a > 20.0:
        return a  # softplus(x) ≈ x for large x
    return a + np.log(-np.expm1(-a))  # log(1-exp(-a)) = log(-expm1(-a))


def _interpolate_alpha(alpha, matched_indices):
    """
    Fill unmatched α values by linear interpolation with flat extrapolation,
    following the same scheme Adachi §6.2 uses for ρ₀.

    Args:
        alpha:           shape (N,) tensor — some entries already matched
        matched_indices: sorted list of 0-based indices that are pinned

    Returns:
        alpha with all entries filled
    """
    alpha = alpha.clone()
    N = alpha.shape[0]
    anchors = sorted(matched_indices)

    for j in range(N):
        if j in anchors:
            continue
        if j <= anchors[0]:
            # Flat extrapolation left
            alpha[j] = alpha[anchors[0]]
        elif j >= anchors[-1]:
            # Flat extrapolation right
            alpha[j] = alpha[anchors[-1]]
        else:
            # Linear interpolation between nearest anchors
            lo = max(a for a in anchors if a < j)
            hi = min(a for a in anchors if a > j)
            frac = (j - lo) / (hi - lo)
            alpha[j] = (1 - frac) * alpha[lo] + frac * alpha[hi]

    return alpha


def initialise_params(mkt, H_init=0.20, eta_init=2.3):
    """
    Create parameter module and warm-start ALL α values.

    Strategy:
      1. Match α_j analytically for each 1Y-tenor swaption.
         These are single-rate, so α_j = σ_ATM / (π_j √G) exactly.
         This pins α at indices corresponding to rates R_{I+1} for each
         (I, I+1) swaption — typically indices {1, 3, 5, 7, 10}.

      2. Interpolate the remaining α values (rates not covered by any
         1Y-tenor swaption) via linear interpolation with flat
         extrapolation from the matched anchors. This follows the same
         principle as Adachi §6.2 for ρ₀ interpolation.

      3. Refine using multi-rate swaptions: run match_all_alphas over
         all swaptions (including multi-rate), which uses Brent
         root-finding to scale the α vector within each swaption's
         rate window so that the model ATM matches market ATM.

    This ensures all 11 α values start near sensible levels rather
    than leaving 6 out of 11 at the uninformed default.
    """
    params = MappedRoughSABRParams(N=mkt.N, device=mkt.device)
    params.set_H(H_init)
    params.set_eta(eta_init)

    with torch.no_grad():
        p = params()

        # --- Pass 1: match 1Y-tenor swaptions (single-rate, analytic) ---
        smile_keys_1y = sorted([k for k in mkt.swaptions.keys() if k[1] == 1])
        # Formula-based matching is adequate for initialization: it provides
        # a warm start that the AMCC optimizer corrects within ~50 steps.
        # MC-based matching fails here because η_init=2.3 makes the variance
        # process too explosive for reliable simulation at this stage.
        # (OOS re-matching uses MC because no optimizer is available there.)
        alpha_matched = match_all_alphas(
            mkt, p["H"], p["eta"], p["rho0"], p["rho"],
            alpha_init=p["alpha"],
            smile_keys=smile_keys_1y,
            variance_curve_mode="simplified",
            method="formula",
        )

        # Identify which 0-based indices were pinned
        matched_indices = []
        for key in smile_keys_1y:
            swn = mkt.swaptions[key]
            if swn.J - swn.I == 1:
                matched_indices.append(swn.I)

        print(f"\n  Pass 1 — 1Y-tenor ATM matching:")
        print(f"    Matched rate indices (0-based): {matched_indices}")
        print(f"    α at matched: "
              + ", ".join(f"α[{i}]={alpha_matched[i].item():.4f}"
                          for i in matched_indices))

        # --- Pass 2: interpolate unmatched rates ---
        alpha_final = _interpolate_alpha(alpha_matched, matched_indices)

        unmatched = [j for j in range(mkt.N) if j not in matched_indices]
        print(f"  Pass 2 — interpolation for unmatched indices: {unmatched}")
        print(f"    α after interpolation: "
              + ", ".join(f"α[{j}]={alpha_final[j].item():.4f}"
                          for j in unmatched))
        print(f"    α final: [{', '.join(f'{a:.4f}' for a in alpha_final.numpy())}]")

        # NOTE: We do NOT refine with multi-rate swaptions here.
        # The Brent common-factor scaling for multi-rate swaptions
        # overwrites the accurate single-rate matches from Pass 1.
        # Instead, we let the optimizer handle the multi-rate fit
        # jointly — that's the whole point of AMCC.

        # --- Write all α back to params ---
        for j in range(mkt.N):
            a = alpha_final[j].item()
            params.alpha_tilde.data[j] = _softplus_inv(a)

    # --- Verify ATM fit across representative swaptions ---
    with torch.no_grad():
        p = params()
        print(f"\n  Initialised parameters:")
        print(f"    H     = {p['H'].item():.4f}")
        print(f"    η     = {p['eta'].item():.4f}")
        print(f"    α     = [{', '.join(f'{a:.4f}' for a in p['alpha'].numpy())}]")
        print(f"    ρ₀    = [{', '.join(f'{r:.3f}' for r in p['rho0'].numpy())}]")

        # ATM verification: confirm formula matching pinned the 1Y-tenor ATMs.
        # Note: √v̄ overestimates actual MC ATM at high η due to Jensen's
        # inequality (E[√V] < √E[V]), so multi-rate and long-dated swaptions
        # will show large positive diffs.  This is expected — the optimizer
        # corrects for it during calibration.
        check_keys = sorted(mkt.swaptions.keys())
        print(f"\n  ATM verification (formula: σ_ATM ≈ √v̄, "
              f"overestimates at high η due to Jensen):")
        for key in check_keys:
            if key not in mkt.swaptions:
                continue
            swn = mkt.swaptions[key]
            eff = compute_effective_params(
                p["alpha"], p["rho0"], p["rho"], swn,
            )
            vbar = compute_vbar(
                T=swn.expiry_years, v0=eff["v0"],
                H=p["H"], eta=p["eta"], mode="simplified",
            )
            atm_model = np.sqrt(vbar.item()) * 100
            atm_mask = (swn.strikes - swn.S0).abs() < 1e-6
            atm_mkt = swn.ivs_black[atm_mask][0].item() * 100 if atm_mask.sum() > 0 \
                else swn.ivs_black[swn.n_strikes // 2].item() * 100
            sqrt_v0 = np.sqrt(eff["v0"].item()) * 100
            G = vbar.item() / (eff["v0"].item() + 1e-30)
            exp, ten = key
            tag = "matched" if ten == 1 else "multi-rate"
            print(f"    {exp:.0f}Y×{ten:>2.0f}Y: "
                  f"√v̄={atm_model:.2f}%  mkt={atm_mkt:.2f}%  "
                  f"diff={atm_model - atm_mkt:+.2f}%  "
                  f"(√v₀={sqrt_v0:.2f}%, G={G:.3f})  [{tag}]")

    return params


# =============================================================================
# MC-based diagnostics
# =============================================================================

def mc_diagnostics(params, mkt, N_paths=100_000, M=100, seed=42):
    """
    Full MC-based calibration report with high path count.

    This is the definitive quality metric — no formula approximation.
    """
    print("\n" + "=" * 72)
    print("MC-BASED CALIBRATION REPORT")
    print(f"({N_paths:,} paths, {M} time steps)")
    print("=" * 72)

    report = print_calibration_report(
        params, mkt,
        variance_curve_mode="full",
        N_paths=N_paths,
        M=M,
        seed=seed,
    )

    return report


# =============================================================================
# Accuracy summary
# =============================================================================

def print_accuracy_summary(report, mkt, label=""):
    """
    Detailed accuracy breakdown from a calibration report.

    Computes per-swaption and aggregated metrics beyond plain RMSE:
      - RMSE, MAE              (scale of errors)
      - Mean signed error       (bias: systematic over/under-pricing)
      - ATM error              (level fit)
      - Left/right wing bias   (skew fit quality)
      - Relative price error   (hedging-relevant)

    Aggregates by expiry row, by tenor column, and overall.

    Args:
        report:  dict returned by mc_diagnostics / print_calibration_report
                 with 'per_swaption' keyed by (expiry, tenor)
        mkt:     MarketData (for access to target_prices, vegas, etc.)
        label:   optional label for the header
    """
    per_swn = report.get("per_swaption", {})
    if not per_swn:
        print("  No per-swaption results to summarise.")
        return

    header = f"ACCURACY SUMMARY — {label}" if label else "ACCURACY SUMMARY"
    print("\n" + "=" * 100)
    print(header)
    print("=" * 100)

    # ---- Per-swaption table ----
    print(f"\n{'Swaption':>10s}  {'RMSE':>7s}  {'MAE':>7s}  {'Bias':>7s}  "
          f"{'ATM':>7s}  {'MaxErr':>7s}  {'LWing':>7s}  {'RWing':>7s}  "
          f"{'PrRMSE%':>8s}")
    print(f"{'':>10s}  {'(bp)':>7s}  {'(bp)':>7s}  {'(bp)':>7s}  "
          f"{'(bp)':>7s}  {'(bp)':>7s}  {'(bp)':>7s}  {'(bp)':>7s}  "
          f"{'':>8s}")
    print("-" * 100)

    # Collectors for aggregation
    by_expiry = {}   # expiry -> list of (sq_errors, abs_errors, ...)
    by_tenor = {}    # tenor  -> list
    all_sq, all_abs, all_signed = [], [], []
    all_price_sq, all_price_denom = [], []

    for key in sorted(per_swn.keys()):
        res = per_swn[key]
        expiry, tenor = key
        swn = mkt.swaptions[key]

        valid = ~torch.isnan(res["iv_errors"])
        if valid.sum() == 0:
            print(f"  {expiry:.0f}Y×{tenor:.0f}Y  {'N/A':>7s}")
            continue

        errors = res["iv_errors"][valid] * 10000  # bp
        errors_np = errors.numpy()

        rmse = np.sqrt((errors_np ** 2).mean())
        mae = np.abs(errors_np).mean()
        bias = errors_np.mean()
        max_err = np.max(np.abs(errors_np))

        # ATM error
        atm_mask = (swn.strikes - swn.S0).abs() < ATM_TOL
        if atm_mask.sum() > 0 and not torch.isnan(res["iv_errors"][atm_mask][0]):
            atm_err = res["iv_errors"][atm_mask][0].item() * 10000
        else:
            atm_err = float('nan')

        # Wing analysis: left = K < S0, right = K > S0
        offsets = (swn.strikes - swn.S0)
        left_mask = (offsets < -ATM_TOL) & valid
        right_mask = (offsets > ATM_TOL) & valid

        left_bias = (res["iv_errors"][left_mask] * 10000).mean().item() \
            if left_mask.sum() > 0 else float('nan')
        right_bias = (res["iv_errors"][right_mask] * 10000).mean().item() \
            if right_mask.sum() > 0 else float('nan')

        # Relative pricing error: |model_price - mkt_price| / mkt_price
        mkt_prices = swn.target_prices[valid]
        mod_prices = res["model_prices"][valid]
        price_mask = mkt_prices.abs() > 1e-12
        if price_mask.sum() > 0:
            rel_errs = ((mod_prices[price_mask] - mkt_prices[price_mask])
                        / mkt_prices[price_mask]).numpy()
            price_rmse_pct = np.sqrt((rel_errs ** 2).mean()) * 100
        else:
            price_rmse_pct = float('nan')

        # Print row
        def _fmt(v, w=7, signed=False):
            if np.isnan(v):
                return f"{'N/A':>{w}s}"
            return f"{v:+{w}.1f}" if signed else f"{v:{w}.1f}"

        print(f"  {expiry:.0f}Y×{tenor:.0f}Y  {_fmt(rmse)}  {_fmt(mae)}  "
              f"{_fmt(bias, signed=True)}  {_fmt(atm_err, signed=True)}  "
              f"{_fmt(max_err)}  {_fmt(left_bias, signed=True)}  "
              f"{_fmt(right_bias, signed=True)}  "
              f"{price_rmse_pct:8.2f}%")

        # Accumulate for aggregation
        sq = (errors_np ** 2).tolist()
        ab = np.abs(errors_np).tolist()
        si = errors_np.tolist()
        all_sq.extend(sq)
        all_abs.extend(ab)
        all_signed.extend(si)

        by_expiry.setdefault(expiry, {"sq": [], "abs": [], "signed": []})
        by_expiry[expiry]["sq"].extend(sq)
        by_expiry[expiry]["abs"].extend(ab)
        by_expiry[expiry]["signed"].extend(si)

        by_tenor.setdefault(tenor, {"sq": [], "abs": [], "signed": []})
        by_tenor[tenor]["sq"].extend(sq)
        by_tenor[tenor]["abs"].extend(ab)
        by_tenor[tenor]["signed"].extend(si)

    # ---- Overall ----
    print("-" * 100)
    total_rmse = np.sqrt(np.mean(all_sq)) if all_sq else float('nan')
    total_mae = np.mean(all_abs) if all_abs else float('nan')
    total_bias = np.mean(all_signed) if all_signed else float('nan')
    print(f"  {'TOTAL':>8s}  {total_rmse:7.1f}  {total_mae:7.1f}  "
          f"{total_bias:+7.1f}")

    # ---- By expiry ----
    print(f"\n  By expiry:")
    for exp in sorted(by_expiry.keys()):
        d = by_expiry[exp]
        r = np.sqrt(np.mean(d["sq"]))
        m = np.mean(d["abs"])
        b = np.mean(d["signed"])
        print(f"    {exp:.0f}Y expiry:  RMSE={r:.1f}bp  MAE={m:.1f}bp  "
              f"Bias={b:+.1f}bp")

    # ---- By tenor ----
    print(f"\n  By tenor:")
    for ten in sorted(by_tenor.keys()):
        d = by_tenor[ten]
        r = np.sqrt(np.mean(d["sq"]))
        m = np.mean(d["abs"])
        b = np.mean(d["signed"])
        n_swn = len([k for k in per_swn if k[1] == ten])
        print(f"    {ten:.0f}Y tenor:   RMSE={r:.1f}bp  MAE={m:.1f}bp  "
              f"Bias={b:+.1f}bp  ({n_swn} swaptions)")


# =============================================================================
# Plotting
# =============================================================================

def save_smile_plots(params, mkt, config, filename="amcc_smile_fits.png",
                     suptitle=None):
    """
    Generate smile-fit plots for all swaptions in mkt.

    Reusable for both in-sample and out-of-sample evaluation.

    Args:
        params:    calibrated MappedRoughSABRParams
        mkt:       MarketData (in-sample or OOS)
        config:    dict with at least 'crn_seed'
        filename:  output filename
        suptitle:  optional super-title for the figure
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping plots")
        return

    keys_by_maturity = {}
    for key in sorted(mkt.swaptions.keys()):
        exp = key[0]
        keys_by_maturity.setdefault(exp, []).append(key)

    maturities = sorted(keys_by_maturity.keys())
    n_mat = len(maturities)
    max_per_row = max(len(v) for v in keys_by_maturity.values())

    fig, axes = plt.subplots(
        n_mat, max_per_row,
        figsize=(4.5 * max_per_row, 3.5 * n_mat),
        squeeze=False,
    )
    if suptitle:
        fig.suptitle(suptitle, fontsize=13, fontweight="bold", y=1.01)

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
                ax.set_title(f"{key[0]:.0f}Y × {key[1]:.0f}Y", fontsize=10)
                ax.set_xlabel("Strike offset (bp)", fontsize=8)
                ax.set_ylabel("IV (%)", fontsize=8)
                ax.tick_params(labelsize=7)
                ax.grid(True, alpha=0.3)
                if row_idx == 0 and col_idx == 0:
                    ax.legend(fontsize=7)

            for col_idx in range(len(keys), max_per_row):
                axes[row_idx, col_idx].set_visible(False)

    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    print(f"Saved: {filename}")
    plt.close(fig)


def save_plots(params, mkt, history, config, history2=None):
    """
    Generate and save calibration diagnostic plots:
      1. Convergence (loss & learning rate)
      2. In-sample smile fits
      3. Correlation matrix heatmap
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping plots")
        return

    # --- 1. Convergence plot ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    if history2 is not None:
        s1_steps = [r["step"] for r in history]
        s1_loss = [r["loss"] for r in history]
        s2_steps = [r["step"] + len(history) for r in history2]
        s2_loss = [r["loss"] for r in history2]

        ax1.semilogy(s1_steps, s1_loss, "b-", alpha=0.7, label="Stage 1 (approx)")
        ax1.semilogy(s2_steps, s2_loss, "r-", alpha=0.7, label="Stage 2 (exact)")
        ax1.axvline(len(history), color="gray", linestyle="--", alpha=0.5)
        ax1.set_title("Calibration convergence (two-stage)")
        ax1.legend()

        s1_lr = [r["lr"] for r in history]
        s2_lr = [r["lr"] for r in history2]
        ax2.semilogy(s1_steps, s1_lr, "b-", alpha=0.7, label="Stage 1")
        ax2.semilogy(s2_steps, s2_lr, "r-", alpha=0.7, label="Stage 2")
        ax2.axvline(len(history), color="gray", linestyle="--", alpha=0.5)
        ax2.legend()
    else:
        steps = [r["step"] for r in history]
        losses = [r["loss"] for r in history]
        lrs = [r["lr"] for r in history]

        ax1.semilogy(steps, losses, "b-", alpha=0.7, label="Hybrid (BLP)")
        ax1.set_title("Calibration convergence (hybrid)")
        ax1.legend()

        ax2.semilogy(steps, lrs, "b-", alpha=0.7, label="Hybrid")
        ax2.legend()

    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Vega-weighted loss")
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("Learning rate")
    ax2.set_title("Learning rate schedule")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("amcc_convergence.png", dpi=150)
    print("Saved: amcc_convergence.png")
    plt.close(fig)

    # --- 2. In-sample smile fits ---
    save_smile_plots(
        params, mkt, config,
        filename="amcc_smile_fits.png",
        suptitle=f"In-sample smile fits ({config.get('in_sample_date', '')})",
    )

    # --- 3. Correlation matrix heatmap ---
    with torch.no_grad():
        p = params()
        Sigma = p["Sigma"].numpy()

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(Sigma, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_title("Calibrated correlation matrix Σ")
    labels = ["W⁰"] + [f"W{i}" for i in range(1, mkt.N + 1)]
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=7, rotation=45)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=7)
    plt.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()
    plt.savefig("amcc_correlation.png", dpi=150)
    print("Saved: amcc_correlation.png")
    plt.close("all")


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":

    t_start = time.time()
    cfg = CONFIG

    # ---- Load in-sample data ----
    print("Loading in-sample market data...")
    print(f"  Date: {cfg['in_sample_date']}")
    mkt = load_market_data(
        cfg["data_file"],
        subset=cfg["subset"],
        date=cfg["in_sample_date"],
        device=cfg["device"],
    )
    print_market_summary(mkt)

    # ---- Initialise ----
    print("\n" + "#" * 60)
    print("# INITIALISATION")
    print("#" * 60)

    params = initialise_params(mkt, H_init=0.20, eta_init=2.3)

    # ---- AMCC calibration ----
    print("\n" + "#" * 60)
    print("# AMCC CALIBRATION")
    print("#" * 60)

    if cfg["mode"] == "hybrid":
        # ---- Single-stage hybrid (BLP scheme, H differentiable) ----
        hcfg = cfg["hybrid"]
        print(f"\nMode: hybrid (BLP κ={hcfg['kappa']})")
        print(f"  {hcfg['iterations']} iterations, {hcfg['N_paths']:,} paths, "
              f"{hcfg['M']} steps, lr={hcfg['lr']}")

        params.unfix_H()
        result = calibrate(
            params, mkt,
            n_iterations=hcfg["iterations"],
            lr=hcfg["lr"],
            N_paths=hcfg["N_paths"],
            M=hcfg["M"],
            scheme="hybrid",
            hybrid_kappa=hcfg["kappa"],
            variance_curve_mode=hcfg["variance_mode"],
            use_crn=True,
            crn_seed=cfg["crn_seed"],
            log_every=20,
            swaption_keys=hcfg["keys"],
            scheduler_type=hcfg.get("scheduler", "cosine"),
            warmup_steps=hcfg.get("warmup_steps", 50),
            early_stop_patience=hcfg.get("early_stop_patience",
                                         cfg.get("early_stop_patience")),
            early_stop_tol=cfg.get("early_stop_tol", 1e-4),
        )

        # Load best parameters
        if result["best_state"] is not None:
            params.load_state_dict(result["best_state"])

        with torch.no_grad():
            H_calibrated = params.get_H().item()
        print(f"\nHybrid calibration complete. H = {H_calibrated:.4f}")

    else:
        # ---- Legacy two-stage calibration ----
        print(f"\nMode: two_stage")

        result = calibrate_two_stage(
            params, mkt,
            # Stage 1
            stage1_iterations=cfg["stage1"]["iterations"],
            stage1_lr=cfg["stage1"]["lr"],
            stage1_N_paths=cfg["stage1"]["N_paths"],
            stage1_M=cfg["stage1"]["M"],
            stage1_keys=cfg["stage1"]["keys"],
            # Stage 2
            stage2_iterations=cfg["stage2"]["iterations"],
            stage2_lr=cfg["stage2"]["lr"],
            stage2_N_paths=cfg["stage2"]["N_paths"],
            stage2_M=cfg["stage2"]["M"],
            stage2_variance_mode=cfg["stage2"]["variance_mode"],
            stage2_keys=cfg["stage2"]["keys"],
            stage2_scheduler=cfg["stage2"].get("scheduler", "cosine"),
            stage2_warmup_steps=cfg["stage2"].get("warmup_steps", 30),
            stage2_cosine_power=cfg["stage2"].get("cosine_power", 0.5),
            # Common
            use_crn=True,
            crn_seed=cfg["crn_seed"],
            log_every=20,
            early_stop_patience=cfg.get("early_stop_patience", 150),
            early_stop_tol=cfg.get("early_stop_tol", 1e-4),
        )

        with torch.no_grad():
            H_calibrated = params.get_H().item()

    # ---- Calibrated parameters ----
    print("\n" + "#" * 60)
    print("# CALIBRATED PARAMETERS")
    print("#" * 60)

    with torch.no_grad():
        p = params()
        print(f"\n  H     = {p['H'].item():.4f}")
        print(f"  η     = {p['eta'].item():.4f}")

        alpha = p["alpha"].numpy()
        print(f"\n  Volatility levels α_j:")
        for j in range(mkt.N):
            print(f"    α_{j+1:2d} = {alpha[j]:.4f}")

        rho0 = p["rho0"].numpy()
        print(f"\n  Spot-vol correlations ρ₀,j:")
        for j in range(mkt.N):
            print(f"    ρ₀,{j+1:2d} = {rho0[j]:+.4f}")

        print(f"\n  Forward-rate correlation matrix ρ_ij:")
        rho = p["rho"].numpy()
        np.set_printoptions(precision=3, linewidth=120)
        print(rho)

    # ---- In-sample MC diagnostics ----
    print("\n" + "#" * 60)
    print(f"# IN-SAMPLE MC DIAGNOSTICS ({cfg['in_sample_date']})")
    print("#" * 60)

    report_in = mc_diagnostics(
        params, mkt,
        N_paths=cfg["diag_N_paths"],
        M=cfg["diag_M"],
    )
    print_accuracy_summary(report_in, mkt, label=f"IN-SAMPLE ({cfg['in_sample_date']})")

    # ---- Per-swaption smile comparisons (in-sample) ----
    print("\n" + "#" * 60)
    print("# SMILE COMPARISONS — in-sample (selected swaptions)")
    print("#" * 60)

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
                variance_curve_mode="full",
                N_paths=cfg["diag_N_paths"],
                M=cfg["diag_M"],
            )

    # ---- Plots (in-sample) ----
    print("\n" + "#" * 60)
    print("# GENERATING PLOTS")
    print("#" * 60)

    save_plots(
        params, mkt,
        history=result["history"] if cfg["mode"] == "hybrid"
                else result["stage1"]["history"],
        config=cfg,
        history2=None if cfg["mode"] == "hybrid"
                 else result["stage2"]["history"],
    )

    # ==================================================================
    # OUT-OF-SAMPLE EVALUATION
    # ==================================================================
    print("\n" + "=" * 72)
    print("  OUT-OF-SAMPLE EVALUATION")
    print(f"  Calibrated on: {cfg['in_sample_date']}")
    print(f"  Evaluating on: {cfg['out_sample_date']}")
    print("=" * 72)

    mkt_oos = load_market_data(
        cfg["data_file"],
        subset=cfg["subset"],
        date=cfg["out_sample_date"],
        device=cfg["device"],
    )
    print_market_summary(mkt_oos)

    # Re-match α to the new curve via MC-based root-finding (Adachi §6.2).
    # We keep the *same* calibrated (H, η, ρ₀, Σ) and only re-pin α to the
    # new ATM levels.  This tests whether the *shape* parameters generalise.
    #
    # The deterministic formula σ_ATM = √v̄(T) ignores Jensen's inequality
    # from stochastic variance, producing ~500bp ATM errors at η ≈ 1.7.
    # MC-based Brent solves for the α that makes the *simulated* ATM match.
    print("\n--- Re-matching α to out-of-sample ATM levels (MC-based) ---")
    params_oos = copy.deepcopy(params)
    with torch.no_grad():
        p_oos = params_oos()
        smile_keys_oos = sorted([k for k in mkt_oos.swaptions.keys() if k[1] == 1])
        alpha_oos = match_all_alphas(
            mkt_oos, p_oos["H"], p_oos["eta"], p_oos["rho0"], p_oos["rho"],
            alpha_init=p_oos["alpha"],
            smile_keys=smile_keys_oos,
            variance_curve_mode="full",
            method="mc",
            N_paths=cfg["diag_N_paths"],
            M=cfg["diag_M"],
            seed=cfg["crn_seed"],
        )

        # Identify matched indices and interpolate unmatched ones
        # (mirroring the initialise_params logic for consistency)
        matched_indices_oos = []
        for key in smile_keys_oos:
            swn = mkt_oos.swaptions[key]
            if swn.J - swn.I == 1:
                matched_indices_oos.append(swn.I)

        if matched_indices_oos:
            alpha_oos = _interpolate_alpha(alpha_oos, matched_indices_oos)

        # Write all α values back using the numerically stable inverse
        for j in range(mkt_oos.N):
            a = alpha_oos[j].item()
            params_oos.alpha_tilde.data[j] = _softplus_inv(a)

    # MC diagnostics on out-of-sample data
    print(f"\n--- Out-of-sample MC report ({cfg['out_sample_date']}) ---")
    report_oos = mc_diagnostics(
        params_oos, mkt_oos,
        N_paths=cfg["diag_N_paths"],
        M=cfg["diag_M"],
    )
    print_accuracy_summary(report_oos, mkt_oos, label=f"OUT-OF-SAMPLE ({cfg['out_sample_date']})")

    # Smile comparisons on out-of-sample
    print("\n" + "#" * 60)
    print("# SMILE COMPARISONS — out-of-sample")
    print("#" * 60)

    for key in representative_keys:
        if key in mkt_oos.swaptions:
            print_smile_comparison(
                params_oos, mkt_oos.swaptions[key], mkt_oos,
                variance_curve_mode="full",
                N_paths=cfg["diag_N_paths"],
                M=cfg["diag_M"],
            )

    # ---- OOS smile plots ----
    print("\n" + "#" * 60)
    print("# GENERATING OOS PLOTS")
    print("#" * 60)

    save_smile_plots(
        params_oos, mkt_oos, cfg,
        filename="amcc_smile_fits_oos.png",
        suptitle=f"Out-of-sample smile fits ({cfg['out_sample_date']})",
    )

    # ---- Save results ----
    elapsed = time.time() - t_start

    results = {
        "config": cfg,
        "mode": cfg["mode"],
        "params_state_dict": params.state_dict(),
        "H": p["H"].item(),
        "eta": p["eta"].item(),
        "alpha": p["alpha"].numpy(),
        "rho0": p["rho0"].numpy(),
        "rho": p["rho"].numpy(),
        "Sigma": p["Sigma"].numpy(),
        "H_calibrated": H_calibrated,
        "in_sample_date": cfg["in_sample_date"],
        "out_sample_date": cfg["out_sample_date"],
        "elapsed_seconds": elapsed,
    }
    if cfg["mode"] == "hybrid":
        results["history"] = result["history"]
    else:
        results["stage1_history"] = result["stage1"]["history"]
        results["stage2_history"] = result["stage2"]["history"]
    torch.save(results, "amcc_calibration_results.pt")
    print(f"\nResults saved to amcc_calibration_results.pt")
    print(f"Total elapsed time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print("\nDone.")