"""
calibration_cross.py
AMCC Calibration of the Mapped Rough SABR FMM
==============================================

Cross-sectional out-of-sample evaluation:
  - Calibrates on a TRAIN subset of the swaption cube (single date)
  - Evaluates on a held-out TEST subset (same date, same curve)
  - No α re-matching for the test set: this tests whether the jointly
    calibrated (H, η, α, ρ₀, ρ) generalise to swaptions the optimizer
    never saw

All 1Y-tenor swaptions must remain in the training set because they
are needed for α initialization (single-rate ATM matching).

Two calibration modes are available:

  "hybrid" (recommended):
      Single-stage calibration using the BLP hybrid simulation scheme
      (Bennedsen, Lunde & Pakkanen, Finance & Stochastics 2017).
      ALL parameters — including H — are differentiable throughout.

  "two_stage":
      Stage 1: approximate Riemann-sum scheme (H differentiable but
               biased near the singularity)
      Stage 2: exact Cholesky scheme (H frozen, unbiased fBm samples)

Usage:
    python calibration_cross.py
"""

import sys
import time
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

# Full cube has 19 swaptions.  We hold out 6 that are spread across
# expiries and tenors.  All 1Y-tenor (single-rate) swaptions MUST stay
# in the training set — they are needed for α initialization.
TEST_KEYS = [
    (1.0, 3), (1.0, 7),       # short expiry, medium/long tenor
    (3.0, 2), (3.0, 7),       # medium expiry, short/long tenor
    (5.0, 3),                  # medium-long expiry
    (7.0, 2),                  # long expiry
]

CONFIG = {
    # Data
    "data_file": "usd_swaption_data.pkl",
    "subset": "joint_all_smiles",       # load all 19 swaptions
    "date": "2024-12-09",
    "device": "cpu",

    # Train/test split
    "test_keys": TEST_KEYS,

    # Calibration mode: "hybrid" (recommended) or "two_stage" (legacy)
    "mode": "two_stage",

    # Single-stage hybrid settings (used when mode="hybrid")
    "hybrid": {
        "iterations": 1200,
        "lr": 5e-3,
        "N_paths": 20_000,
        "M": 50,
        "kappa": 2,
        "variance_mode": "full",
        "scheduler": "cosine",
        "warmup_steps": 50,
        "early_stop_patience": 200,
    },

    # Two-stage legacy settings (used when mode="two_stage")
    "stage1": {
        "iterations": 800,
        "lr": 5e-3,
        "N_paths": 10_000,
        "M": 50,
    },
    "stage2": {
        "iterations": 800,
        "lr": 5e-3,
        "N_paths": 30_000,
        "M": 50,
        "variance_mode": "full",
        "scheduler": "cosine",
        "warmup_steps": 30,
        "cosine_power": 0.5,
    },

    # Early stopping (applies to all stages)
    "early_stop_patience": 150,
    "early_stop_tol": 1e-4,

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


def initialise_params(mkt, H_init=0.20, eta_init=2.3):
    """
    Create parameter module and warm-start ALL α values.

    Strategy:
      1. Match α_j analytically for each 1Y-tenor swaption (single-rate).
      2. Interpolate remaining α values via linear interp + flat extrap.
    """
    params = MappedRoughSABRParams(N=mkt.N, device=mkt.device)
    params.set_H(H_init)
    params.set_eta(eta_init)

    with torch.no_grad():
        p = params()

        smile_keys_1y = sorted([k for k in mkt.swaptions.keys() if k[1] == 1])
        alpha_matched = match_all_alphas(
            mkt, p["H"], p["eta"], p["rho0"], p["rho"],
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

        print(f"\n  Pass 1 — 1Y-tenor ATM matching:")
        print(f"    Matched rate indices (0-based): {matched_indices}")
        print(f"    α at matched: "
              + ", ".join(f"α[{i}]={alpha_matched[i].item():.4f}"
                          for i in matched_indices))

        alpha_final = _interpolate_alpha(alpha_matched, matched_indices)

        unmatched = [j for j in range(mkt.N) if j not in matched_indices]
        print(f"  Pass 2 — interpolation for unmatched indices: {unmatched}")
        print(f"    α after interpolation: "
              + ", ".join(f"α[{j}]={alpha_final[j].item():.4f}"
                          for j in unmatched))
        print(f"    α final: [{', '.join(f'{a:.4f}' for a in alpha_final.numpy())}]")

        for j in range(mkt.N):
            a = alpha_final[j].item()
            params.alpha_tilde.data[j] = _softplus_inv(a)

    with torch.no_grad():
        p = params()
        print(f"\n  Initialised parameters:")
        print(f"    H     = {p['H'].item():.4f}")
        print(f"    η     = {p['eta'].item():.4f}")
        print(f"    α     = [{', '.join(f'{a:.4f}' for a in p['alpha'].numpy())}]")
        print(f"    ρ₀    = [{', '.join(f'{r:.3f}' for r in p['rho0'].numpy())}]")

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

def mc_diagnostics(params, mkt, N_paths=100_000, M=100, seed=42,
                   swaption_keys=None, label=""):
    """Full MC-based calibration report with high path count."""
    header = f"MC REPORT — {label}" if label else "MC-BASED CALIBRATION REPORT"
    print("\n" + "=" * 72)
    print(header)
    print(f"({N_paths:,} paths, {M} time steps)")
    print("=" * 72)

    report = print_calibration_report(
        params, mkt,
        variance_curve_mode="full",
        N_paths=N_paths,
        M=M,
        seed=seed,
        swaption_keys=swaption_keys,
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

    by_expiry = {}
    by_tenor = {}
    all_sq, all_abs, all_signed = [], [], []

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

        # Accumulate
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
                     suptitle=None, test_keys=None):
    """
    Generate smile-fit plots for all swaptions in mkt.

    Reusable for both in-sample and out-of-sample evaluation.
    Test swaptions (if given) are colour-coded blue with [TEST] tag.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping plots")
        return

    test_set = set(test_keys) if test_keys else set()

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

                is_test = key in test_set
                model_color = "b^--" if is_test else "r^--"
                ax.plot(offsets, mkt_ivs, "ko-", markersize=3, label="Market")
                ax.plot(offsets, mod_ivs, model_color, markersize=3, label="Model")

                tag = " [TEST]" if is_test else ""
                ax.set_title(f"{key[0]:.0f}Y × {key[1]:.0f}Y{tag}", fontsize=10)
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


def save_plots(params, mkt, history, config, history2=None,
               train_keys=None, test_keys=None):
    """
    Generate and save calibration diagnostic plots:
      1. Convergence (loss & learning rate)
      2. Smile fits (train/test colour-coded)
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

    # --- 2. Smile fits (train/test colour-coded) ---
    save_smile_plots(
        params, mkt, config,
        filename="amcc_smile_fits.png",
        suptitle=f"Smile fits — train (red) / test (blue)  [{config.get('date', '')}]",
        test_keys=test_keys,
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

    # ==================================================================
    # LOAD DATA
    # ==================================================================
    print("Loading market data...")
    print(f"  Date: {cfg['date']}")
    mkt = load_market_data(
        cfg["data_file"],
        subset=cfg["subset"],
        date=cfg["date"],
        device=cfg["device"],
    )
    print_market_summary(mkt)

    # ==================================================================
    # TRAIN / TEST SPLIT
    # ==================================================================
    all_keys = sorted(mkt.swaptions.keys())
    test_keys = [k for k in cfg["test_keys"] if k in mkt.swaptions]
    train_keys = [k for k in all_keys if k not in test_keys]

    print("\n" + "=" * 60)
    print("TRAIN / TEST SPLIT")
    print("=" * 60)
    print(f"\n  Train ({len(train_keys)} swaptions — used in optimizer):")
    for k in train_keys:
        print(f"    {k[0]:.0f}Y × {k[1]:.0f}Y")
    print(f"\n  Test ({len(test_keys)} swaptions — held out):")
    for k in test_keys:
        print(f"    {k[0]:.0f}Y × {k[1]:.0f}Y")

    # Sanity: all 1Y-tenor swaptions must be in training for α init
    test_1y = [k for k in test_keys if k[1] == 1]
    if test_1y:
        print(f"\n  ⚠ WARNING: 1Y-tenor swaptions in test set: {test_1y}")
        print(f"    Moving to train set (needed for α matching).")
        for k in test_1y:
            test_keys.remove(k)
            train_keys.append(k)
        train_keys.sort()

    # ==================================================================
    # INITIALISATION
    # ==================================================================
    print("\n" + "#" * 60)
    print("# INITIALISATION")
    print("#" * 60)

    params = initialise_params(mkt, H_init=0.20, eta_init=2.3)

    # ==================================================================
    # AMCC CALIBRATION (train set only)
    # ==================================================================
    print("\n" + "#" * 60)
    print("# AMCC CALIBRATION (train set only)")
    print("#" * 60)

    if cfg["mode"] == "hybrid":
        hcfg = cfg["hybrid"]
        print(f"\nMode: hybrid (BLP κ={hcfg['kappa']})")
        print(f"  {hcfg['iterations']} iterations, {hcfg['N_paths']:,} paths, "
              f"{hcfg['M']} steps, lr={hcfg['lr']}")
        print(f"  Training on {len(train_keys)} swaptions, "
              f"holding out {len(test_keys)}")

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
            swaption_keys=train_keys,
            scheduler_type=hcfg.get("scheduler", "cosine"),
            warmup_steps=hcfg.get("warmup_steps", 50),
            early_stop_patience=hcfg.get("early_stop_patience",
                                         cfg.get("early_stop_patience")),
            early_stop_tol=cfg.get("early_stop_tol", 1e-4),
        )

        if result["best_state"] is not None:
            params.load_state_dict(result["best_state"])

        with torch.no_grad():
            H_calibrated = params.get_H().item()
        print(f"\nHybrid calibration complete. H = {H_calibrated:.4f}")

    else:
        print(f"\nMode: two_stage")
        print(f"  Training on {len(train_keys)} swaptions, "
              f"holding out {len(test_keys)}")

        result = calibrate_two_stage(
            params, mkt,
            stage1_iterations=cfg["stage1"]["iterations"],
            stage1_lr=cfg["stage1"]["lr"],
            stage1_N_paths=cfg["stage1"]["N_paths"],
            stage1_M=cfg["stage1"]["M"],
            stage1_keys=train_keys,
            stage2_iterations=cfg["stage2"]["iterations"],
            stage2_lr=cfg["stage2"]["lr"],
            stage2_N_paths=cfg["stage2"]["N_paths"],
            stage2_M=cfg["stage2"]["M"],
            stage2_variance_mode=cfg["stage2"]["variance_mode"],
            stage2_keys=train_keys,
            stage2_scheduler=cfg["stage2"].get("scheduler", "cosine"),
            stage2_warmup_steps=cfg["stage2"].get("warmup_steps", 30),
            stage2_cosine_power=cfg["stage2"].get("cosine_power", 0.5),
            use_crn=True,
            crn_seed=cfg["crn_seed"],
            log_every=20,
            early_stop_patience=cfg.get("early_stop_patience", 150),
            early_stop_tol=cfg.get("early_stop_tol", 1e-4),
        )

        with torch.no_grad():
            H_calibrated = params.get_H().item()

    # ==================================================================
    # CALIBRATED PARAMETERS
    # ==================================================================
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

    # ==================================================================
    # TRAIN SET MC DIAGNOSTICS
    # ==================================================================
    print("\n" + "#" * 60)
    print(f"# TRAIN SET MC DIAGNOSTICS ({len(train_keys)} swaptions)")
    print("#" * 60)

    report_train = mc_diagnostics(
        params, mkt,
        N_paths=cfg["diag_N_paths"],
        M=cfg["diag_M"],
        swaption_keys=train_keys,
        label=f"TRAIN ({len(train_keys)} swaptions)",
    )
    print_accuracy_summary(report_train, mkt, label="TRAIN")

    # ==================================================================
    # TEST SET MC DIAGNOSTICS (held out — no re-matching)
    # ==================================================================
    print("\n" + "#" * 60)
    print(f"# TEST SET MC DIAGNOSTICS ({len(test_keys)} swaptions — held out)")
    print("#" * 60)

    report_test = mc_diagnostics(
        params, mkt,
        N_paths=cfg["diag_N_paths"],
        M=cfg["diag_M"],
        swaption_keys=test_keys,
        label=f"TEST ({len(test_keys)} swaptions, held out)",
    )
    print_accuracy_summary(report_test, mkt, label="TEST (held out)")

    # ==================================================================
    # SMILE COMPARISONS — TRAIN
    # ==================================================================
    print("\n" + "#" * 60)
    print("# SMILE COMPARISONS — TRAIN")
    print("#" * 60)

    for key in train_keys:
        if key in mkt.swaptions:
            print_smile_comparison(
                params, mkt.swaptions[key], mkt,
                variance_curve_mode="full",
                N_paths=cfg["diag_N_paths"],
                M=cfg["diag_M"],
            )

    # ==================================================================
    # SMILE COMPARISONS — TEST (held out)
    # ==================================================================
    print("\n" + "#" * 60)
    print("# SMILE COMPARISONS — TEST (held out)")
    print("#" * 60)

    for key in test_keys:
        if key in mkt.swaptions:
            print_smile_comparison(
                params, mkt.swaptions[key], mkt,
                variance_curve_mode="full",
                N_paths=cfg["diag_N_paths"],
                M=cfg["diag_M"],
            )

    # ==================================================================
    # PLOTS
    # ==================================================================
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
        train_keys=train_keys,
        test_keys=test_keys,
    )

    # ==================================================================
    # SAVE RESULTS
    # ==================================================================
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
        "date": cfg["date"],
        "train_keys": train_keys,
        "test_keys": test_keys,
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