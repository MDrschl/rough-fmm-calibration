"""
calibration_roughness.py
Roughness Ablation Study: H < 1/2 vs H = 1/2
==============================================

This script answers the question: how much does roughness contribute to
calibration quality?  It calibrates the Mapped Rough SABR FMM twice --
once with H free (the full rough model) and once with H pinned at 0.5
(the Markovian special case) -- then compares the fits on the same
in-sample data.

When H = 1/2, the fractional Brownian motion degenerates to standard
Brownian motion.  The variance process becomes Markovian (no memory,
no path dependence), but the full multi-factor FMM structure survives:
N forward rates, the correlation matrix rho_{ij}, the rho_0 vector,
and the alpha term structure.  The only thing lost is the power-law
kernel that creates long-range dependence in variance.

The comparison breaks down the improvement (or lack thereof) by:
  - per-swaption RMSE
  - expiry bucket (short vs long-dated)
  - tenor bucket (single-rate vs multi-rate)
  - moneyness bucket (wings vs ATM)
  - left wing vs right wing (skew structure)

This is the key finding for the thesis: if roughness buys 20+ bp RMSE,
it justifies the added model complexity.  If it buys only 5 bp, the
multi-factor correlation structure is doing the heavy lifting.

Usage:
    python calibration_roughness.py

Requires:
    - main.py (model and infrastructure)
    - usd_swaption_data.pkl
    - amcc_calibration_results.pt (optional: loads existing rough results)
"""

import sys
import os
import time
import copy
import numpy as np
import torch

from main import (
    MappedRoughSABRParams,
    load_market_data,
    match_all_alphas,
    calibrate,
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

CONFIG = {
    # Data
    "data_file": "usd_swaption_data.pkl",
    "subset": "joint_all_smiles",
    "date": "2024-12-09",
    "device": "cpu",

    # Rough model: load from existing calibration if available
    "rough_results_file": "amcc_calibration_results.pt",

    # H=0.5 calibration settings
    # Since H is fixed, we skip Stage 1 (approximate scheme for H gradients)
    # and go directly to exact Cholesky calibration.
    "h05": {
        "eta_init": 1.5,          # starting eta (= kappa when H=0.5)
        "iterations": 1200,       # more iterations to compensate for lost DOF
        "lr": 5e-3,
        "N_paths": 30_000,
        "M": 50,
        "variance_mode": "full",
        "scheduler": "cosine",
        "warmup_steps": 50,
        "cosine_power": 0.5,
        "rematch_alpha": True,
        "early_stop_patience": 200,
    },

    # Diagnostics (same for both models -- apples-to-apples)
    "diag_N_paths": 100_000,
    "diag_M": 100,

    # Reproducibility
    "crn_seed": 42,
}


# =============================================================================
# Helpers (shared with calibration.py)
# =============================================================================

def _softplus_inv(a):
    """Numerically stable inverse of softplus."""
    a = float(a)
    if a > 20.0:
        return a
    return a + np.log(-np.expm1(-a))


def _interpolate_alpha(alpha, matched_indices):
    """Fill unmatched alpha by linear interpolation, flat extrapolation."""
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


# =============================================================================
# Initialise H=0.5 model
# =============================================================================

def initialise_h05(mkt, eta_init=1.5):
    """
    Create parameter module with H fixed at 0.5.

    When H = 0.5, the rough Bergomi variance process degenerates to
    geometric Brownian motion -- this is the Markovian special case.
    The deterministic ATM formula sigma_ATM = alpha * sqrt(G) is more
    accurate here than at H = 0.2, because the variance-of-variance
    growth is smoother.
    """
    params = MappedRoughSABRParams(N=mkt.N, device=mkt.device)
    params.set_H(0.5)
    params.fix_H()          # freeze H for the entire calibration
    params.set_eta(eta_init)

    with torch.no_grad():
        p = params()

        # Match alpha analytically on 1Y-tenor swaptions
        smile_keys_1y = sorted([k for k in mkt.swaptions.keys() if k[1] == 1])
        alpha_matched = match_all_alphas(
            mkt, p["H"], p["eta"], p["rho0"], p["rho"],
            alpha_init=p["alpha"],
            smile_keys=smile_keys_1y,
            variance_curve_mode="simplified",
            method="formula",
        )

        # Identify matched indices and interpolate
        matched_indices = []
        for key in smile_keys_1y:
            swn = mkt.swaptions[key]
            if swn.J - swn.I == 1:
                matched_indices.append(swn.I)

        alpha_final = _interpolate_alpha(alpha_matched, matched_indices)

        for j in range(mkt.N):
            a = alpha_final[j].item()
            params.alpha_tilde.data[j] = _softplus_inv(a)

    with torch.no_grad():
        p = params()
        print(f"\n  H=0.5 initialisation:")
        print(f"    H     = {p['H'].item():.4f}  [FIXED]")
        print(f"    eta   = {p['eta'].item():.4f}  (= kappa when H=0.5)")
        print(f"    alpha = [{', '.join(f'{a:.4f}' for a in p['alpha'].numpy())}]")

    return params


# =============================================================================
# MC diagnostics (reusable)
# =============================================================================

def run_diagnostics(params, mkt, label="", N_paths=100_000, M=100, seed=42):
    """Run MC diagnostics and return the report dict."""
    print(f"\n{'=' * 72}")
    print(f"MC DIAGNOSTICS -- {label}")
    print(f"{'=' * 72}")

    report = print_calibration_report(
        params, mkt,
        variance_curve_mode="full",
        N_paths=N_paths,
        M=M,
        seed=seed,
    )
    return report


# =============================================================================
# Head-to-head comparison
# =============================================================================

def compare_reports(report_rough, report_h05, mkt):
    """
    Detailed head-to-head comparison of rough vs H=0.5 calibrations.

    Breaks down by: per-swaption, expiry, tenor, moneyness bucket,
    wing asymmetry.
    """
    per_r = report_rough.get("per_swaption", {})
    per_h = report_h05.get("per_swaption", {})
    keys = sorted(set(per_r.keys()) & set(per_h.keys()))

    if not keys:
        print("  No common swaptions to compare.")
        return {}

    # ================================================================
    # 1. Per-swaption RMSE comparison
    # ================================================================
    print("\n" + "=" * 100)
    print("PER-SWAPTION COMPARISON: Rough (H free) vs Markovian (H = 0.5)")
    print("=" * 100)

    print(f"\n  {'Swaption':>10s}  {'Rough':>8s}  {'H=0.5':>8s}  "
          f"{'D RMSE':>8s}  {'Rough':>8s}  {'H=0.5':>8s}  "
          f"{'D ATM':>8s}  {'Winner':>8s}")
    print(f"  {'':>10s}  {'RMSE':>8s}  {'RMSE':>8s}  "
          f"{'(bp)':>8s}  {'ATM':>8s}  {'ATM':>8s}  "
          f"{'(bp)':>8s}  {'':>8s}")
    print("  " + "-" * 90)

    # Collectors
    by_expiry = {}
    by_tenor = {}
    moneyness_buckets = {
        "deep_ITM": {"rough": [], "h05": []},
        "ITM":      {"rough": [], "h05": []},
        "ATM":      {"rough": [], "h05": []},
        "OTM":      {"rough": [], "h05": []},
        "deep_OTM": {"rough": [], "h05": []},
    }
    wing_data = {
        "left_rough": [], "left_h05": [],
        "right_rough": [], "right_h05": [],
    }

    all_rough_sq, all_h05_sq = [], []
    rough_wins, h05_wins = 0, 0
    comparison_rows = []

    for key in keys:
        swn = mkt.swaptions[key]
        res_r = per_r[key]
        res_h = per_h[key]
        expiry, tenor = key

        valid_r = ~torch.isnan(res_r["iv_errors"])
        valid_h = ~torch.isnan(res_h["iv_errors"])
        valid = valid_r & valid_h
        if valid.sum() == 0:
            continue

        err_r = res_r["iv_errors"][valid] * 10000  # bp
        err_h = res_h["iv_errors"][valid] * 10000

        rmse_r = np.sqrt((err_r.numpy() ** 2).mean())
        rmse_h = np.sqrt((err_h.numpy() ** 2).mean())
        delta_rmse = rmse_h - rmse_r  # positive = rough is better

        # ATM errors
        atm_mask = (swn.strikes - swn.S0).abs() < ATM_TOL
        atm_r = (res_r["iv_errors"][atm_mask][0].item() * 10000
                 if atm_mask.sum() > 0 else float('nan'))
        atm_h = (res_h["iv_errors"][atm_mask][0].item() * 10000
                 if atm_mask.sum() > 0 else float('nan'))
        delta_atm = abs(atm_h) - abs(atm_r)

        winner = "Rough" if rmse_r < rmse_h else "H=0.5"
        if rmse_r < rmse_h:
            rough_wins += 1
        else:
            h05_wins += 1

        print(f"  {expiry:.0f}Y x {tenor:.0f}Y  "
              f"{rmse_r:8.1f}  {rmse_h:8.1f}  {delta_rmse:+8.1f}  "
              f"{atm_r:+8.1f}  {atm_h:+8.1f}  {delta_atm:+8.1f}  "
              f"{winner:>8s}")

        comparison_rows.append({
            "key": key, "rmse_rough": rmse_r, "rmse_h05": rmse_h,
            "delta": delta_rmse, "atm_rough": atm_r, "atm_h05": atm_h,
        })

        # Accumulate
        sq_r = (err_r.numpy() ** 2).tolist()
        sq_h = (err_h.numpy() ** 2).tolist()
        all_rough_sq.extend(sq_r)
        all_h05_sq.extend(sq_h)

        by_expiry.setdefault(expiry, {"rough_sq": [], "h05_sq": []})
        by_expiry[expiry]["rough_sq"].extend(sq_r)
        by_expiry[expiry]["h05_sq"].extend(sq_h)

        by_tenor.setdefault(tenor, {"rough_sq": [], "h05_sq": []})
        by_tenor[tenor]["rough_sq"].extend(sq_r)
        by_tenor[tenor]["h05_sq"].extend(sq_h)

        # Per-strike: moneyness buckets and wing analysis
        offsets = ((swn.strikes - swn.S0) * 10000).numpy()  # bp
        for i_s in range(swn.n_strikes):
            if not valid[i_s]:
                continue
            off = offsets[i_s]
            e_r = res_r["iv_errors"][i_s].item() * 10000
            e_h = res_h["iv_errors"][i_s].item() * 10000

            if off <= -100:
                bucket = "deep_ITM"
            elif off <= -25:
                bucket = "ITM"
            elif off < 25:
                bucket = "ATM"
            elif off < 100:
                bucket = "OTM"
            else:
                bucket = "deep_OTM"
            moneyness_buckets[bucket]["rough"].append(e_r ** 2)
            moneyness_buckets[bucket]["h05"].append(e_h ** 2)

            if off < -ATM_TOL * 10000:
                wing_data["left_rough"].append(e_r)
                wing_data["left_h05"].append(e_h)
            elif off > ATM_TOL * 10000:
                wing_data["right_rough"].append(e_r)
                wing_data["right_h05"].append(e_h)

    # ---- Totals ----
    print("  " + "-" * 90)
    total_r = np.sqrt(np.mean(all_rough_sq))
    total_h = np.sqrt(np.mean(all_h05_sq))
    delta_total = total_h - total_r
    print(f"  {'TOTAL':>10s}  {total_r:8.1f}  {total_h:8.1f}  "
          f"{delta_total:+8.1f}  {'':>8s}  {'':>8s}  {'':>8s}  "
          f"{'Rough' if total_r < total_h else 'H=0.5':>8s}")
    print(f"\n  Rough wins {rough_wins}/{len(keys)} swaptions, "
          f"H=0.5 wins {h05_wins}/{len(keys)}")

    # ================================================================
    # 2. By expiry
    # ================================================================
    print(f"\n  {'BREAKDOWN BY EXPIRY':=^70}")
    print(f"  {'Expiry':>8s}  {'Rough':>8s}  {'H=0.5':>8s}  "
          f"{'D RMSE':>8s}  {'Rel D':>8s}")
    print(f"  {'':>8s}  {'RMSE':>8s}  {'RMSE':>8s}  "
          f"{'(bp)':>8s}  {'(%)':>8s}")
    print("  " + "-" * 50)

    for exp in sorted(by_expiry.keys()):
        d = by_expiry[exp]
        r = np.sqrt(np.mean(d["rough_sq"]))
        h = np.sqrt(np.mean(d["h05_sq"]))
        delta = h - r
        rel = delta / h * 100 if h > 0 else 0
        print(f"  {exp:>6.0f}Y  {r:8.1f}  {h:8.1f}  "
              f"{delta:+8.1f}  {rel:+7.1f}%")

    # ================================================================
    # 3. By tenor
    # ================================================================
    print(f"\n  {'BREAKDOWN BY TENOR':=^70}")
    print(f"  {'Tenor':>8s}  {'Rough':>8s}  {'H=0.5':>8s}  "
          f"{'D RMSE':>8s}  {'Rel D':>8s}  {'#swn':>6s}")
    print(f"  {'':>8s}  {'RMSE':>8s}  {'RMSE':>8s}  "
          f"{'(bp)':>8s}  {'(%)':>8s}  {'':>6s}")
    print("  " + "-" * 55)

    for ten in sorted(by_tenor.keys()):
        d = by_tenor[ten]
        r = np.sqrt(np.mean(d["rough_sq"]))
        h = np.sqrt(np.mean(d["h05_sq"]))
        delta = h - r
        rel = delta / h * 100 if h > 0 else 0
        n_swn = len([k for k in keys if k[1] == ten])
        print(f"  {ten:>6.0f}Y  {r:8.1f}  {h:8.1f}  "
              f"{delta:+8.1f}  {rel:+7.1f}%  {n_swn:>6d}")

    # ================================================================
    # 4. By moneyness
    # ================================================================
    print(f"\n  {'BREAKDOWN BY MONEYNESS':=^70}")
    bucket_labels = {
        "deep_ITM": "Deep ITM (<=-100bp)",
        "ITM":      "ITM (-100 to -25bp)",
        "ATM":      "ATM (+/-25bp)",
        "OTM":      "OTM (+25 to +100bp)",
        "deep_OTM": "Deep OTM (>=+100bp)",
    }
    print(f"  {'Bucket':>22s}  {'Rough':>8s}  {'H=0.5':>8s}  "
          f"{'D RMSE':>8s}  {'Rel D':>8s}  {'#pts':>6s}")
    print("  " + "-" * 65)

    for bk in ["deep_ITM", "ITM", "ATM", "OTM", "deep_OTM"]:
        r_sq = moneyness_buckets[bk]["rough"]
        h_sq = moneyness_buckets[bk]["h05"]
        if not r_sq:
            continue
        r = np.sqrt(np.mean(r_sq))
        h = np.sqrt(np.mean(h_sq))
        delta = h - r
        rel = delta / h * 100 if h > 0 else 0
        print(f"  {bucket_labels[bk]:>22s}  {r:8.1f}  {h:8.1f}  "
              f"{delta:+8.1f}  {rel:+7.1f}%  {len(r_sq):>6d}")

    # ================================================================
    # 5. Wing asymmetry
    # ================================================================
    print(f"\n  {'WING BIAS COMPARISON':=^70}")
    for side, label in [("left", "Left wing (K < S0)"),
                        ("right", "Right wing (K > S0)")]:
        r_arr = np.array(wing_data[f"{side}_rough"])
        h_arr = np.array(wing_data[f"{side}_h05"])
        if len(r_arr) == 0:
            continue
        r_bias = r_arr.mean()
        h_bias = h_arr.mean()
        r_rmse = np.sqrt((r_arr ** 2).mean())
        h_rmse = np.sqrt((h_arr ** 2).mean())
        print(f"  {label}:")
        print(f"    Rough:  bias={r_bias:+.1f}bp  RMSE={r_rmse:.1f}bp")
        print(f"    H=0.5:  bias={h_bias:+.1f}bp  RMSE={h_rmse:.1f}bp")
        print(f"    D RMSE: {h_rmse - r_rmse:+.1f}bp")

    return {
        "total_rmse_rough": total_r,
        "total_rmse_h05": total_h,
        "delta_total": delta_total,
        "by_expiry": by_expiry,
        "by_tenor": by_tenor,
        "moneyness": moneyness_buckets,
        "per_swaption": comparison_rows,
        "rough_wins": rough_wins,
        "h05_wins": h05_wins,
    }


# =============================================================================
# Comparison plots
# =============================================================================

def save_comparison_plots(params_rough, params_h05, mkt, cfg):
    """Side-by-side smile fits: rough vs H=0.5."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping plots")
        return

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
    fig.suptitle(
        "Roughness ablation: Rough (H free) vs Markovian (H = 0.5)",
        fontsize=13, fontweight="bold", y=1.01,
    )

    cholesky_cache_r, cholesky_cache_h = {}, {}

    with torch.no_grad():
        for row_idx, mat in enumerate(maturities):
            keys_row = keys_by_maturity[mat]
            for col_idx, key in enumerate(keys_row):
                ax = axes[row_idx, col_idx]
                swn = mkt.swaptions[key]
                strikes_pct = swn.strikes.numpy() * 100
                mkt_ivs = swn.ivs_black.numpy() * 100

                # Rough model
                torch.manual_seed(cfg["crn_seed"])
                S_T_r = simulate_swaption(
                    params_rough, swn, mkt,
                    N_paths=50_000, M=80, use_exact=True,
                    variance_curve_mode="full",
                    cholesky_cache=cholesky_cache_r,
                )
                mc_r = compute_swaption_prices(S_T_r, swn)
                mod_r = mc_prices_to_black_iv(mc_r, swn).numpy() * 100

                # H=0.5 model
                torch.manual_seed(cfg["crn_seed"])
                S_T_h = simulate_swaption(
                    params_h05, swn, mkt,
                    N_paths=50_000, M=80, use_exact=True,
                    variance_curve_mode="full",
                    cholesky_cache=cholesky_cache_h,
                )
                mc_h = compute_swaption_prices(S_T_h, swn)
                mod_h = mc_prices_to_black_iv(mc_h, swn).numpy() * 100

                ax.plot(strikes_pct, mkt_ivs, "o-", color="black",
                        markersize=3, linewidth=0.5, label="Market")
                ax.plot(strikes_pct, mod_r, "x-", color="blue",
                        markersize=4, linewidth=0.5, label="Rough")
                ax.plot(strikes_pct, mod_h, "x-", color="red",
                        markersize=4, linewidth=0.5, label="H=0.5")
                ax.set_title(f"{key[0]:.0f}Y x {key[1]:.0f}Y", fontsize=10)
                ax.set_xlabel("Strike (%)", fontsize=8)
                ax.set_ylabel("IV (%)", fontsize=8)
                ax.set_ylim(bottom=0)
                ax.tick_params(labelsize=7)
                ax.grid(True, alpha=0.3)
                if row_idx == 0 and col_idx == 0:
                    ax.legend(fontsize=7)

            for col_idx in range(len(keys_row), max_per_row):
                axes[row_idx, col_idx].set_visible(False)

    plt.tight_layout()
    plt.savefig("roughness_ablation_smiles.png", dpi=150)
    print("Saved: roughness_ablation_smiles.png")
    plt.close(fig)

    # ---- RMSE bar chart ----
    fig, ax = plt.subplots(figsize=(12, 5))
    all_keys = sorted(set(mkt.swaptions.keys()))
    labels = [f"{k[0]:.0f}Y x {k[1]:.0f}Y" for k in all_keys]

    rmse_r_list, rmse_h_list = [], []
    for key in all_keys:
        swn = mkt.swaptions[key]

        res_r = compute_model_smile(
            params_rough, swn, mkt,
            variance_curve_mode="full",
            N_paths=50_000, M=80, seed=cfg["crn_seed"],
        )
        res_h = compute_model_smile(
            params_h05, swn, mkt,
            variance_curve_mode="full",
            N_paths=50_000, M=80, seed=cfg["crn_seed"],
        )

        valid = ~torch.isnan(res_r["iv_errors"]) & ~torch.isnan(res_h["iv_errors"])
        if valid.sum() > 0:
            rmse_r_list.append(
                np.sqrt(((res_r["iv_errors"][valid] * 10000).numpy() ** 2).mean()))
            rmse_h_list.append(
                np.sqrt(((res_h["iv_errors"][valid] * 10000).numpy() ** 2).mean()))
        else:
            rmse_r_list.append(0)
            rmse_h_list.append(0)

    x = np.arange(len(labels))
    w = 0.35
    ax.bar(x - w / 2, rmse_r_list, w, label="Rough (H free)", color="steelblue")
    ax.bar(x + w / 2, rmse_h_list, w, label="H = 0.5", color="indianred")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("RMSE (bp)")
    ax.set_title("Per-swaption RMSE: Rough vs Markovian SABR FMM")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig("roughness_ablation_rmse_bars.png", dpi=150)
    print("Saved: roughness_ablation_rmse_bars.png")
    plt.close(fig)


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":

    t_start = time.time()
    cfg = CONFIG

    # ----------------------------------------------------------------
    # 1. Load market data
    # ----------------------------------------------------------------
    print("=" * 72)
    print("ROUGHNESS ABLATION STUDY: H < 1/2  vs  H = 1/2")
    print("=" * 72)

    print(f"\nLoading market data ({cfg['date']})...")
    mkt = load_market_data(
        cfg["data_file"],
        subset=cfg["subset"],
        date=cfg["date"],
        device=cfg["device"],
    )
    print_market_summary(mkt)

    # ----------------------------------------------------------------
    # 2. Load or reconstruct rough model
    # ----------------------------------------------------------------
    print("\n" + "#" * 60)
    print("# MODEL A: Rough (H free)")
    print("#" * 60)

    rough_file = cfg["rough_results_file"]
    if os.path.exists(rough_file):
        print(f"\nLoading existing rough calibration from {rough_file}...")
        saved = torch.load(rough_file, map_location=cfg["device"],
                           weights_only=False)

        params_rough = MappedRoughSABRParams(N=mkt.N, device=cfg["device"])
        params_rough.load_state_dict(saved["params_state_dict"])
        params_rough.fix_H()

        with torch.no_grad():
            p_r = params_rough()
            H_rough = p_r["H"].item()
            eta_rough = p_r["eta"].item()
            kappa_rough = eta_rough * np.sqrt(2.0 * H_rough)

        print(f"  H     = {H_rough:.4f}")
        print(f"  eta   = {eta_rough:.4f}")
        print(f"  kappa = {kappa_rough:.4f}")
        print(f"  alpha = [{', '.join(f'{a:.4f}' for a in p_r['alpha'].numpy())}]")
    else:
        print(f"\n  {rough_file} not found.")
        print(f"  Run calibration.py first, then re-run this script.")
        sys.exit(1)

    # ----------------------------------------------------------------
    # 3. Calibrate H=0.5 model
    # ----------------------------------------------------------------
    print("\n" + "#" * 60)
    print("# MODEL B: Markovian special case (H = 0.5)")
    print("#" * 60)

    h05cfg = cfg["h05"]
    params_h05 = initialise_h05(mkt, eta_init=h05cfg["eta_init"])

    print(f"\n  Calibrating with H = 0.5 fixed...")
    print(f"  {h05cfg['iterations']} iterations, {h05cfg['N_paths']:,} paths, "
          f"{h05cfg['M']} steps, lr={h05cfg['lr']}")
    print("=" * 60)

    result_h05 = calibrate(
        params_h05, mkt,
        n_iterations=h05cfg["iterations"],
        lr=h05cfg["lr"],
        N_paths=h05cfg["N_paths"],
        M=h05cfg["M"],
        use_exact=True,
        variance_curve_mode=h05cfg["variance_mode"],
        use_crn=True,
        crn_seed=cfg["crn_seed"],
        log_every=20,
        swaption_keys=None,            # all swaptions
        scheduler_type=h05cfg["scheduler"],
        warmup_steps=h05cfg["warmup_steps"],
        cosine_power=h05cfg["cosine_power"],
        early_stop_patience=h05cfg["early_stop_patience"],
        early_stop_tol=1e-4,
        rematch_alpha=h05cfg.get("rematch_alpha", True),
    )

    if result_h05["best_state"] is not None:
        params_h05.load_state_dict(result_h05["best_state"])

    with torch.no_grad():
        p_h = params_h05()
        print(f"\n  H=0.5 calibration complete.")
        print(f"    H   = {p_h['H'].item():.4f}  [fixed]")
        print(f"    eta = {p_h['eta'].item():.4f}  (= kappa)")
        print(f"    alpha = [{', '.join(f'{a:.4f}' for a in p_h['alpha'].numpy())}]")
        print(f"    rho0  = [{', '.join(f'{r:.3f}' for r in p_h['rho0'].numpy())}]")

    # ----------------------------------------------------------------
    # 4. MC diagnostics -- both models, same settings
    # ----------------------------------------------------------------
    print("\n" + "#" * 60)
    print("# DIAGNOSTICS")
    print("#" * 60)

    report_rough = run_diagnostics(
        params_rough, mkt, label="Rough (H free)",
        N_paths=cfg["diag_N_paths"], M=cfg["diag_M"], seed=cfg["crn_seed"],
    )

    report_h05 = run_diagnostics(
        params_h05, mkt, label="Markovian (H = 0.5)",
        N_paths=cfg["diag_N_paths"], M=cfg["diag_M"], seed=cfg["crn_seed"],
    )

    # ----------------------------------------------------------------
    # 5. Head-to-head comparison
    # ----------------------------------------------------------------
    print("\n" + "#" * 60)
    print("# HEAD-TO-HEAD COMPARISON")
    print("#" * 60)

    comparison = compare_reports(report_rough, report_h05, mkt)

    # ----------------------------------------------------------------
    # 6. Smile comparisons (selected swaptions)
    # ----------------------------------------------------------------
    print("\n" + "#" * 60)
    print("# SMILE COMPARISONS (selected swaptions)")
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
            print(f"\n--- {key[0]:.0f}Y x {key[1]:.0f}Y ---")
            print("  [Rough]")
            print_smile_comparison(
                params_rough, mkt.swaptions[key], mkt,
                variance_curve_mode="full",
                N_paths=cfg["diag_N_paths"], M=cfg["diag_M"],
            )
            print("  [H=0.5]")
            print_smile_comparison(
                params_h05, mkt.swaptions[key], mkt,
                variance_curve_mode="full",
                N_paths=cfg["diag_N_paths"], M=cfg["diag_M"],
            )

    # ----------------------------------------------------------------
    # 7. Plots
    # ----------------------------------------------------------------
    print("\n" + "#" * 60)
    print("# GENERATING PLOTS")
    print("#" * 60)

    save_comparison_plots(params_rough, params_h05, mkt, cfg)

    # ----------------------------------------------------------------
    # 8. Summary verdict
    # ----------------------------------------------------------------
    print("\n" + "#" * 60)
    print("# SUMMARY")
    print("#" * 60)

    delta = comparison["delta_total"]
    total_r = comparison["total_rmse_rough"]
    total_h = comparison["total_rmse_h05"]
    rel_improvement = delta / total_h * 100 if total_h > 0 else 0

    print(f"\n  Rough model:    H = {H_rough:.4f},  RMSE = {total_r:.1f} bp")
    print(f"  Markovian:      H = 0.5000,  RMSE = {total_h:.1f} bp")
    print(f"\n  Roughness improvement: {delta:+.1f} bp  ({rel_improvement:+.1f}% relative)")
    print(f"  Rough wins {comparison['rough_wins']}/{comparison['rough_wins'] + comparison['h05_wins']} swaptions")

    if delta > 20:
        print("\n  Verdict: Roughness contributes meaningfully (>20bp).")
        print("  The H < 1/2 specification captures smile dynamics that")
        print("  the Markovian special case of the FMM cannot.")
    elif delta > 5:
        print("\n  Verdict: Moderate roughness contribution (5-20bp).")
        print("  Both roughness and the multi-factor correlation structure")
        print("  contribute, but roughness alone is not transformative.")
    else:
        print("\n  Verdict: Roughness contribution is small (<5bp).")
        print("  The multi-factor FMM correlation structure does the")
        print("  heavy lifting; H < 1/2 is a refinement, not the main driver.")

    # ----------------------------------------------------------------
    # 9. Save results
    # ----------------------------------------------------------------
    elapsed = time.time() - t_start

    results = {
        "H_rough": H_rough,
        "eta_rough": eta_rough,
        "H_h05": 0.5,
        "eta_h05": p_h["eta"].item(),
        "total_rmse_rough": total_r,
        "total_rmse_h05": total_h,
        "delta_rmse": delta,
        "rough_wins": comparison["rough_wins"],
        "h05_wins": comparison["h05_wins"],
        "per_swaption": comparison["per_swaption"],
        "params_h05_state_dict": params_h05.state_dict(),
        "h05_history": result_h05["history"],
        "elapsed_seconds": elapsed,
    }
    torch.save(results, "roughness_ablation_results.pt")
    print(f"\nResults saved to roughness_ablation_results.pt")
    print(f"Total elapsed time: {elapsed:.1f}s ({elapsed / 60:.1f} min)")
    print("\nDone.")