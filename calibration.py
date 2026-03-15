import sys
import os
import time
import copy
import numpy as np
import torch

import main
from main import (
    DTYPE,
    set_dtype,
    MappedRoughSABRParams,
    load_market_data,
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
    # --- Data ---
    "data_file": "usd_swaption_data.pkl",
    "subset": "joint_all_smiles",
    "in_sample_date": "2024-12-09",
    "out_sample_date": "2024-12-10",
    "device": "cpu",
    "dtype": "float64",

    # --- Calibration mode ---
    #   "hybrid"            — Mode A: single-stage hybrid, H differentiable
    #   "hybrid_two_stage"  — Mode C: two-stage hybrid, H frozen in S2
    #   "two_stage"         — Mode B: approx S1 → exact Cholesky S2
    #   "hybrid_exact"      — Mode G: hybrid S1 → exact Cholesky S2
    #   "roughness"         — Mode E: ablation study, H free vs H = 0.5
    #   "cross"             — Mode F: train/test split cross-validation
    "mode": "hybrid_two_stage",

    "hybrid": {
        "iterations": 800,
        "lr": 3e-3,
        "N_paths": 30_000,
        "M": 100,
        "kappa": 2,
        "variance_mode": "full",
        "keys": None,
        "scheduler": "cosine",
        "warmup_steps": 50,
        "cosine_power": 0.5,
        "early_stop_patience": 200,
        "H_lr_factor": 1.5,
        "grad_clip_norm": 1.0,
    },

    "hybrid_two_stage": {
        "stage1": {
            "iterations": 400,
            "lr": 5e-3,
            "N_paths": 20_000,
            "M": 50,
            "kappa": 2,
            "variance_mode": "simplified",
            "keys": None,
            "scheduler": "cosine",
            "warmup_steps": 30,
            "cosine_power": 0.5,
            "H_lr_factor": 1.5,
            "grad_clip_norm": 1.0,
        },
        "stage2": {
            "iterations": 600,
            "lr": 1e-3,
            "N_paths": 30_000,
            "M": 100,
            "kappa": 2,
            "variance_mode": "full",
            "keys": None,
            "scheduler": "cosine",
            "warmup_steps": 30,
            "cosine_power": 0.5,
            "grad_clip_norm": 1.0,
        },
    },

    "hybrid_exact": {
        "stage1": {
            "iterations": 400,
            "lr": 5e-3,
            "N_paths": 20_000,
            "M": 50,
            "kappa": 2,
            "variance_mode": "simplified",
            "keys": None,
            "scheduler": "cosine",
            "warmup_steps": 30,
            "cosine_power": 0.5,
            "H_lr_factor": 1.5,
            "grad_clip_norm": 1.0,
        },
        "stage2": {
            "iterations": 600,
            "lr": 1e-3,
            "N_paths": 30_000,
            "M": 100,
            "variance_mode": "full",
            "keys": None,
            "scheduler": "cosine",
            "warmup_steps": 30,
            "cosine_power": 0.5,
        },
    },

    "two_stage": {
        "stage1": {
            "iterations": 800,
            "lr": 5e-3,
            "N_paths": 10_000,
            "M": 50,
            "variance_mode": "simplified",  # paper §6.1; use "full" if keys include multi-rate
            "keys": None,
        },
        "stage2": {
            "iterations": 800,
            "lr": 5e-3,
            "N_paths": 30_000,
            "M": 50,
            "variance_mode": "full",
            "keys": None,
            "scheduler": "cosine",
            "warmup_steps": 30,
            "cosine_power": 0.5,
        },
    },

    "cross": {
        "test_keys": [
            (1.0, 3), (1.0, 7),
            (3.0, 2), (3.0, 7),
            (5.0, 3),
            (7.0, 2),
        ],
        "stage1": {
            "iterations": 400,
            "lr": 5e-3,
            "N_paths": 20_000,
            "M": 50,
            "kappa": 2,
            "variance_mode": "simplified",
            "keys": None,
            "scheduler": "cosine",
            "warmup_steps": 30,
            "cosine_power": 0.5,
            "H_lr_factor": 1.5,
            "grad_clip_norm": 1.0,
        },
        "stage2": {
            "iterations": 600,
            "lr": 1e-3,
            "N_paths": 30_000,
            "M": 100,
            "kappa": 2,
            "variance_mode": "full",
            "keys": None,
            "scheduler": "cosine",
            "warmup_steps": 30,
            "cosine_power": 0.5,
            "grad_clip_norm": 1.0,
        },
    },

    "roughness": {
        "rough_results_file": "amcc_calibration_results.pt",
        "H_values": [round(0.05 * i, 2) for i in range(1, 11)],
        "fixed_H": {
            "eta_init": 1.5,
            "iterations": 1200,
            "lr": 5e-3,
            "N_paths": 30_000,
            "M": 50,
            "variance_mode": "full",
            "scheduler": "cosine",
            "warmup_steps": 50,
            "cosine_power": 0.5,
            "early_stop_patience": 200,
        },
    },

    # --- Shared ---
    "early_stop_patience": 150,
    "early_stop_tol": 1e-4,
    "crn_seed": 42,
    "antithetic": True,

    # --- OOS α fine-tuning ---
    # After formula-based warm-start, run a short gradient pass with only
    # α free to correct the Jensen gap and multi-rate interpolation error.
    # H, η, ρ₀, ρ_{ij} remain frozen — only the ATM level is re-anchored.
    "oos_alpha_finetune": {
        "iterations": 80,
        "lr": 5e-3,
        "N_paths": 20_000,
        "M": 50,
        "kappa": 2,
        "scheduler": "cosine",
        "warmup_steps": 10,
        "cosine_power": 0.5,
    },

    # --- Diagnostics ---
    "diag_N_paths": 100_000,
    "diag_M": 100,
    "diag_scheme": "auto",
    "diag_hybrid_kappa": 2,
}


# =============================================================================
# Helpers
# =============================================================================

def _softplus_inv(a):
    a = float(a)
    if a > 20.0:
        return a
    return a + np.log(-np.expm1(-a))


def _interpolate_alpha(alpha, matched_indices):
    """Fill unmatched α values by linear interpolation with flat extrapolation."""
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



def _resolve_diag_scheme(cfg):
    """Resolve diag_scheme='auto' based on calibration mode."""
    raw = cfg.get("diag_scheme", "exact")
    if raw == "auto":
        scheme = "hybrid" if cfg["mode"] in (
            "hybrid", "hybrid_two_stage", "cross") else "exact"
        print(f"  diag_scheme=auto → resolved to '{scheme}' "
              f"(matches calibration mode)")
        return scheme
    return raw


def _auto_select_test_keys(all_keys):
    """Auto-select ~5 test keys for cross-validation.

    Strategy: for each expiry that has multiple tenors, hold out one
    multi-rate swaption (preferring the middle tenor). Never selects
    1Y-tenor swaptions since those anchor α.
    """
    by_expiry = {}
    for key in all_keys:
        exp, ten = key
        by_expiry.setdefault(exp, []).append(key)

    test_keys = []
    for exp in sorted(by_expiry.keys()):
        keys = by_expiry[exp]
        # Only multi-rate candidates (tenor > 1)
        multi = [k for k in keys if k[1] > 1]
        if not multi:
            continue
        # Pick the middle tenor
        multi.sort(key=lambda k: k[1])
        mid = multi[len(multi) // 2]
        test_keys.append(mid)

    print(f"  Auto-selected test keys: "
          + ", ".join(f"{k[0]:.0f}Y×{k[1]:.0f}Y" for k in test_keys))
    return test_keys


# =============================================================================
# Initialisation
# =============================================================================

def _formula_alpha_warmstart(params, mkt, H, eta):
    """Warm-start α from 1Y-tenor ATM IVs via the simplified variance formula.

    For a single-rate swaption (I, I+1) with expiry T:
        σ_ATM ≈ α_I · π_I · √G,   G = ∫₀¹ exp(η² (Ts)^{2H} / 4) ds
    so  α_I = σ_ATM / (π_I · √G).

    Rates not anchored by a 1Y-tenor swaption are filled by linear
    interpolation with flat extrapolation.
    """
    with torch.no_grad():
        p = params()
        alpha = p["alpha"].clone()

        # Compute G via Gauss–Legendre quadrature
        n_quad = 50
        nodes_np, weights_np = np.polynomial.legendre.leggauss(n_quad)
        s = torch.tensor(0.5 * (nodes_np + 1.0), dtype=main.DTYPE)
        w = torch.tensor(0.5 * weights_np, dtype=main.DTYPE)

        smile_keys_1y = sorted([k for k in mkt.swaptions.keys() if k[1] == 1])
        matched_indices = []

        for key in smile_keys_1y:
            swn = mkt.swaptions[key]
            I, J = swn.I, swn.J
            if J - I != 1:
                continue

            T = swn.expiry_years
            two_H = 2.0 * H
            G = (w * torch.exp(eta**2 * (T * s)**two_H / 4.0)).sum()

            atm_mask = (swn.strikes - swn.S0).abs() < 1e-6
            if atm_mask.sum() > 0:
                sigma_atm = swn.ivs_black[atm_mask][0]
            else:
                sigma_atm = swn.ivs_black[swn.n_strikes // 2]

            pi_j = swn.pi[0]
            alpha[I] = torch.clamp(sigma_atm / (pi_j * torch.sqrt(G) + 1e-30),
                                   min=1e-6)
            matched_indices.append(I)

        alpha = _interpolate_alpha(alpha, matched_indices)

        for j in range(mkt.N):
            params.alpha_tilde.data[j] = _softplus_inv(alpha[j].item())

    return matched_indices, alpha


def initialise_params(mkt, H_init=0.10, eta_init=2.0):
    """Create parameter module and warm-start all α via formula-based ATM matching."""
    params = MappedRoughSABRParams(N=mkt.N, device=mkt.device)
    params.set_H(H_init)
    params.set_eta(eta_init)

    with torch.no_grad():
        p = params()
        matched_indices, alpha_final = _formula_alpha_warmstart(
            params, mkt, p["H"], p["eta"])

        print(f"\n  Pass 1 — 1Y-tenor ATM matching:")
        print(f"    Matched rate indices (0-based): {matched_indices}")
        print(f"    α at matched: "
              + ", ".join(f"α[{i}]={alpha_final[i].item():.4f}"
                          for i in matched_indices))

        unmatched = [j for j in range(mkt.N) if j not in matched_indices]
        print(f"  Pass 2 — interpolation for unmatched indices: {unmatched}")
        print(f"    α after interpolation: "
              + ", ".join(f"α[{j}]={alpha_final[j].item():.4f}"
                          for j in unmatched))
        print(f"    α final: [{', '.join(f'{a:.4f}' for a in alpha_final.numpy())}]")

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


def initialise_fixed_H(mkt, H_val, eta_init=2.0):
    """Create parameter module with H fixed at a given value."""
    params = MappedRoughSABRParams(N=mkt.N, device=mkt.device)
    params.set_H(H_val)
    params.fix_H()
    params.set_eta(eta_init)

    with torch.no_grad():
        p = params()
        _formula_alpha_warmstart(params, mkt, p["H"], p["eta"])

    with torch.no_grad():
        p = params()
        print(f"\n  H={H_val:.2f} initialisation:")
        print(f"    H     = {p['H'].item():.4f}  [FIXED]")
        print(f"    η     = {p['eta'].item():.4f}")
        print(f"    α     = [{', '.join(f'{a:.4f}' for a in p['alpha'].numpy())}]")

    return params


def initialise_h05(mkt, eta_init=2.0):
    """Create parameter module with H fixed at 0.5 (Markovian SABR)."""
    return initialise_fixed_H(mkt, 0.5, eta_init=eta_init)


# =============================================================================
# Calibration modes
# =============================================================================

def _run_hybrid_stage(params, mkt, cfg, scfg, *,
                      freeze_H=False, crn_offset=0, label=""):
    """Run a single hybrid BLP calibration stage."""
    print(f"\n{'=' * 60}")
    h_status = (f"H frozen at {params.get_H().item():.4f}"
                if freeze_H else "H differentiable")
    print(f"{label}: Hybrid BLP (κ={scfg['kappa']}), {h_status}")
    print("=" * 60)
    print(f"  {scfg['iterations']} iterations, {scfg['N_paths']:,} paths, "
          f"{scfg['M']} steps, lr={scfg['lr']}")

    if freeze_H:
        params.fix_H()
    else:
        params.unfix_H()

    h_lr = scfg.get("H_lr_factor", 1.0)
    gc = scfg.get("grad_clip_norm", 10.0)
    if not freeze_H and h_lr != 1.0:
        print(f"  H_lr_factor={h_lr}, grad_clip_norm={gc}")

    result = calibrate(
        params, mkt,
        n_iterations=scfg["iterations"],
        lr=scfg["lr"],
        N_paths=scfg["N_paths"],
        M=scfg["M"],
        scheme="hybrid",
        hybrid_kappa=scfg["kappa"],
        variance_curve_mode=scfg["variance_mode"],
        use_crn=True,
        crn_seed=cfg["crn_seed"] + crn_offset,
        log_every=20,
        swaption_keys=scfg["keys"],
        scheduler_type=scfg.get("scheduler", "cosine"),
        warmup_steps=scfg.get("warmup_steps", 30),
        cosine_power=scfg.get("cosine_power", 0.5),
        early_stop_patience=scfg.get("early_stop_patience",
                                     cfg.get("early_stop_patience", 150)),
        early_stop_tol=cfg.get("early_stop_tol", 1e-4),
        H_lr_factor=h_lr if not freeze_H else 1.0,
        grad_clip_norm=gc,
        antithetic=cfg.get("antithetic", False),
    )

    if result["best_state"] is not None:
        params.load_state_dict(result["best_state"])

    with torch.no_grad():
        H_val = params.get_H().item()
        eta_val = params.get_eta().item()
        kappa_val = eta_val * np.sqrt(2.0 * H_val)
    print(f"\n{label} complete. H = {H_val:.4f}, "
          f"η = {eta_val:.4f}, κ = {kappa_val:.4f}")

    return result


def run_mode_hybrid(params, mkt, cfg):
    """Single-stage hybrid calibration.
    α is warm-started by formula-based ATM matching at initialisation
    and updated jointly by gradient descent throughout.
    """
    hcfg = cfg["hybrid"]
    params.alpha_tilde.requires_grad_(True)
    result = _run_hybrid_stage(
        params, mkt, cfg, hcfg,
        freeze_H=False, label="HYBRID")

    return {"history": result["history"]}


def run_mode_hybrid_two_stage(params, mkt, cfg):
    """Hybrid two-stage calibration (recommended).
    α is warm-started by formula-based ATM matching at initialisation.
    In Stage 1 α is updated jointly with H, η, ρ₀, ρ by gradient descent.
    At the stage boundary α remains in the graph throughout Stage 2.
    """
    h2cfg = cfg["hybrid_two_stage"]

    params.alpha_tilde.requires_grad_(True)
    stage1_result = _run_hybrid_stage(
        params, mkt, cfg, h2cfg["stage1"],
        freeze_H=False, crn_offset=0, label="STAGE 1")

    # Ensure α remains in graph for Stage 2 (freeze_H will fix H only)
    params.alpha_tilde.requires_grad_(True)

    stage2_result = _run_hybrid_stage(
        params, mkt, cfg, h2cfg["stage2"],
        freeze_H=True, crn_offset=10000, label="STAGE 2")

    return {
        "stage1": stage1_result,
        "stage2": stage2_result,
    }


def run_mode_hybrid_exact(params, mkt, cfg):
    """Hybrid → Exact two-stage calibration.

    Stage 1: Hybrid BLP scheme with H differentiable.
    Stage 2: Exact Cholesky scheme with H frozen.
    """
    gecfg = cfg["hybrid_exact"]

    params.alpha_tilde.requires_grad_(True)
    stage1_result = _run_hybrid_stage(
        params, mkt, cfg, gecfg["stage1"],
        freeze_H=False, crn_offset=0, label="STAGE 1 (hybrid)")

    # α remains in graph for Stage 2

    s2cfg = gecfg["stage2"]
    with torch.no_grad():
        H_val = params.get_H().item()
    print(f"\n{'=' * 60}")
    print(f"STAGE 2 (exact): Exact Cholesky scheme "
          f"(H fixed at {H_val:.4f})")
    print("=" * 60)
    print(f"  {s2cfg['iterations']} iterations, {s2cfg['N_paths']:,} paths, "
          f"{s2cfg['M']} steps, lr={s2cfg['lr']}")

    params.fix_H()

    stage2_result = calibrate(
        params, mkt,
        n_iterations=s2cfg["iterations"],
        lr=s2cfg["lr"],
        N_paths=s2cfg["N_paths"],
        M=s2cfg["M"],
        use_exact=True,
        variance_curve_mode=s2cfg["variance_mode"],
        use_crn=True,
        crn_seed=cfg["crn_seed"] + 10000,
        log_every=20,
        swaption_keys=s2cfg["keys"],
        scheduler_type=s2cfg.get("scheduler", "cosine"),
        warmup_steps=s2cfg.get("warmup_steps", 30),
        cosine_power=s2cfg.get("cosine_power", 0.5),
        early_stop_patience=cfg.get("early_stop_patience", 150),
        early_stop_tol=cfg.get("early_stop_tol", 1e-4),
        antithetic=cfg.get("antithetic", False),
    )

    if stage2_result["best_state"] is not None:
        params.load_state_dict(stage2_result["best_state"])

    with torch.no_grad():
        eta_val = params.get_eta().item()
        kappa_val = eta_val * np.sqrt(2.0 * H_val)
    print(f"\nStage 2 complete. H = {H_val:.4f}, "
          f"η = {eta_val:.4f}, κ = {kappa_val:.4f}")

    return {
        "stage1": stage1_result,
        "stage2": stage2_result,
    }


def run_mode_two_stage(params, mkt, cfg):
    """Legacy two-stage calibration (approx → exact Cholesky)."""
    s1cfg = cfg["two_stage"]["stage1"]
    s2cfg = cfg["two_stage"]["stage2"]

    result = calibrate_two_stage(
        params, mkt,
        stage1_iterations=s1cfg["iterations"],
        stage1_lr=s1cfg["lr"],
        stage1_N_paths=s1cfg["N_paths"],
        stage1_M=s1cfg["M"],
        stage1_keys=s1cfg["keys"],
        stage1_variance_mode=s1cfg["variance_mode"],
        stage2_iterations=s2cfg["iterations"],
        stage2_lr=s2cfg["lr"],
        stage2_N_paths=s2cfg["N_paths"],
        stage2_M=s2cfg["M"],
        stage2_variance_mode=s2cfg["variance_mode"],
        stage2_keys=s2cfg["keys"],
        stage2_scheduler=s2cfg.get("scheduler", "cosine"),
        stage2_warmup_steps=s2cfg.get("warmup_steps", 30),
        stage2_cosine_power=s2cfg.get("cosine_power", 0.5),
        use_crn=True,
        crn_seed=cfg["crn_seed"],
        log_every=20,
        early_stop_patience=cfg.get("early_stop_patience", 150),
        early_stop_tol=cfg.get("early_stop_tol", 1e-4),
    )

    return result


def run_mode_cross(params, mkt, cfg):
    """Cross-validation with train/test split.

    Splits swaptions into train and test sets. Runs hybrid two-stage
    calibration on train keys only. Returns result with split info for
    separate train/test diagnostics.

    If fewer than 3 of the configured test_keys exist in the market data
    (e.g. EUR has a different swaption grid), test keys are selected
    automatically: one multi-rate swaption per expiry bucket, choosing
    the middle tenor where available.
    """
    ccfg = cfg["cross"]

    all_keys = sorted(mkt.swaptions.keys())
    test_keys = [k for k in ccfg["test_keys"] if k in mkt.swaptions]

    if len(test_keys) < 3:
        print(f"  Only {len(test_keys)} configured test keys found in market data.")
        print(f"  Auto-selecting test keys from available swaptions...")
        test_keys = _auto_select_test_keys(all_keys)

    train_keys = [k for k in all_keys if k not in test_keys]

    # Never hold out 1Y-tenor swaptions — they anchor α
    test_1y = [k for k in test_keys if k[1] == 1]
    if test_1y:
        print(f"  Moving 1Y-tenor test keys to train: {test_1y}")
        for k in test_1y:
            test_keys.remove(k)
            train_keys.append(k)
        train_keys.sort()

    print(f"\n  Train ({len(train_keys)} swaptions — used in optimiser):")
    for k in train_keys:
        print(f"    {k[0]:.0f}Y×{k[1]:.0f}Y")
    print(f"\n  Test ({len(test_keys)} swaptions — held out):")
    for k in test_keys:
        print(f"    {k[0]:.0f}Y×{k[1]:.0f}Y")

    s1cfg = {**ccfg["stage1"], "keys": train_keys}
    params.alpha_tilde.requires_grad_(True)
    stage1_result = _run_hybrid_stage(
        params, mkt, cfg, s1cfg,
        freeze_H=False, crn_offset=0, label="STAGE 1 (train)")

    params.alpha_tilde.requires_grad_(True)
    s2cfg = {**ccfg["stage2"], "keys": train_keys}
    stage2_result = _run_hybrid_stage(
        params, mkt, cfg, s2cfg,
        freeze_H=True, crn_offset=10000, label="STAGE 2 (train)")

    return {
        "stage1": stage1_result,
        "stage2": stage2_result,
        "train_keys": train_keys,
        "test_keys": test_keys,
    }


def run_mode_roughness(mkt, cfg):
    """Roughness ablation — sweep fixed H values and compare.

    Loads existing rough calibration (H free) from file, then calibrates
    a model for each fixed H value in the configured sweep (default
    0.05, 0.10, ..., 0.50). Returns all parameter sets and histories.
    """
    rcfg = cfg["roughness"]
    hcfg = rcfg["fixed_H"]
    H_values = rcfg["H_values"]

    rough_file = rcfg["rough_results_file"]
    if not os.path.exists(rough_file):
        print(f"\n  {rough_file} not found.")
        print(f"  Run a standard calibration first, then re-run with mode='roughness'.")
        sys.exit(1)

    print(f"\nLoading rough calibration from {rough_file}...")
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
    print(f"  η     = {eta_rough:.4f}")
    print(f"  κ     = {kappa_rough:.4f}")
    print(f"  α     = [{', '.join(f'{a:.4f}' for a in p_r['alpha'].numpy())}]")

    fixed_models = {}
    for H_val in H_values:
        print("\n" + "#" * 60)
        print(f"# FIXED H = {H_val:.2f}")
        print("#" * 60)

        params_h = initialise_fixed_H(mkt, H_val, eta_init=hcfg["eta_init"])

        # Use dedicated SABR scheme for H ≈ 0.5 (no Cholesky needed)
        is_markovian = H_val >= 0.49
        h_scheme = "sabr" if is_markovian else "exact"

        print(f"\n  Calibrating with H = {H_val:.2f} fixed "
              f"(scheme={h_scheme})...")
        print(f"  {hcfg['iterations']} iterations, {hcfg['N_paths']:,} paths, "
              f"{hcfg['M']} steps, lr={hcfg['lr']}")

        result_h = calibrate(
            params_h, mkt,
            n_iterations=hcfg["iterations"],
            lr=hcfg["lr"],
            N_paths=hcfg["N_paths"],
            M=hcfg["M"],
            scheme=h_scheme,
            variance_curve_mode=hcfg["variance_mode"],
            use_crn=True,
            crn_seed=cfg["crn_seed"],
            log_every=20,
            swaption_keys=None,
            scheduler_type=hcfg.get("scheduler", "cosine"),
            warmup_steps=hcfg.get("warmup_steps", 50),
            cosine_power=hcfg.get("cosine_power", 0.5),
            early_stop_patience=hcfg.get("early_stop_patience", 200),
            early_stop_tol=1e-4,
            antithetic=cfg.get("antithetic", False),
        )

        if result_h["best_state"] is not None:
            params_h.load_state_dict(result_h["best_state"])

        with torch.no_grad():
            p_h = params_h()
            eta_h = p_h["eta"].item()
            kappa_h = eta_h * np.sqrt(2.0 * H_val)
            print(f"\n  H={H_val:.2f} calibration complete.")
            print(f"    η     = {eta_h:.4f},  κ = {kappa_h:.4f}")
            print(f"    α     = [{', '.join(f'{a:.4f}' for a in p_h['alpha'].numpy())}]")
            print(f"    ρ₀    = [{', '.join(f'{r:.3f}' for r in p_h['rho0'].numpy())}]")

        fixed_models[H_val] = {
            "params": params_h,
            "history": result_h["history"],
            "eta": eta_h,
        }

    return {
        "params_rough": params_rough,
        "H_rough": H_rough,
        "eta_rough": eta_rough,
        "fixed_models": fixed_models,
    }


# =============================================================================
# Diagnostics and reporting
# =============================================================================

def mc_diagnostics(params, mkt, N_paths=100_000, M=100, seed=42,
                   swaption_keys=None, scheme="exact", hybrid_kappa=2,
                   label=""):
    """Full MC-based calibration report with high path count."""
    header = f"MC-BASED CALIBRATION REPORT — {label}" if label else \
             "MC-BASED CALIBRATION REPORT"
    print("\n" + "=" * 72)
    print(header)
    print(f"({N_paths:,} paths, {M} time steps, scheme={scheme})")
    print("=" * 72)

    report = print_calibration_report(
        params, mkt,
        variance_curve_mode="full",
        N_paths=N_paths,
        M=M,
        seed=seed,
        swaption_keys=swaption_keys,
        scheme=scheme,
        hybrid_kappa=hybrid_kappa,
    )
    return report


def print_accuracy_summary(report, mkt, label=""):
    """Detailed accuracy breakdown from a calibration report."""
    per_swn = report.get("per_swaption", {})
    if not per_swn:
        print("  No per-swaption results to summarise.")
        return

    header = f"ACCURACY SUMMARY — {label}" if label else "ACCURACY SUMMARY"
    print("\n" + "=" * 100)
    print(header)
    print("=" * 100)

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

        errors = res["iv_errors"][valid] * 10000
        errors_np = errors.numpy()

        rmse = np.sqrt((errors_np ** 2).mean())
        mae = np.abs(errors_np).mean()
        bias = errors_np.mean()
        max_err = np.max(np.abs(errors_np))

        atm_mask = (swn.strikes - swn.S0).abs() < ATM_TOL
        if atm_mask.sum() > 0 and not torch.isnan(res["iv_errors"][atm_mask][0]):
            atm_err = res["iv_errors"][atm_mask][0].item() * 10000
        else:
            atm_err = float('nan')

        offsets = (swn.strikes - swn.S0)
        left_mask = (offsets < -ATM_TOL) & valid
        right_mask = (offsets > ATM_TOL) & valid

        left_bias = (res["iv_errors"][left_mask] * 10000).mean().item() \
            if left_mask.sum() > 0 else float('nan')
        right_bias = (res["iv_errors"][right_mask] * 10000).mean().item() \
            if right_mask.sum() > 0 else float('nan')

        mkt_prices = swn.target_prices[valid]
        mod_prices = res["model_prices"][valid]
        price_mask = mkt_prices.abs() > 1e-12
        if price_mask.sum() > 0:
            rel_errs = ((mod_prices[price_mask] - mkt_prices[price_mask])
                        / mkt_prices[price_mask]).numpy()
            price_rmse_pct = np.sqrt((rel_errs ** 2).mean()) * 100
        else:
            price_rmse_pct = float('nan')

        def _fmt(v, w=7, signed=False):
            if np.isnan(v):
                return f"{'N/A':>{w}s}"
            return f"{v:+{w}.1f}" if signed else f"{v:{w}.1f}"

        print(f"  {expiry:.0f}Y×{tenor:.0f}Y  {_fmt(rmse)}  {_fmt(mae)}  "
              f"{_fmt(bias, signed=True)}  {_fmt(atm_err, signed=True)}  "
              f"{_fmt(max_err)}  {_fmt(left_bias, signed=True)}  "
              f"{_fmt(right_bias, signed=True)}  "
              f"{price_rmse_pct:8.2f}%")

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

    print("-" * 100)
    total_rmse = np.sqrt(np.mean(all_sq)) if all_sq else float('nan')
    total_mae = np.mean(all_abs) if all_abs else float('nan')
    total_bias = np.mean(all_signed) if all_signed else float('nan')
    print(f"  {'TOTAL':>8s}  {total_rmse:7.1f}  {total_mae:7.1f}  "
          f"{total_bias:+7.1f}")

    print(f"\n  By expiry:")
    for exp in sorted(by_expiry.keys()):
        d = by_expiry[exp]
        r = np.sqrt(np.mean(d["sq"]))
        m = np.mean(d["abs"])
        b = np.mean(d["signed"])
        print(f"    {exp:.0f}Y expiry:  RMSE={r:.1f}bp  MAE={m:.1f}bp  "
              f"Bias={b:+.1f}bp")

    print(f"\n  By tenor:")
    for ten in sorted(by_tenor.keys()):
        d = by_tenor[ten]
        r = np.sqrt(np.mean(d["sq"]))
        m = np.mean(d["abs"])
        b = np.mean(d["signed"])
        n_swn = len([k for k in per_swn if k[1] == ten])
        print(f"    {ten:.0f}Y tenor:   RMSE={r:.1f}bp  MAE={m:.1f}bp  "
              f"Bias={b:+.1f}bp  ({n_swn} swaptions)")


def print_calibrated_params(params, mkt):
    """Print all calibrated parameter values."""
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


def run_in_sample_diagnostics(params, mkt, cfg, diag_scheme, diag_kappa):
    """Run in-sample MC diagnostics and smile comparisons."""
    print("\n" + "#" * 60)
    print(f"# IN-SAMPLE MC DIAGNOSTICS ({cfg['in_sample_date']})")
    print("#" * 60)

    report = mc_diagnostics(
        params, mkt,
        N_paths=cfg["diag_N_paths"],
        M=cfg["diag_M"],
        scheme=diag_scheme,
        hybrid_kappa=diag_kappa,
    )
    print_accuracy_summary(report, mkt,
                           label=f"IN-SAMPLE ({cfg['in_sample_date']})")

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
                scheme=diag_scheme,
                hybrid_kappa=diag_kappa,
            )

    return report, representative_keys


def run_oos_evaluation(params, mkt_oos, cfg, diag_scheme, diag_kappa,
                       representative_keys):
    """Run full out-of-sample evaluation with α re-anchoring.

    Fixes H, η, ρ₀, ρ_{ij} at their calibrated values.  α is first
    warm-started to the OOS date's ATM levels via the simplified variance
    formula, then fine-tuned by a short gradient-based pass through the
    full MC graph with only α free.  This corrects both the Jensen gap
    (formula proxy ignores stochastic variance) and the multi-rate
    interpolation error (formula only anchors 1Y-tenor swaptions).

    The smile shape parameters are never touched, so the OOS test
    measures whether the calibrated skew/curvature structure generalises
    across dates.
    """
    print("\n" + "=" * 72)
    print("  OUT-OF-SAMPLE EVALUATION")
    print(f"  Calibrated on: {cfg['in_sample_date']}")
    print(f"  Evaluating on: {cfg['out_sample_date']}")
    print("=" * 72)

    print_market_summary(mkt_oos)

    params_oos = copy.deepcopy(params)

    # α starts from the calibrated values — on consecutive dates the
    # yield curve barely moves, so the calibrated α is already close.
    # The fine-tuning below makes the small adjustments needed.
    with torch.no_grad():
        p = params_oos()
        print(f"\n--- Using calibrated α as OOS starting point ---")
        print(f"  α = [{', '.join(f'{a:.4f}' for a in p['alpha'].numpy())}]")

    # Gradient-based α fine-tuning (all other params frozen)
    ft_cfg = cfg.get("oos_alpha_finetune", {})
    ft_iters = ft_cfg.get("iterations", 80)

    if ft_iters > 0:
        print(f"\n--- Fine-tuning α on OOS date ({ft_iters} iterations, "
              f"only α free) ---")

        # Freeze everything except α
        params_oos.fix_H()
        params_oos.fix_eta()
        params_oos.fix_omega()
        params_oos.alpha_tilde.requires_grad_(True)

        ft_result = calibrate(
            params_oos, mkt_oos,
            n_iterations=ft_iters,
            lr=ft_cfg.get("lr", 5e-3),
            N_paths=ft_cfg.get("N_paths", 20_000),
            M=ft_cfg.get("M", 50),
            scheme=diag_scheme,
            hybrid_kappa=diag_kappa,
            variance_curve_mode="full",
            use_crn=True,
            crn_seed=cfg["crn_seed"] + 20000,
            log_every=20,
            swaption_keys=None,
            scheduler_type=ft_cfg.get("scheduler", "cosine"),
            warmup_steps=ft_cfg.get("warmup_steps", 10),
            cosine_power=ft_cfg.get("cosine_power", 0.5),
            early_stop_patience=ft_iters,   # no early stopping
            early_stop_tol=1e-6,
            grad_clip_norm=1.0,
            antithetic=cfg.get("antithetic", False),
        )

        if ft_result["best_state"] is not None:
            params_oos.load_state_dict(ft_result["best_state"])

        # Restore grad state for diagnostics (no grads needed)
        params_oos.alpha_tilde.requires_grad_(False)

        with torch.no_grad():
            p_ft = params_oos()
            print(f"\n  α after fine-tuning: "
                  f"[{', '.join(f'{a:.4f}' for a in p_ft['alpha'].numpy())}]")

    print(f"\n--- Out-of-sample MC report ({cfg['out_sample_date']}) ---")
    report_oos = mc_diagnostics(
        params_oos, mkt_oos,
        N_paths=cfg["diag_N_paths"],
        M=cfg["diag_M"],
        scheme=diag_scheme,
        hybrid_kappa=diag_kappa,
    )
    print_accuracy_summary(report_oos, mkt_oos,
                           label=f"OUT-OF-SAMPLE ({cfg['out_sample_date']})")

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
                scheme=diag_scheme,
                hybrid_kappa=diag_kappa,
            )

    return params_oos


def compare_reports(report_rough, report_h05, mkt):
    """Head-to-head comparison of rough vs H=0.5 calibrations.

    Returns dict with total RMSE for each model, per-swaption breakdowns,
    and win counts.
    """
    per_r = report_rough.get("per_swaption", {})
    per_h = report_h05.get("per_swaption", {})
    keys = sorted(set(per_r.keys()) & set(per_h.keys()))

    if not keys:
        print("  No common swaptions to compare.")
        return {}

    print("\n" + "=" * 100)
    print("PER-SWAPTION COMPARISON: Rough (H free) vs Markovian (H = 0.5)")
    print("=" * 100)

    print(f"\n  {'Swaption':>10s}  {'Rough':>8s}  {'H=0.5':>8s}  "
          f"{'ΔRMSE':>8s}  {'Rough':>8s}  {'H=0.5':>8s}  "
          f"{'ΔATM':>8s}  {'Winner':>8s}")
    print(f"  {'':>10s}  {'RMSE':>8s}  {'RMSE':>8s}  "
          f"{'(bp)':>8s}  {'ATM':>8s}  {'ATM':>8s}  "
          f"{'(bp)':>8s}  {'':>8s}")
    print("  " + "-" * 90)

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

        err_r = res_r["iv_errors"][valid] * 10000
        err_h = res_h["iv_errors"][valid] * 10000

        rmse_r = np.sqrt((err_r.numpy() ** 2).mean())
        rmse_h = np.sqrt((err_h.numpy() ** 2).mean())
        delta_rmse = rmse_h - rmse_r

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

        print(f"  {expiry:.0f}Y×{tenor:.0f}Y  "
              f"{rmse_r:8.1f}  {rmse_h:8.1f}  {delta_rmse:+8.1f}  "
              f"{atm_r:+8.1f}  {atm_h:+8.1f}  {delta_atm:+8.1f}  "
              f"{winner:>8s}")

        comparison_rows.append({
            "key": key, "rmse_rough": rmse_r, "rmse_h05": rmse_h,
            "delta": delta_rmse, "atm_rough": atm_r, "atm_h05": atm_h,
        })

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

        offsets = ((swn.strikes - swn.S0) * 10000).numpy()
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

    print("  " + "-" * 90)
    total_r = np.sqrt(np.mean(all_rough_sq))
    total_h = np.sqrt(np.mean(all_h05_sq))
    delta_total = total_h - total_r
    print(f"  {'TOTAL':>10s}  {total_r:8.1f}  {total_h:8.1f}  "
          f"{delta_total:+8.1f}  {'':>8s}  {'':>8s}  {'':>8s}  "
          f"{'Rough' if total_r < total_h else 'H=0.5':>8s}")
    print(f"\n  Rough wins {rough_wins}/{len(keys)} swaptions, "
          f"H=0.5 wins {h05_wins}/{len(keys)}")

    print(f"\n  {'BREAKDOWN BY EXPIRY':=^70}")
    print(f"  {'Expiry':>8s}  {'Rough':>8s}  {'H=0.5':>8s}  "
          f"{'ΔRMSE':>8s}  {'Rel Δ':>8s}")
    print("  " + "-" * 50)

    for exp in sorted(by_expiry.keys()):
        d = by_expiry[exp]
        r = np.sqrt(np.mean(d["rough_sq"]))
        h = np.sqrt(np.mean(d["h05_sq"]))
        delta = h - r
        rel = delta / h * 100 if h > 0 else 0
        print(f"  {exp:>6.0f}Y  {r:8.1f}  {h:8.1f}  "
              f"{delta:+8.1f}  {rel:+7.1f}%")

    print(f"\n  {'BREAKDOWN BY TENOR':=^70}")
    print(f"  {'Tenor':>8s}  {'Rough':>8s}  {'H=0.5':>8s}  "
          f"{'ΔRMSE':>8s}  {'Rel Δ':>8s}  {'#swn':>6s}")
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

    bucket_labels = {
        "deep_ITM": "Deep ITM (<=-100bp)",
        "ITM":      "ITM (-100 to -25bp)",
        "ATM":      "ATM (±25bp)",
        "OTM":      "OTM (+25 to +100bp)",
        "deep_OTM": "Deep OTM (>=+100bp)",
    }
    print(f"\n  {'BREAKDOWN BY MONEYNESS':=^70}")
    print(f"  {'Bucket':>22s}  {'Rough':>8s}  {'H=0.5':>8s}  "
          f"{'ΔRMSE':>8s}  {'Rel Δ':>8s}  {'#pts':>6s}")
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
        print(f"    ΔRMSE: {h_rmse - r_rmse:+.1f}bp")

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


def run_cross_diagnostics(params, mkt, cfg, diag_scheme, diag_kappa,
                          train_keys, test_keys):
    """Run separate MC diagnostics on train and test sets."""
    print("\n" + "#" * 60)
    print(f"# TRAIN SET MC DIAGNOSTICS ({len(train_keys)} swaptions)")
    print("#" * 60)

    report_train = mc_diagnostics(
        params, mkt,
        N_paths=cfg["diag_N_paths"],
        M=cfg["diag_M"],
        swaption_keys=train_keys,
        label=f"TRAIN ({len(train_keys)} swaptions)",
        scheme=diag_scheme,
        hybrid_kappa=diag_kappa,
    )
    print_accuracy_summary(report_train, mkt, label="TRAIN")

    print("\n" + "#" * 60)
    print(f"# TEST SET MC DIAGNOSTICS ({len(test_keys)} swaptions — held out)")
    print("#" * 60)

    report_test = mc_diagnostics(
        params, mkt,
        N_paths=cfg["diag_N_paths"],
        M=cfg["diag_M"],
        swaption_keys=test_keys,
        label=f"TEST ({len(test_keys)} swaptions, held out)",
        scheme=diag_scheme,
        hybrid_kappa=diag_kappa,
    )
    print_accuracy_summary(report_test, mkt, label="TEST (held out)")

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
                scheme=diag_scheme,
                hybrid_kappa=diag_kappa,
            )

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
                scheme=diag_scheme,
                hybrid_kappa=diag_kappa,
            )

    return report_train, report_test


# =============================================================================
# Plotting
# =============================================================================

def save_smile_plots(params, mkt, config, filename="amcc_smile_fits.png",
                     suptitle=None, test_keys=None):
    """Generate smile-fit plots for all swaptions.

    If test_keys is given, test swaptions are plotted in blue with [TEST]
    labels; train swaptions in red.
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
    raw_scheme = config.get("diag_scheme", "exact")
    if raw_scheme == "auto":
        diag_scheme = "hybrid" if config.get("mode") in (
            "hybrid", "hybrid_two_stage", "cross") else "exact"
    else:
        diag_scheme = raw_scheme
    diag_kappa = config.get("diag_hybrid_kappa", 2)

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
                    use_exact=(diag_scheme == "exact"),
                    variance_curve_mode="full",
                    cholesky_cache=cholesky_cache,
                    scheme=diag_scheme,
                    hybrid_kappa=diag_kappa,
                )
                mc_prices = compute_swaption_prices(S_T, swn)
                model_ivs = mc_prices_to_black_iv(mc_prices, swn)

                strikes_pct = swn.strikes.numpy() * 100
                mkt_ivs = swn.ivs_black.numpy() * 100
                mod_ivs = model_ivs.numpy() * 100

                is_test = key in test_set
                model_color = "blue" if is_test else "red"
                ax.plot(strikes_pct, mkt_ivs, "o-", color="black",
                        markersize=3, linewidth=0.5, label="Market")
                ax.plot(strikes_pct, mod_ivs, "x-", color=model_color,
                        markersize=4, linewidth=0.5, label="Model")

                tag = " [TEST]" if is_test else ""
                ax.set_title(f"{key[0]:.0f}Y × {key[1]:.0f}Y{tag}", fontsize=10)
                ax.set_xlabel("Strike (%)", fontsize=8)
                ax.set_ylabel("IV (%)", fontsize=8)
                ax.set_ylim(bottom=0)
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


def save_comparison_plots(params_rough, params_h05, mkt, cfg):
    """Side-by-side smile fits and RMSE bar chart: rough vs H=0.5."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping plots")
        return

    diag_scheme = cfg.get("diag_scheme", "exact")
    diag_kappa = cfg.get("diag_hybrid_kappa", 2)
    if diag_scheme == "auto":
        diag_scheme = "hybrid"

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

    cache_r, cache_h = {}, {}
    with torch.no_grad():
        for row_idx, mat in enumerate(maturities):
            keys_row = keys_by_maturity[mat]
            for col_idx, key in enumerate(keys_row):
                ax = axes[row_idx, col_idx]
                swn = mkt.swaptions[key]
                strikes_pct = swn.strikes.numpy() * 100
                mkt_ivs = swn.ivs_black.numpy() * 100

                torch.manual_seed(cfg["crn_seed"])
                S_T_r = simulate_swaption(
                    params_rough, swn, mkt,
                    N_paths=50_000, M=80,
                    use_exact=(diag_scheme == "exact"),
                    variance_curve_mode="full",
                    cholesky_cache=cache_r,
                    scheme=diag_scheme, hybrid_kappa=diag_kappa,
                )
                mod_r = mc_prices_to_black_iv(
                    compute_swaption_prices(S_T_r, swn), swn).numpy() * 100

                torch.manual_seed(cfg["crn_seed"])
                S_T_h = simulate_swaption(
                    params_h05, swn, mkt,
                    N_paths=50_000, M=80,
                    variance_curve_mode="full",
                    scheme="sabr",
                )
                mod_h = mc_prices_to_black_iv(
                    compute_swaption_prices(S_T_h, swn), swn).numpy() * 100

                ax.plot(strikes_pct, mkt_ivs, "o-", color="black",
                        markersize=3, linewidth=0.5, label="Market")
                ax.plot(strikes_pct, mod_r, "x-", color="blue",
                        markersize=4, linewidth=0.5, label="Rough")
                ax.plot(strikes_pct, mod_h, "x-", color="red",
                        markersize=4, linewidth=0.5, label="H=0.5")
                ax.set_title(f"{key[0]:.0f}Y × {key[1]:.0f}Y", fontsize=10)
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

    # RMSE bar chart
    fig, ax = plt.subplots(figsize=(12, 5))
    all_keys = sorted(mkt.swaptions.keys())
    labels = [f"{k[0]:.0f}Y×{k[1]:.0f}Y" for k in all_keys]

    rmse_r_list, rmse_h_list = [], []
    for key in all_keys:
        swn = mkt.swaptions[key]
        res_r = compute_model_smile(
            params_rough, swn, mkt, variance_curve_mode="full",
            N_paths=50_000, M=80, seed=cfg["crn_seed"],
            scheme=diag_scheme, hybrid_kappa=diag_kappa,
        )
        res_h = compute_model_smile(
            params_h05, swn, mkt, variance_curve_mode="full",
            N_paths=50_000, M=80, seed=cfg["crn_seed"],
            scheme="sabr",
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


def save_plots(params, mkt, history, config, history2=None, test_keys=None):
    """Generate convergence, gradient, smile, and correlation plots."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping plots")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    if history2 is not None:
        s1_steps = [r["step"] for r in history]
        s1_loss = [r["loss"] for r in history]
        s2_steps = [r["step"] + len(history) for r in history2]
        s2_loss = [r["loss"] for r in history2]

        ax1.semilogy(s1_steps, s1_loss, "b-", alpha=0.7, label="Stage 1")
        ax1.semilogy(s2_steps, s2_loss, "r-", alpha=0.7, label="Stage 2")
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

    all_history = history + (history2 or [])
    has_grad_norms = any("grad_norms" in r and r["grad_norms"]
                         for r in all_history)
    if has_grad_norms:
        _save_gradient_norm_plot(history, history2)

    if test_keys:
        suptitle = (f"Smile fits — train (red) / test (blue)  "
                    f"[{config.get('in_sample_date', '')}]")
    else:
        suptitle = f"In-sample smile fits ({config.get('in_sample_date', '')})"

    save_smile_plots(
        params, mkt, config,
        filename="amcc_smile_fits.png",
        suptitle=suptitle,
        test_keys=test_keys,
    )

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


def _save_gradient_norm_plot(history, history2=None):
    """Plot per-parameter gradient norms over the calibration run.

    For two-stage procedures, Stage 2 steps are offset so the x-axis is
    continuous, and a vertical line marks the stage boundary. Zero-gradient
    entries (frozen parameters) are excluded from plotting.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Build unified step/gradient arrays with proper offset
    s1_len = len(history)
    steps, H_grads, eta_grads, alpha_grads, omega_grads = [], [], [], [], []

    for i, r in enumerate(history):
        gn = r.get("grad_norms", {})
        if not gn:
            continue
        steps.append(r["step"])
        H_grads.append(gn.get("H_tilde", 0.0))
        eta_grads.append(gn.get("eta_tilde", 0.0))
        alpha_grads.append(gn.get("alpha_tilde", 0.0))
        omega_grads.append(gn.get("omega_tilde", 0.0))

    if history2:
        for r in history2:
            gn = r.get("grad_norms", {})
            if not gn:
                continue
            steps.append(r["step"] + s1_len)
            H_grads.append(gn.get("H_tilde", 0.0))
            eta_grads.append(gn.get("eta_tilde", 0.0))
            alpha_grads.append(gn.get("alpha_tilde", 0.0))
            omega_grads.append(gn.get("omega_tilde", 0.0))

    if not steps:
        return

    def _ema(steps_in, data_in, alpha=0.05):
        """EMA that skips zeros (frozen params) and returns clean arrays."""
        s_out, d_out = [], []
        prev = None
        for s, v in zip(steps_in, data_in):
            if v <= 0:
                continue
            if prev is None:
                prev = v
            else:
                prev = alpha * v + (1 - alpha) * prev
            s_out.append(s)
            d_out.append(prev)
        return s_out, d_out

    def _nonzero(steps_in, data_in):
        """Filter out zero entries (frozen parameters)."""
        s_out, d_out = [], []
        for s, v in zip(steps_in, data_in):
            if v > 0:
                s_out.append(s)
                d_out.append(v)
        return s_out, d_out

    # Per-parameter subplot
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
    fig.suptitle("Per-parameter gradient norms (before clipping)", fontsize=13)

    for ax, data, name, color in [
        (axes[0, 0], H_grads, "|∇H̃|", "red"),
        (axes[0, 1], eta_grads, "|∇η̃|", "blue"),
        (axes[1, 0], alpha_grads, "|∇α̃|", "green"),
        (axes[1, 1], omega_grads, "|∇ω̃|", "orange"),
    ]:
        s_nz, d_nz = _nonzero(steps, data)
        if s_nz:
            ax.semilogy(s_nz, d_nz, color=color, alpha=0.15, linewidth=0.5)
            s_ema, d_ema = _ema(steps, data)
            if len(d_ema) > 10:
                ax.semilogy(s_ema, d_ema, color=color, linewidth=1.5,
                            label=f"{name} (EMA)")
        ax.set_ylabel(name, fontsize=11)
        ax.set_xlabel("Iteration")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)
        if history2:
            ax.axvline(s1_len, color="gray", linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig("amcc_gradient_norms.png", dpi=150)
    print("Saved: amcc_gradient_norms.png")
    plt.close(fig)

    # Comparison overlay
    fig, ax = plt.subplots(figsize=(10, 5))
    for data, name, color in [
        (H_grads, "|∇H̃|", "red"),
        (eta_grads, "|∇η̃|", "blue"),
        (alpha_grads, "|∇α̃|", "green"),
        (omega_grads, "|∇ω̃|", "orange"),
    ]:
        s_ema, d_ema = _ema(steps, data)
        if len(d_ema) > 10:
            ax.semilogy(s_ema, d_ema, color=color, linewidth=1.5,
                        label=name)
    if history2:
        ax.axvline(s1_len, color="gray", linestyle="--", alpha=0.5,
                   label="Stage boundary")
    ax.set_xlabel("Iteration", fontsize=11)
    ax.set_ylabel("Gradient norm (EMA)", fontsize=11)
    ax.set_title("Gradient norm comparison — H vs other parameters")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("amcc_gradient_comparison.png", dpi=150)
    print("Saved: amcc_gradient_comparison.png")
    plt.close(fig)


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    t_start = time.time()
    cfg = CONFIG
    mode = cfg["mode"]

    # --- Set dtype ---
    _dtype_str = cfg.get("dtype", "float32")
    _dt = torch.float64 if _dtype_str == "float64" else torch.float32
    set_dtype(_dt)
    print(f"  dtype: {_dt}")

    # --- Load data ---
    print("Loading market data...")
    print(f"  Date: {cfg['in_sample_date']}")
    mkt = load_market_data(
        cfg["data_file"],
        subset=cfg["subset"],
        date=cfg["in_sample_date"],
        device=cfg["device"],
    )
    print_market_summary(mkt)

    if mode == "roughness":
        H_values = cfg["roughness"]["H_values"]
        print("\n" + "=" * 72)
        print(f"ROUGHNESS ABLATION STUDY: H free vs fixed H ∈ "
              f"{{{', '.join(f'{h:.2f}' for h in H_values)}}}")
        print("=" * 72)

        rr = run_mode_roughness(mkt, cfg)
        params_rough = rr["params_rough"]
        fixed_models = rr["fixed_models"]

        diag_kappa = cfg.get("diag_hybrid_kappa", 2)

        # The rough (H free) model was calibrated with the hybrid scheme,
        # so diagnostics must also use hybrid to avoid scheme mismatch.
        # Fixed-H models were calibrated with exact (or sabr for H=0.5),
        # so they use the matching scheme.
        report_rough = mc_diagnostics(
            params_rough, mkt, label="Rough (H free)",
            N_paths=cfg["diag_N_paths"], M=cfg["diag_M"],
            scheme="hybrid", hybrid_kappa=diag_kappa,
        )

        reports_fixed = {}
        for H_val in H_values:
            h_scheme = "sabr" if H_val >= 0.49 else "exact"
            reports_fixed[H_val] = mc_diagnostics(
                fixed_models[H_val]["params"], mkt,
                label=f"Fixed H = {H_val:.2f}",
                N_paths=cfg["diag_N_paths"], M=cfg["diag_M"],
                scheme=h_scheme, hybrid_kappa=diag_kappa,
            )

        params_h05 = fixed_models[0.5]["params"]
        report_h05 = reports_fixed[0.5]

        print("\n" + "#" * 60)
        print("# HEAD-TO-HEAD COMPARISON (Rough vs H=0.5)")
        print("#" * 60)

        comparison = compare_reports(report_rough, report_h05, mkt)

        print("\n" + "#" * 60)
        print("# RMSE ACROSS ALL FIXED H VALUES")
        print("#" * 60)

        per_rough = report_rough.get("per_swaption", {})
        all_keys = sorted(per_rough.keys())
        rough_sq_all = []
        for key in all_keys:
            res = per_rough[key]
            valid = ~torch.isnan(res["iv_errors"])
            if valid.sum() > 0:
                rough_sq_all.extend(
                    (res["iv_errors"][valid] * 10000).numpy() ** 2)
        total_rmse_rough = np.sqrt(np.mean(rough_sq_all)) if rough_sq_all else 0.0

        print(f"\n  {'H':>6s}  {'RMSE (bp)':>10s}  {'vs Rough':>10s}")
        print("  " + "-" * 32)
        print(f"  {'free':>6s}  {total_rmse_rough:10.1f}  {'---':>10s}")

        rmse_by_H = {}
        for H_val in H_values:
            per_h = reports_fixed[H_val].get("per_swaption", {})
            h_sq_all = []
            for key in all_keys:
                if key not in per_h:
                    continue
                res = per_h[key]
                valid = ~torch.isnan(res["iv_errors"])
                if valid.sum() > 0:
                    h_sq_all.extend(
                        (res["iv_errors"][valid] * 10000).numpy() ** 2)
            total_h = np.sqrt(np.mean(h_sq_all)) if h_sq_all else 0.0
            rmse_by_H[H_val] = total_h
            delta = total_h - total_rmse_rough
            print(f"  {H_val:6.2f}  {total_h:10.1f}  {delta:+10.1f}")

        best_H = min(rmse_by_H, key=rmse_by_H.get)
        print(f"\n  Best fixed H: {best_H:.2f} "
              f"(RMSE = {rmse_by_H[best_H]:.1f} bp)")

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
                    scheme="hybrid", hybrid_kappa=diag_kappa,
                )
                print(f"  [Best fixed H={best_H:.2f}]")
                best_scheme = "sabr" if best_H >= 0.49 else "exact"
                print_smile_comparison(
                    fixed_models[best_H]["params"],
                    mkt.swaptions[key], mkt,
                    variance_curve_mode="full",
                    N_paths=cfg["diag_N_paths"], M=cfg["diag_M"],
                    scheme=best_scheme, hybrid_kappa=diag_kappa,
                )

        print("\n" + "#" * 60)
        print("# GENERATING PLOTS")
        print("#" * 60)

        save_comparison_plots(params_rough, params_h05, mkt, cfg)

        total_h05 = rmse_by_H.get(0.5, 0.0)
        delta_h05 = total_h05 - total_rmse_rough
        rel = delta_h05 / total_h05 * 100 if total_h05 > 0 else 0

        print(f"\n  Rough (H free): H = {rr['H_rough']:.4f},  "
              f"RMSE = {total_rmse_rough:.1f} bp")
        print(f"  Best fixed:     H = {best_H:.2f},  "
              f"RMSE = {rmse_by_H[best_H]:.1f} bp")
        print(f"  Markovian:      H = 0.50,  "
              f"RMSE = {total_h05:.1f} bp")
        print(f"  Rough vs H=0.5 improvement: "
              f"{delta_h05:+.1f} bp  ({rel:+.1f}% relative)")

        elapsed = time.time() - t_start
        fixed_results = {}
        for H_val in H_values:
            m = fixed_models[H_val]
            with torch.no_grad():
                p_h = m["params"]()
            fixed_results[H_val] = {
                "H": H_val,
                "eta": m["eta"],
                "rmse": rmse_by_H[H_val],
                "params_state_dict": m["params"].state_dict(),
                "history": m["history"],
            }
        results = {
            "H_rough": rr["H_rough"],
            "eta_rough": rr["eta_rough"],
            "total_rmse_rough": total_rmse_rough,
            "H_values": H_values,
            "fixed_results": fixed_results,
            "rmse_by_H": rmse_by_H,
            "best_H": best_H,
            "comparison_rough_vs_h05": comparison,
            "elapsed_seconds": elapsed,
        }
        torch.save(results, "roughness_ablation_results.pt")
        print(f"\nResults saved to roughness_ablation_results.pt")

    elif mode == "cross":
        print("\n" + "#" * 60)
        print("# INITIALISATION")
        print("#" * 60)

        params = initialise_params(mkt, H_init=0.10, eta_init=2.0)

        print("\n" + "#" * 60)
        print("# AMCC CALIBRATION (train set only)")
        print("#" * 60)

        result = run_mode_cross(params, mkt, cfg)
        train_keys = result["train_keys"]
        test_keys = result["test_keys"]

        with torch.no_grad():
            H_calibrated = params.get_H().item()

        print_calibrated_params(params, mkt)

        diag_scheme = _resolve_diag_scheme(cfg)
        diag_kappa = cfg.get("diag_hybrid_kappa", 2)

        run_cross_diagnostics(
            params, mkt, cfg, diag_scheme, diag_kappa,
            train_keys, test_keys)

        print("\n" + "#" * 60)
        print("# GENERATING PLOTS")
        print("#" * 60)

        save_plots(
            params, mkt,
            history=result["stage1"]["history"],
            config=cfg,
            history2=result["stage2"]["history"],
            test_keys=test_keys,
        )

        elapsed = time.time() - t_start
        with torch.no_grad():
            p = params()
        results = {
            "config": cfg,
            "mode": mode,
            "params_state_dict": params.state_dict(),
            "H": p["H"].item(),
            "eta": p["eta"].item(),
            "alpha": p["alpha"].numpy(),
            "rho0": p["rho0"].numpy(),
            "rho": p["rho"].numpy(),
            "Sigma": p["Sigma"].numpy(),
            "H_calibrated": H_calibrated,
            "train_keys": train_keys,
            "test_keys": test_keys,
            "stage1_history": result["stage1"]["history"],
            "stage2_history": result["stage2"]["history"],
            "H_grad_norms": [
                r.get("grad_norms", {}).get("H_tilde", 0.0)
                for r in result["stage1"]["history"]
            ],
            "elapsed_seconds": elapsed,
        }
        torch.save(results, "amcc_calibration_results.pt")
        print(f"\nResults saved to amcc_calibration_results.pt")

    else:
        # Standard calibration modes: hybrid, hybrid_two_stage,
        # hybrid_exact, two_stage
        print("\n" + "#" * 60)
        print("# INITIALISATION")
        print("#" * 60)

        params = initialise_params(mkt, H_init=0.10, eta_init=2.0)

        print("\n" + "#" * 60)
        print("# AMCC CALIBRATION")
        print("#" * 60)

        if mode == "hybrid":
            result = run_mode_hybrid(params, mkt, cfg)
        elif mode == "hybrid_two_stage":
            result = run_mode_hybrid_two_stage(params, mkt, cfg)
        elif mode == "hybrid_exact":
            result = run_mode_hybrid_exact(params, mkt, cfg)
        elif mode == "two_stage":
            result = run_mode_two_stage(params, mkt, cfg)
        else:
            raise ValueError(f"Unknown mode: {mode}")

        with torch.no_grad():
            H_calibrated = params.get_H().item()

        print_calibrated_params(params, mkt)

        diag_scheme = _resolve_diag_scheme(cfg)
        diag_kappa = cfg.get("diag_hybrid_kappa", 2)

        report_in, representative_keys = run_in_sample_diagnostics(
            params, mkt, cfg, diag_scheme, diag_kappa)

        print("\n" + "#" * 60)
        print("# GENERATING PLOTS")
        print("#" * 60)

        is_single_stage = (mode == "hybrid")
        save_plots(
            params, mkt,
            history=result["history"] if is_single_stage
                    else result["stage1"]["history"],
            config=cfg,
            history2=None if is_single_stage
                     else result["stage2"]["history"],
        )

        mkt_oos = load_market_data(
            cfg["data_file"],
            subset=cfg["subset"],
            date=cfg["out_sample_date"],
            device=cfg["device"],
        )
        params_oos = run_oos_evaluation(
            params, mkt_oos, cfg, diag_scheme, diag_kappa,
            representative_keys)

        print("\n" + "#" * 60)
        print("# GENERATING OOS PLOTS")
        print("#" * 60)

        save_smile_plots(
            params_oos, mkt_oos, cfg,
            filename="amcc_smile_fits_oos.png",
            suptitle=f"Out-of-sample smile fits ({cfg['out_sample_date']})",
        )

        elapsed = time.time() - t_start
        with torch.no_grad():
            p = params()

        results = {
            "config": cfg,
            "mode": mode,
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
        if mode == "hybrid":
            results["history"] = result["history"]
            results["H_grad_norms"] = [
                r.get("grad_norms", {}).get("H_tilde", 0.0)
                for r in result["history"]
            ]
        else:
            results["stage1_history"] = result["stage1"]["history"]
            results["stage2_history"] = result["stage2"]["history"]
            if mode in ("hybrid_two_stage", "hybrid_exact"):
                results["H_grad_norms"] = [
                    r.get("grad_norms", {}).get("H_tilde", 0.0)
                    for r in result["stage1"]["history"]
                ]

        torch.save(results, "amcc_calibration_results.pt")
        print(f"\nResults saved to amcc_calibration_results.pt")

    elapsed = time.time() - t_start
    print(f"Total elapsed time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print("\nDone.")