#!/usr/bin/env python3
"""
run_all_calibrations.py
========================
Master script for the full calibration analysis.

Phase 1: Mode comparison
    Compare hybrid, hybrid_two_stage, hybrid_exact, two_stage
    across USD and EUR, training on 2024-12-09 / 2025-12-08,
    with OOS on the following day.

Phase 2: Deep analysis with best mode (hybrid_two_stage)
    - Roughness ablation (H sweep 0.05..0.50)
    - Cross-validation (train/test split)
    Both for USD+EUR on 2024-12-09 and 2025-12-08.
    OOS forecasting for USD+EUR: train on 2024-12-09/2025-12-08,
    evaluate on 2024-12-10/2025-12-09.

Usage:
    python run_all_calibrations.py [--device cuda] [--phase 1|2|all]
                                   [--dry-run] [--currency usd|eur|both]

Outputs saved to results/{currency}/{phase}/{mode}/{date}/
Each run directory contains:
    - amcc_calibration_results.pt  (all calibrated params, histories)
    - config_patch.json            (exact config used)
    - log.txt                      (full stdout)
    - *.png                        (plots)
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path

# ============================================================================
# Configuration
# ============================================================================

CURRENCIES = {
    "usd": {
        "data_file": "usd_swaption_data.pkl",
        "train_dates": ["2024-12-09", "2025-12-08"],
        "oos_pairs": [
            ("2024-12-09", "2024-12-10"),
            ("2025-12-08", "2025-12-09"),
        ],
    },
    "eur": {
        "data_file": "eur_swaption_data.pkl",
        "train_dates": ["2024-12-09", "2025-12-08"],
        "oos_pairs": [
            ("2024-12-09", "2024-12-10"),
            ("2025-12-08", "2025-12-09"),
        ],
    },
}

COMPARISON_MODES = ["hybrid_two_stage"]
BEST_MODE = "hybrid_two_stage"

# Time estimates per mode in minutes (GPU / CPU)
TIME_GPU = {
    "hybrid": 25, "hybrid_two_stage": 25,
    "hybrid_exact": 30, "two_stage": 20,
    "roughness": 90, "cross": 25,
}
CPU_FACTOR = 6


def run_dir(base, currency, phase, mode, date):
    d = Path(base) / currency / phase / mode / date
    d.mkdir(parents=True, exist_ok=True)
    return d


def generate_driver(run, results_dir):
    """Write a standalone Python driver script for one calibration run."""
    idx = run["index"]
    patch = run["patch"]
    prereq = run.get("prereq_dir", "")
    out = run["out_dir"]

    lines = [
        "#!/usr/bin/env python3",
        f'"""Auto-generated driver: {run["label"]}"""',
        "import sys, os, shutil, json",
        "",
        "# Work from the code directory (one level up from results/)",
        "os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))",
        "",
    ]

    # For roughness: copy base calibration .pt into cwd
    if run["mode"] == "roughness" and prereq:
        lines += [
            f"prereq = {prereq!r}",
            "base_pt = os.path.join(prereq, 'amcc_calibration_results.pt')",
            "if os.path.exists(base_pt):",
            "    shutil.copy2(base_pt, 'amcc_calibration_results.pt')",
            "    print(f'Copied base calibration from {base_pt}')",
            "else:",
            "    print(f'WARNING: {base_pt} not found')",
            "    sys.exit(1)",
            "",
        ]

    lines += [
        f"patch = {json.dumps(patch, indent=2)}",
        "",
        "# Write config override file (calibration.py reads this at startup)",
        "with open('_config_override.json', 'w') as _f:",
        "    json.dump(patch, _f, indent=2)",
        "",
        "# Run calibration (picks up _config_override.json automatically)",
        "try:",
        "    exec(compile(open('calibration.py').read(), 'calibration.py', 'exec'))",
        "finally:",
        "    # Clean up override file",
        "    if os.path.exists('_config_override.json'):",
        "        os.remove('_config_override.json')",
    ]

    driver_path = results_dir / f"_run_{idx:03d}.py"
    with open(driver_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    return driver_path


def generate_shell_script(runs, args, results_dir):
    """Generate run_all.sh that executes every run sequentially."""
    out_files = [
        "amcc_calibration_results.pt",
        "amcc_convergence.png",
        "amcc_gradient_norms.png",
        "amcc_gradient_comparison.png",
        "amcc_smile_fits.png",
        "amcc_smile_fits_oos.png",
        "amcc_correlation.png",
        "roughness_ablation_results.pt",
        "roughness_ablation_smiles.png",
        "roughness_ablation_rmse_bars.png",
    ]

    lines = [
        "#!/bin/bash",
        "set -e",
        f"# Auto-generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"# Device: {args.device} | Runs: {len(runs)}",
        "",
        'BASEDIR="$(cd "$(dirname "$0")" && pwd)"',
        "",
        "TOTAL_START=$(date +%s)",
        "",
    ]

    for run in runs:
        idx = run["index"]
        label = run["label"]
        out_dir = run["out_dir"]
        driver = f"{results_dir}/_run_{idx:03d}.py"

        lines += [
            f"# === Run {idx}/{len(runs)}: {label} ===",
            f"echo ''",
            f"echo '============================================'",
            f"echo 'Run {idx}/{len(runs)}: {label}'",
            f"echo '============================================'",
            f"RUN_START=$(date +%s)",
            f"mkdir -p {out_dir}",
            f"cd $BASEDIR",
            f"python {driver} 2>&1 | tee {out_dir}/log.txt",
            f"",
            f"# Move outputs",
        ]

        for f in out_files:
            lines.append(f'[ -f "$BASEDIR/{f}" ] && mv "$BASEDIR/{f}" {out_dir}/ 2>/dev/null || true')

        lines += [
            f"RUN_END=$(date +%s)",
            f"echo \"Run {idx} completed in $((RUN_END - RUN_START))s\"",
            f"echo ''",
            "",
        ]

    lines += [
        "TOTAL_END=$(date +%s)",
        "TOTAL=$((TOTAL_END - TOTAL_START))",
        "echo '=========================================='",
        "echo \"ALL ${0} RUNS COMPLETE in ${TOTAL}s ($(( TOTAL / 60 )) min)\"",
        "echo '=========================================='",
    ]

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Generate calibration run scripts")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--dtype", default="float32", choices=["float32", "float64"])
    parser.add_argument("--phase", default="all", choices=["1", "2", "all"])
    parser.add_argument("--currency", default="both", choices=["usd", "eur", "both"])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--results-dir", default="results")
    args = parser.parse_args()

    ccys = ["usd", "eur"] if args.currency == "both" else [args.currency]
    results_dir = Path(args.results_dir)
    runs = []
    idx = 0

    # ================================================================
    # PHASE 1: Mode comparison
    #   For each currency × mode × anchor date:
    #     - Train on anchor date (IS)
    #     - Evaluate OOS on following day
    #   This gives IS RMSE, OOS RMSE, wall time, params for each combo.
    # ================================================================
    if args.phase in ("1", "all"):
        print(f"\n{'=' * 60}")
        print("PHASE 1: Mode Comparison")
        print(f"  Modes: {COMPARISON_MODES}")
        print(f"  Currencies: {[c.upper() for c in ccys]}")
        print(f"{'=' * 60}")

        for ccy in ccys:
            cfg = CURRENCIES[ccy]
            for mode in COMPARISON_MODES:
                for train_date, oos_date in cfg["oos_pairs"]:
                    idx += 1
                    patch = {
                        "data_file": cfg["data_file"],
                        "in_sample_date": train_date,
                        "out_sample_date": oos_date,
                        "device": args.device,
                        "dtype": args.dtype,
                        "mode": mode,
                    }
                    out = run_dir(args.results_dir, ccy,
                                  "phase1_modes", mode, train_date)
                    label = (f"P1 | {ccy.upper()} | {mode:20s} | "
                             f"{train_date} → {oos_date}")

                    runs.append({
                        "index": idx, "label": label, "patch": patch,
                        "out_dir": str(out), "phase": 1,
                        "currency": ccy, "mode": mode,
                        "train_date": train_date,
                    })
                    print(f"  [{idx:3d}] {label}")

    # ================================================================
    # PHASE 2: Deep analysis with BEST_MODE
    # ================================================================
    if args.phase in ("2", "all"):
        print(f"\n{'=' * 60}")
        print(f"PHASE 2: Deep Analysis — {BEST_MODE}")
        print(f"  Currencies: {[c.upper() for c in ccys]}")
        print(f"{'=' * 60}")

        for ccy in ccys:
            cfg = CURRENCIES[ccy]

            # 2a: Roughness ablation on each anchor date
            #     Requires base hybrid_two_stage results from Phase 1.
            for train_date in cfg["train_dates"]:
                idx += 1
                oos_date = next(p[1] for p in cfg["oos_pairs"]
                                if p[0] == train_date)

                base_dir = run_dir(args.results_dir, ccy,
                                   "phase1_modes", BEST_MODE, train_date)

                patch = {
                    "data_file": cfg["data_file"],
                    "in_sample_date": train_date,
                    "out_sample_date": oos_date,
                    "device": args.device,
                    "dtype": args.dtype,
                    "mode": "roughness",
                }
                out = run_dir(args.results_dir, ccy,
                              "phase2_roughness", "roughness", train_date)

                label = (f"P2 | {ccy.upper()} | roughness           | "
                         f"{train_date}")

                runs.append({
                    "index": idx, "label": label, "patch": patch,
                    "out_dir": str(out), "phase": 2,
                    "currency": ccy, "mode": "roughness",
                    "train_date": train_date,
                    "prereq_dir": str(base_dir),
                })
                print(f"  [{idx:3d}] {label}  (needs Phase 1 base)")

            # 2b: Cross-validation on each anchor date
            for train_date in cfg["train_dates"]:
                idx += 1
                oos_date = next(p[1] for p in cfg["oos_pairs"]
                                if p[0] == train_date)

                patch = {
                    "data_file": cfg["data_file"],
                    "in_sample_date": train_date,
                    "out_sample_date": oos_date,
                    "device": args.device,
                    "dtype": args.dtype,
                    "mode": "cross",
                }
                out = run_dir(args.results_dir, ccy,
                              "phase2_cross", "cross", train_date)

                label = (f"P2 | {ccy.upper()} | cross                | "
                         f"{train_date}")

                runs.append({
                    "index": idx, "label": label, "patch": patch,
                    "out_dir": str(out), "phase": 2,
                    "currency": ccy, "mode": "cross",
                    "train_date": train_date,
                })
                print(f"  [{idx:3d}] {label}")

    # ================================================================
    # Summary
    # ================================================================
    p1 = sum(1 for r in runs if r["phase"] == 1)
    p2 = sum(1 for r in runs if r["phase"] == 2)

    total_gpu = sum(TIME_GPU.get(r["mode"], 30) for r in runs)
    total_cpu = total_gpu * CPU_FACTOR
    est = total_gpu if args.device == "cuda" else total_cpu

    print(f"\n{'=' * 60}")
    print(f"TOTAL: {len(runs)} runs  (Phase 1: {p1}, Phase 2: {p2})")
    print(f"Estimated: ~{est} min ({est/60:.1f} hrs) on {args.device}")
    print(f"{'=' * 60}")

    if args.dry_run:
        print("\n[DRY RUN] No scripts generated.")
        return

    # ================================================================
    # Write driver scripts + shell script
    # ================================================================
    results_dir.mkdir(parents=True, exist_ok=True)

    for run in runs:
        generate_driver(run, results_dir)

        out_path = Path(run["out_dir"])
        out_path.mkdir(parents=True, exist_ok=True)
        with open(out_path / "config_patch.json", "w") as f:
            json.dump(run["patch"], f, indent=2)

    shell = generate_shell_script(runs, args, results_dir)
    script_path = Path("run_all.sh")
    with open(script_path, "w") as f:
        f.write(shell)
    os.chmod(script_path, 0o755)

    print(f"\nGenerated:")
    print(f"  {len(runs)} driver scripts in {results_dir}/")
    print(f"  Master script: run_all.sh")
    print(f"\nRun everything:")
    print(f"  bash run_all.sh")
    print(f"\nOr individual runs:")
    print(f"  python {results_dir}/_run_001.py")


if __name__ == "__main__":
    main()