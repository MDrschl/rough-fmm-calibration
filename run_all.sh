#!/bin/bash
set -e
# Auto-generated: 2026-04-01 23:04:07
# Device: cpu | Runs: 6

BASEDIR="$(cd "$(dirname "$0")" && pwd)"

TOTAL_START=$(date +%s)

# === Run 1/6: P1 | EUR | hybrid_two_stage     | 2025-12-08 → 2025-12-09 ===
echo ''
echo '============================================'
echo 'Run 1/6: P1 | EUR | hybrid_two_stage     | 2025-12-08 → 2025-12-09'
echo '============================================'
RUN_START=$(date +%s)
mkdir -p results/eur/phase1_modes/hybrid_two_stage/2025-12-08
cd $BASEDIR
python results/_run_001.py 2>&1 | tee results/eur/phase1_modes/hybrid_two_stage/2025-12-08/log.txt

# Move outputs
[ -f "$BASEDIR/amcc_calibration_results.pt" ] && mv "$BASEDIR/amcc_calibration_results.pt" results/eur/phase1_modes/hybrid_two_stage/2025-12-08/ 2>/dev/null || true
[ -f "$BASEDIR/amcc_convergence.png" ] && mv "$BASEDIR/amcc_convergence.png" results/eur/phase1_modes/hybrid_two_stage/2025-12-08/ 2>/dev/null || true
[ -f "$BASEDIR/amcc_gradient_norms.png" ] && mv "$BASEDIR/amcc_gradient_norms.png" results/eur/phase1_modes/hybrid_two_stage/2025-12-08/ 2>/dev/null || true
[ -f "$BASEDIR/amcc_gradient_comparison.png" ] && mv "$BASEDIR/amcc_gradient_comparison.png" results/eur/phase1_modes/hybrid_two_stage/2025-12-08/ 2>/dev/null || true
[ -f "$BASEDIR/amcc_smile_fits.png" ] && mv "$BASEDIR/amcc_smile_fits.png" results/eur/phase1_modes/hybrid_two_stage/2025-12-08/ 2>/dev/null || true
[ -f "$BASEDIR/amcc_smile_fits_oos.png" ] && mv "$BASEDIR/amcc_smile_fits_oos.png" results/eur/phase1_modes/hybrid_two_stage/2025-12-08/ 2>/dev/null || true
[ -f "$BASEDIR/amcc_correlation.png" ] && mv "$BASEDIR/amcc_correlation.png" results/eur/phase1_modes/hybrid_two_stage/2025-12-08/ 2>/dev/null || true
[ -f "$BASEDIR/roughness_ablation_results.pt" ] && mv "$BASEDIR/roughness_ablation_results.pt" results/eur/phase1_modes/hybrid_two_stage/2025-12-08/ 2>/dev/null || true
[ -f "$BASEDIR/roughness_ablation_smiles.png" ] && mv "$BASEDIR/roughness_ablation_smiles.png" results/eur/phase1_modes/hybrid_two_stage/2025-12-08/ 2>/dev/null || true
[ -f "$BASEDIR/roughness_ablation_rmse_bars.png" ] && mv "$BASEDIR/roughness_ablation_rmse_bars.png" results/eur/phase1_modes/hybrid_two_stage/2025-12-08/ 2>/dev/null || true
RUN_END=$(date +%s)
echo "Run 1 completed in $((RUN_END - RUN_START))s"
echo ''

# === Run 2/6: P1 | EUR | hybrid               | 2025-12-08 → 2025-12-09 ===
echo ''
echo '============================================'
echo 'Run 2/6: P1 | EUR | hybrid               | 2025-12-08 → 2025-12-09'
echo '============================================'
RUN_START=$(date +%s)
mkdir -p results/eur/phase1_modes/hybrid/2025-12-08
cd $BASEDIR
python results/_run_002.py 2>&1 | tee results/eur/phase1_modes/hybrid/2025-12-08/log.txt

# Move outputs
[ -f "$BASEDIR/amcc_calibration_results.pt" ] && mv "$BASEDIR/amcc_calibration_results.pt" results/eur/phase1_modes/hybrid/2025-12-08/ 2>/dev/null || true
[ -f "$BASEDIR/amcc_convergence.png" ] && mv "$BASEDIR/amcc_convergence.png" results/eur/phase1_modes/hybrid/2025-12-08/ 2>/dev/null || true
[ -f "$BASEDIR/amcc_gradient_norms.png" ] && mv "$BASEDIR/amcc_gradient_norms.png" results/eur/phase1_modes/hybrid/2025-12-08/ 2>/dev/null || true
[ -f "$BASEDIR/amcc_gradient_comparison.png" ] && mv "$BASEDIR/amcc_gradient_comparison.png" results/eur/phase1_modes/hybrid/2025-12-08/ 2>/dev/null || true
[ -f "$BASEDIR/amcc_smile_fits.png" ] && mv "$BASEDIR/amcc_smile_fits.png" results/eur/phase1_modes/hybrid/2025-12-08/ 2>/dev/null || true
[ -f "$BASEDIR/amcc_smile_fits_oos.png" ] && mv "$BASEDIR/amcc_smile_fits_oos.png" results/eur/phase1_modes/hybrid/2025-12-08/ 2>/dev/null || true
[ -f "$BASEDIR/amcc_correlation.png" ] && mv "$BASEDIR/amcc_correlation.png" results/eur/phase1_modes/hybrid/2025-12-08/ 2>/dev/null || true
[ -f "$BASEDIR/roughness_ablation_results.pt" ] && mv "$BASEDIR/roughness_ablation_results.pt" results/eur/phase1_modes/hybrid/2025-12-08/ 2>/dev/null || true
[ -f "$BASEDIR/roughness_ablation_smiles.png" ] && mv "$BASEDIR/roughness_ablation_smiles.png" results/eur/phase1_modes/hybrid/2025-12-08/ 2>/dev/null || true
[ -f "$BASEDIR/roughness_ablation_rmse_bars.png" ] && mv "$BASEDIR/roughness_ablation_rmse_bars.png" results/eur/phase1_modes/hybrid/2025-12-08/ 2>/dev/null || true
RUN_END=$(date +%s)
echo "Run 2 completed in $((RUN_END - RUN_START))s"
echo ''

# === Run 3/6: P1 | EUR | hybrid_exact         | 2025-12-08 → 2025-12-09 ===
echo ''
echo '============================================'
echo 'Run 3/6: P1 | EUR | hybrid_exact         | 2025-12-08 → 2025-12-09'
echo '============================================'
RUN_START=$(date +%s)
mkdir -p results/eur/phase1_modes/hybrid_exact/2025-12-08
cd $BASEDIR
python results/_run_003.py 2>&1 | tee results/eur/phase1_modes/hybrid_exact/2025-12-08/log.txt

# Move outputs
[ -f "$BASEDIR/amcc_calibration_results.pt" ] && mv "$BASEDIR/amcc_calibration_results.pt" results/eur/phase1_modes/hybrid_exact/2025-12-08/ 2>/dev/null || true
[ -f "$BASEDIR/amcc_convergence.png" ] && mv "$BASEDIR/amcc_convergence.png" results/eur/phase1_modes/hybrid_exact/2025-12-08/ 2>/dev/null || true
[ -f "$BASEDIR/amcc_gradient_norms.png" ] && mv "$BASEDIR/amcc_gradient_norms.png" results/eur/phase1_modes/hybrid_exact/2025-12-08/ 2>/dev/null || true
[ -f "$BASEDIR/amcc_gradient_comparison.png" ] && mv "$BASEDIR/amcc_gradient_comparison.png" results/eur/phase1_modes/hybrid_exact/2025-12-08/ 2>/dev/null || true
[ -f "$BASEDIR/amcc_smile_fits.png" ] && mv "$BASEDIR/amcc_smile_fits.png" results/eur/phase1_modes/hybrid_exact/2025-12-08/ 2>/dev/null || true
[ -f "$BASEDIR/amcc_smile_fits_oos.png" ] && mv "$BASEDIR/amcc_smile_fits_oos.png" results/eur/phase1_modes/hybrid_exact/2025-12-08/ 2>/dev/null || true
[ -f "$BASEDIR/amcc_correlation.png" ] && mv "$BASEDIR/amcc_correlation.png" results/eur/phase1_modes/hybrid_exact/2025-12-08/ 2>/dev/null || true
[ -f "$BASEDIR/roughness_ablation_results.pt" ] && mv "$BASEDIR/roughness_ablation_results.pt" results/eur/phase1_modes/hybrid_exact/2025-12-08/ 2>/dev/null || true
[ -f "$BASEDIR/roughness_ablation_smiles.png" ] && mv "$BASEDIR/roughness_ablation_smiles.png" results/eur/phase1_modes/hybrid_exact/2025-12-08/ 2>/dev/null || true
[ -f "$BASEDIR/roughness_ablation_rmse_bars.png" ] && mv "$BASEDIR/roughness_ablation_rmse_bars.png" results/eur/phase1_modes/hybrid_exact/2025-12-08/ 2>/dev/null || true
RUN_END=$(date +%s)
echo "Run 3 completed in $((RUN_END - RUN_START))s"
echo ''

# === Run 4/6: P1 | EUR | two_stage            | 2025-12-08 → 2025-12-09 ===
echo ''
echo '============================================'
echo 'Run 4/6: P1 | EUR | two_stage            | 2025-12-08 → 2025-12-09'
echo '============================================'
RUN_START=$(date +%s)
mkdir -p results/eur/phase1_modes/two_stage/2025-12-08
cd $BASEDIR
python results/_run_004.py 2>&1 | tee results/eur/phase1_modes/two_stage/2025-12-08/log.txt

# Move outputs
[ -f "$BASEDIR/amcc_calibration_results.pt" ] && mv "$BASEDIR/amcc_calibration_results.pt" results/eur/phase1_modes/two_stage/2025-12-08/ 2>/dev/null || true
[ -f "$BASEDIR/amcc_convergence.png" ] && mv "$BASEDIR/amcc_convergence.png" results/eur/phase1_modes/two_stage/2025-12-08/ 2>/dev/null || true
[ -f "$BASEDIR/amcc_gradient_norms.png" ] && mv "$BASEDIR/amcc_gradient_norms.png" results/eur/phase1_modes/two_stage/2025-12-08/ 2>/dev/null || true
[ -f "$BASEDIR/amcc_gradient_comparison.png" ] && mv "$BASEDIR/amcc_gradient_comparison.png" results/eur/phase1_modes/two_stage/2025-12-08/ 2>/dev/null || true
[ -f "$BASEDIR/amcc_smile_fits.png" ] && mv "$BASEDIR/amcc_smile_fits.png" results/eur/phase1_modes/two_stage/2025-12-08/ 2>/dev/null || true
[ -f "$BASEDIR/amcc_smile_fits_oos.png" ] && mv "$BASEDIR/amcc_smile_fits_oos.png" results/eur/phase1_modes/two_stage/2025-12-08/ 2>/dev/null || true
[ -f "$BASEDIR/amcc_correlation.png" ] && mv "$BASEDIR/amcc_correlation.png" results/eur/phase1_modes/two_stage/2025-12-08/ 2>/dev/null || true
[ -f "$BASEDIR/roughness_ablation_results.pt" ] && mv "$BASEDIR/roughness_ablation_results.pt" results/eur/phase1_modes/two_stage/2025-12-08/ 2>/dev/null || true
[ -f "$BASEDIR/roughness_ablation_smiles.png" ] && mv "$BASEDIR/roughness_ablation_smiles.png" results/eur/phase1_modes/two_stage/2025-12-08/ 2>/dev/null || true
[ -f "$BASEDIR/roughness_ablation_rmse_bars.png" ] && mv "$BASEDIR/roughness_ablation_rmse_bars.png" results/eur/phase1_modes/two_stage/2025-12-08/ 2>/dev/null || true
RUN_END=$(date +%s)
echo "Run 4 completed in $((RUN_END - RUN_START))s"
echo ''

# === Run 5/6: P2 | EUR | roughness           | 2025-12-08 ===
echo ''
echo '============================================'
echo 'Run 5/6: P2 | EUR | roughness           | 2025-12-08'
echo '============================================'
RUN_START=$(date +%s)
mkdir -p results/eur/phase2_roughness/roughness/2025-12-08
cd $BASEDIR
python results/_run_005.py 2>&1 | tee results/eur/phase2_roughness/roughness/2025-12-08/log.txt

# Move outputs
[ -f "$BASEDIR/amcc_calibration_results.pt" ] && mv "$BASEDIR/amcc_calibration_results.pt" results/eur/phase2_roughness/roughness/2025-12-08/ 2>/dev/null || true
[ -f "$BASEDIR/amcc_convergence.png" ] && mv "$BASEDIR/amcc_convergence.png" results/eur/phase2_roughness/roughness/2025-12-08/ 2>/dev/null || true
[ -f "$BASEDIR/amcc_gradient_norms.png" ] && mv "$BASEDIR/amcc_gradient_norms.png" results/eur/phase2_roughness/roughness/2025-12-08/ 2>/dev/null || true
[ -f "$BASEDIR/amcc_gradient_comparison.png" ] && mv "$BASEDIR/amcc_gradient_comparison.png" results/eur/phase2_roughness/roughness/2025-12-08/ 2>/dev/null || true
[ -f "$BASEDIR/amcc_smile_fits.png" ] && mv "$BASEDIR/amcc_smile_fits.png" results/eur/phase2_roughness/roughness/2025-12-08/ 2>/dev/null || true
[ -f "$BASEDIR/amcc_smile_fits_oos.png" ] && mv "$BASEDIR/amcc_smile_fits_oos.png" results/eur/phase2_roughness/roughness/2025-12-08/ 2>/dev/null || true
[ -f "$BASEDIR/amcc_correlation.png" ] && mv "$BASEDIR/amcc_correlation.png" results/eur/phase2_roughness/roughness/2025-12-08/ 2>/dev/null || true
[ -f "$BASEDIR/roughness_ablation_results.pt" ] && mv "$BASEDIR/roughness_ablation_results.pt" results/eur/phase2_roughness/roughness/2025-12-08/ 2>/dev/null || true
[ -f "$BASEDIR/roughness_ablation_smiles.png" ] && mv "$BASEDIR/roughness_ablation_smiles.png" results/eur/phase2_roughness/roughness/2025-12-08/ 2>/dev/null || true
[ -f "$BASEDIR/roughness_ablation_rmse_bars.png" ] && mv "$BASEDIR/roughness_ablation_rmse_bars.png" results/eur/phase2_roughness/roughness/2025-12-08/ 2>/dev/null || true
RUN_END=$(date +%s)
echo "Run 5 completed in $((RUN_END - RUN_START))s"
echo ''

# === Run 6/6: P2 | EUR | cross                | 2025-12-08 ===
echo ''
echo '============================================'
echo 'Run 6/6: P2 | EUR | cross                | 2025-12-08'
echo '============================================'
RUN_START=$(date +%s)
mkdir -p results/eur/phase2_cross/cross/2025-12-08
cd $BASEDIR
python results/_run_006.py 2>&1 | tee results/eur/phase2_cross/cross/2025-12-08/log.txt

# Move outputs
[ -f "$BASEDIR/amcc_calibration_results.pt" ] && mv "$BASEDIR/amcc_calibration_results.pt" results/eur/phase2_cross/cross/2025-12-08/ 2>/dev/null || true
[ -f "$BASEDIR/amcc_convergence.png" ] && mv "$BASEDIR/amcc_convergence.png" results/eur/phase2_cross/cross/2025-12-08/ 2>/dev/null || true
[ -f "$BASEDIR/amcc_gradient_norms.png" ] && mv "$BASEDIR/amcc_gradient_norms.png" results/eur/phase2_cross/cross/2025-12-08/ 2>/dev/null || true
[ -f "$BASEDIR/amcc_gradient_comparison.png" ] && mv "$BASEDIR/amcc_gradient_comparison.png" results/eur/phase2_cross/cross/2025-12-08/ 2>/dev/null || true
[ -f "$BASEDIR/amcc_smile_fits.png" ] && mv "$BASEDIR/amcc_smile_fits.png" results/eur/phase2_cross/cross/2025-12-08/ 2>/dev/null || true
[ -f "$BASEDIR/amcc_smile_fits_oos.png" ] && mv "$BASEDIR/amcc_smile_fits_oos.png" results/eur/phase2_cross/cross/2025-12-08/ 2>/dev/null || true
[ -f "$BASEDIR/amcc_correlation.png" ] && mv "$BASEDIR/amcc_correlation.png" results/eur/phase2_cross/cross/2025-12-08/ 2>/dev/null || true
[ -f "$BASEDIR/roughness_ablation_results.pt" ] && mv "$BASEDIR/roughness_ablation_results.pt" results/eur/phase2_cross/cross/2025-12-08/ 2>/dev/null || true
[ -f "$BASEDIR/roughness_ablation_smiles.png" ] && mv "$BASEDIR/roughness_ablation_smiles.png" results/eur/phase2_cross/cross/2025-12-08/ 2>/dev/null || true
[ -f "$BASEDIR/roughness_ablation_rmse_bars.png" ] && mv "$BASEDIR/roughness_ablation_rmse_bars.png" results/eur/phase2_cross/cross/2025-12-08/ 2>/dev/null || true
RUN_END=$(date +%s)
echo "Run 6 completed in $((RUN_END - RUN_START))s"
echo ''

TOTAL_END=$(date +%s)
TOTAL=$((TOTAL_END - TOTAL_START))
echo '=========================================='
echo "ALL ${0} RUNS COMPLETE in ${TOTAL}s ($(( TOTAL / 60 )) min)"
echo '=========================================='