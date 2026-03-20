#!/bin/bash
set -e
# Auto-generated: 2026-03-20 07:59:45
# Device: cpu | Runs: 12

BASEDIR="$(cd "$(dirname "$0")" && pwd)"

TOTAL_START=$(date +%s)

# === Run 1/12: P1 | USD | hybrid_two_stage     | 2024-12-09 → 2024-12-10 ===
echo ''
echo '============================================'
echo 'Run 1/12: P1 | USD | hybrid_two_stage     | 2024-12-09 → 2024-12-10'
echo '============================================'
RUN_START=$(date +%s)
mkdir -p results/usd/phase1_modes/hybrid_two_stage/2024-12-09
cd $BASEDIR
python results/_run_001.py 2>&1 | tee results/usd/phase1_modes/hybrid_two_stage/2024-12-09/log.txt

# Move outputs
[ -f "$BASEDIR/amcc_calibration_results.pt" ] && mv "$BASEDIR/amcc_calibration_results.pt" results/usd/phase1_modes/hybrid_two_stage/2024-12-09/ 2>/dev/null || true
[ -f "$BASEDIR/amcc_convergence.png" ] && mv "$BASEDIR/amcc_convergence.png" results/usd/phase1_modes/hybrid_two_stage/2024-12-09/ 2>/dev/null || true
[ -f "$BASEDIR/amcc_gradient_norms.png" ] && mv "$BASEDIR/amcc_gradient_norms.png" results/usd/phase1_modes/hybrid_two_stage/2024-12-09/ 2>/dev/null || true
[ -f "$BASEDIR/amcc_gradient_comparison.png" ] && mv "$BASEDIR/amcc_gradient_comparison.png" results/usd/phase1_modes/hybrid_two_stage/2024-12-09/ 2>/dev/null || true
[ -f "$BASEDIR/amcc_smile_fits.png" ] && mv "$BASEDIR/amcc_smile_fits.png" results/usd/phase1_modes/hybrid_two_stage/2024-12-09/ 2>/dev/null || true
[ -f "$BASEDIR/amcc_smile_fits_oos.png" ] && mv "$BASEDIR/amcc_smile_fits_oos.png" results/usd/phase1_modes/hybrid_two_stage/2024-12-09/ 2>/dev/null || true
[ -f "$BASEDIR/amcc_correlation.png" ] && mv "$BASEDIR/amcc_correlation.png" results/usd/phase1_modes/hybrid_two_stage/2024-12-09/ 2>/dev/null || true
[ -f "$BASEDIR/roughness_ablation_results.pt" ] && mv "$BASEDIR/roughness_ablation_results.pt" results/usd/phase1_modes/hybrid_two_stage/2024-12-09/ 2>/dev/null || true
[ -f "$BASEDIR/roughness_ablation_smiles.png" ] && mv "$BASEDIR/roughness_ablation_smiles.png" results/usd/phase1_modes/hybrid_two_stage/2024-12-09/ 2>/dev/null || true
[ -f "$BASEDIR/roughness_ablation_rmse_bars.png" ] && mv "$BASEDIR/roughness_ablation_rmse_bars.png" results/usd/phase1_modes/hybrid_two_stage/2024-12-09/ 2>/dev/null || true
RUN_END=$(date +%s)
echo "Run 1 completed in $((RUN_END - RUN_START))s"
echo ''

# === Run 2/12: P1 | USD | hybrid_two_stage     | 2025-12-08 → 2025-12-09 ===
echo ''
echo '============================================'
echo 'Run 2/12: P1 | USD | hybrid_two_stage     | 2025-12-08 → 2025-12-09'
echo '============================================'
RUN_START=$(date +%s)
mkdir -p results/usd/phase1_modes/hybrid_two_stage/2025-12-08
cd $BASEDIR
python results/_run_002.py 2>&1 | tee results/usd/phase1_modes/hybrid_two_stage/2025-12-08/log.txt

# Move outputs
[ -f "$BASEDIR/amcc_calibration_results.pt" ] && mv "$BASEDIR/amcc_calibration_results.pt" results/usd/phase1_modes/hybrid_two_stage/2025-12-08/ 2>/dev/null || true
[ -f "$BASEDIR/amcc_convergence.png" ] && mv "$BASEDIR/amcc_convergence.png" results/usd/phase1_modes/hybrid_two_stage/2025-12-08/ 2>/dev/null || true
[ -f "$BASEDIR/amcc_gradient_norms.png" ] && mv "$BASEDIR/amcc_gradient_norms.png" results/usd/phase1_modes/hybrid_two_stage/2025-12-08/ 2>/dev/null || true
[ -f "$BASEDIR/amcc_gradient_comparison.png" ] && mv "$BASEDIR/amcc_gradient_comparison.png" results/usd/phase1_modes/hybrid_two_stage/2025-12-08/ 2>/dev/null || true
[ -f "$BASEDIR/amcc_smile_fits.png" ] && mv "$BASEDIR/amcc_smile_fits.png" results/usd/phase1_modes/hybrid_two_stage/2025-12-08/ 2>/dev/null || true
[ -f "$BASEDIR/amcc_smile_fits_oos.png" ] && mv "$BASEDIR/amcc_smile_fits_oos.png" results/usd/phase1_modes/hybrid_two_stage/2025-12-08/ 2>/dev/null || true
[ -f "$BASEDIR/amcc_correlation.png" ] && mv "$BASEDIR/amcc_correlation.png" results/usd/phase1_modes/hybrid_two_stage/2025-12-08/ 2>/dev/null || true
[ -f "$BASEDIR/roughness_ablation_results.pt" ] && mv "$BASEDIR/roughness_ablation_results.pt" results/usd/phase1_modes/hybrid_two_stage/2025-12-08/ 2>/dev/null || true
[ -f "$BASEDIR/roughness_ablation_smiles.png" ] && mv "$BASEDIR/roughness_ablation_smiles.png" results/usd/phase1_modes/hybrid_two_stage/2025-12-08/ 2>/dev/null || true
[ -f "$BASEDIR/roughness_ablation_rmse_bars.png" ] && mv "$BASEDIR/roughness_ablation_rmse_bars.png" results/usd/phase1_modes/hybrid_two_stage/2025-12-08/ 2>/dev/null || true
RUN_END=$(date +%s)
echo "Run 2 completed in $((RUN_END - RUN_START))s"
echo ''

# === Run 3/12: P1 | USD | hybrid               | 2024-12-09 → 2024-12-10 ===
echo ''
echo '============================================'
echo 'Run 3/12: P1 | USD | hybrid               | 2024-12-09 → 2024-12-10'
echo '============================================'
RUN_START=$(date +%s)
mkdir -p results/usd/phase1_modes/hybrid/2024-12-09
cd $BASEDIR
python results/_run_003.py 2>&1 | tee results/usd/phase1_modes/hybrid/2024-12-09/log.txt

# Move outputs
[ -f "$BASEDIR/amcc_calibration_results.pt" ] && mv "$BASEDIR/amcc_calibration_results.pt" results/usd/phase1_modes/hybrid/2024-12-09/ 2>/dev/null || true
[ -f "$BASEDIR/amcc_convergence.png" ] && mv "$BASEDIR/amcc_convergence.png" results/usd/phase1_modes/hybrid/2024-12-09/ 2>/dev/null || true
[ -f "$BASEDIR/amcc_gradient_norms.png" ] && mv "$BASEDIR/amcc_gradient_norms.png" results/usd/phase1_modes/hybrid/2024-12-09/ 2>/dev/null || true
[ -f "$BASEDIR/amcc_gradient_comparison.png" ] && mv "$BASEDIR/amcc_gradient_comparison.png" results/usd/phase1_modes/hybrid/2024-12-09/ 2>/dev/null || true
[ -f "$BASEDIR/amcc_smile_fits.png" ] && mv "$BASEDIR/amcc_smile_fits.png" results/usd/phase1_modes/hybrid/2024-12-09/ 2>/dev/null || true
[ -f "$BASEDIR/amcc_smile_fits_oos.png" ] && mv "$BASEDIR/amcc_smile_fits_oos.png" results/usd/phase1_modes/hybrid/2024-12-09/ 2>/dev/null || true
[ -f "$BASEDIR/amcc_correlation.png" ] && mv "$BASEDIR/amcc_correlation.png" results/usd/phase1_modes/hybrid/2024-12-09/ 2>/dev/null || true
[ -f "$BASEDIR/roughness_ablation_results.pt" ] && mv "$BASEDIR/roughness_ablation_results.pt" results/usd/phase1_modes/hybrid/2024-12-09/ 2>/dev/null || true
[ -f "$BASEDIR/roughness_ablation_smiles.png" ] && mv "$BASEDIR/roughness_ablation_smiles.png" results/usd/phase1_modes/hybrid/2024-12-09/ 2>/dev/null || true
[ -f "$BASEDIR/roughness_ablation_rmse_bars.png" ] && mv "$BASEDIR/roughness_ablation_rmse_bars.png" results/usd/phase1_modes/hybrid/2024-12-09/ 2>/dev/null || true
RUN_END=$(date +%s)
echo "Run 3 completed in $((RUN_END - RUN_START))s"
echo ''

# === Run 4/12: P1 | USD | hybrid               | 2025-12-08 → 2025-12-09 ===
echo ''
echo '============================================'
echo 'Run 4/12: P1 | USD | hybrid               | 2025-12-08 → 2025-12-09'
echo '============================================'
RUN_START=$(date +%s)
mkdir -p results/usd/phase1_modes/hybrid/2025-12-08
cd $BASEDIR
python results/_run_004.py 2>&1 | tee results/usd/phase1_modes/hybrid/2025-12-08/log.txt

# Move outputs
[ -f "$BASEDIR/amcc_calibration_results.pt" ] && mv "$BASEDIR/amcc_calibration_results.pt" results/usd/phase1_modes/hybrid/2025-12-08/ 2>/dev/null || true
[ -f "$BASEDIR/amcc_convergence.png" ] && mv "$BASEDIR/amcc_convergence.png" results/usd/phase1_modes/hybrid/2025-12-08/ 2>/dev/null || true
[ -f "$BASEDIR/amcc_gradient_norms.png" ] && mv "$BASEDIR/amcc_gradient_norms.png" results/usd/phase1_modes/hybrid/2025-12-08/ 2>/dev/null || true
[ -f "$BASEDIR/amcc_gradient_comparison.png" ] && mv "$BASEDIR/amcc_gradient_comparison.png" results/usd/phase1_modes/hybrid/2025-12-08/ 2>/dev/null || true
[ -f "$BASEDIR/amcc_smile_fits.png" ] && mv "$BASEDIR/amcc_smile_fits.png" results/usd/phase1_modes/hybrid/2025-12-08/ 2>/dev/null || true
[ -f "$BASEDIR/amcc_smile_fits_oos.png" ] && mv "$BASEDIR/amcc_smile_fits_oos.png" results/usd/phase1_modes/hybrid/2025-12-08/ 2>/dev/null || true
[ -f "$BASEDIR/amcc_correlation.png" ] && mv "$BASEDIR/amcc_correlation.png" results/usd/phase1_modes/hybrid/2025-12-08/ 2>/dev/null || true
[ -f "$BASEDIR/roughness_ablation_results.pt" ] && mv "$BASEDIR/roughness_ablation_results.pt" results/usd/phase1_modes/hybrid/2025-12-08/ 2>/dev/null || true
[ -f "$BASEDIR/roughness_ablation_smiles.png" ] && mv "$BASEDIR/roughness_ablation_smiles.png" results/usd/phase1_modes/hybrid/2025-12-08/ 2>/dev/null || true
[ -f "$BASEDIR/roughness_ablation_rmse_bars.png" ] && mv "$BASEDIR/roughness_ablation_rmse_bars.png" results/usd/phase1_modes/hybrid/2025-12-08/ 2>/dev/null || true
RUN_END=$(date +%s)
echo "Run 4 completed in $((RUN_END - RUN_START))s"
echo ''

# === Run 5/12: P1 | USD | hybrid_exact         | 2024-12-09 → 2024-12-10 ===
echo ''
echo '============================================'
echo 'Run 5/12: P1 | USD | hybrid_exact         | 2024-12-09 → 2024-12-10'
echo '============================================'
RUN_START=$(date +%s)
mkdir -p results/usd/phase1_modes/hybrid_exact/2024-12-09
cd $BASEDIR
python results/_run_005.py 2>&1 | tee results/usd/phase1_modes/hybrid_exact/2024-12-09/log.txt

# Move outputs
[ -f "$BASEDIR/amcc_calibration_results.pt" ] && mv "$BASEDIR/amcc_calibration_results.pt" results/usd/phase1_modes/hybrid_exact/2024-12-09/ 2>/dev/null || true
[ -f "$BASEDIR/amcc_convergence.png" ] && mv "$BASEDIR/amcc_convergence.png" results/usd/phase1_modes/hybrid_exact/2024-12-09/ 2>/dev/null || true
[ -f "$BASEDIR/amcc_gradient_norms.png" ] && mv "$BASEDIR/amcc_gradient_norms.png" results/usd/phase1_modes/hybrid_exact/2024-12-09/ 2>/dev/null || true
[ -f "$BASEDIR/amcc_gradient_comparison.png" ] && mv "$BASEDIR/amcc_gradient_comparison.png" results/usd/phase1_modes/hybrid_exact/2024-12-09/ 2>/dev/null || true
[ -f "$BASEDIR/amcc_smile_fits.png" ] && mv "$BASEDIR/amcc_smile_fits.png" results/usd/phase1_modes/hybrid_exact/2024-12-09/ 2>/dev/null || true
[ -f "$BASEDIR/amcc_smile_fits_oos.png" ] && mv "$BASEDIR/amcc_smile_fits_oos.png" results/usd/phase1_modes/hybrid_exact/2024-12-09/ 2>/dev/null || true
[ -f "$BASEDIR/amcc_correlation.png" ] && mv "$BASEDIR/amcc_correlation.png" results/usd/phase1_modes/hybrid_exact/2024-12-09/ 2>/dev/null || true
[ -f "$BASEDIR/roughness_ablation_results.pt" ] && mv "$BASEDIR/roughness_ablation_results.pt" results/usd/phase1_modes/hybrid_exact/2024-12-09/ 2>/dev/null || true
[ -f "$BASEDIR/roughness_ablation_smiles.png" ] && mv "$BASEDIR/roughness_ablation_smiles.png" results/usd/phase1_modes/hybrid_exact/2024-12-09/ 2>/dev/null || true
[ -f "$BASEDIR/roughness_ablation_rmse_bars.png" ] && mv "$BASEDIR/roughness_ablation_rmse_bars.png" results/usd/phase1_modes/hybrid_exact/2024-12-09/ 2>/dev/null || true
RUN_END=$(date +%s)
echo "Run 5 completed in $((RUN_END - RUN_START))s"
echo ''

# === Run 6/12: P1 | USD | hybrid_exact         | 2025-12-08 → 2025-12-09 ===
echo ''
echo '============================================'
echo 'Run 6/12: P1 | USD | hybrid_exact         | 2025-12-08 → 2025-12-09'
echo '============================================'
RUN_START=$(date +%s)
mkdir -p results/usd/phase1_modes/hybrid_exact/2025-12-08
cd $BASEDIR
python results/_run_006.py 2>&1 | tee results/usd/phase1_modes/hybrid_exact/2025-12-08/log.txt

# Move outputs
[ -f "$BASEDIR/amcc_calibration_results.pt" ] && mv "$BASEDIR/amcc_calibration_results.pt" results/usd/phase1_modes/hybrid_exact/2025-12-08/ 2>/dev/null || true
[ -f "$BASEDIR/amcc_convergence.png" ] && mv "$BASEDIR/amcc_convergence.png" results/usd/phase1_modes/hybrid_exact/2025-12-08/ 2>/dev/null || true
[ -f "$BASEDIR/amcc_gradient_norms.png" ] && mv "$BASEDIR/amcc_gradient_norms.png" results/usd/phase1_modes/hybrid_exact/2025-12-08/ 2>/dev/null || true
[ -f "$BASEDIR/amcc_gradient_comparison.png" ] && mv "$BASEDIR/amcc_gradient_comparison.png" results/usd/phase1_modes/hybrid_exact/2025-12-08/ 2>/dev/null || true
[ -f "$BASEDIR/amcc_smile_fits.png" ] && mv "$BASEDIR/amcc_smile_fits.png" results/usd/phase1_modes/hybrid_exact/2025-12-08/ 2>/dev/null || true
[ -f "$BASEDIR/amcc_smile_fits_oos.png" ] && mv "$BASEDIR/amcc_smile_fits_oos.png" results/usd/phase1_modes/hybrid_exact/2025-12-08/ 2>/dev/null || true
[ -f "$BASEDIR/amcc_correlation.png" ] && mv "$BASEDIR/amcc_correlation.png" results/usd/phase1_modes/hybrid_exact/2025-12-08/ 2>/dev/null || true
[ -f "$BASEDIR/roughness_ablation_results.pt" ] && mv "$BASEDIR/roughness_ablation_results.pt" results/usd/phase1_modes/hybrid_exact/2025-12-08/ 2>/dev/null || true
[ -f "$BASEDIR/roughness_ablation_smiles.png" ] && mv "$BASEDIR/roughness_ablation_smiles.png" results/usd/phase1_modes/hybrid_exact/2025-12-08/ 2>/dev/null || true
[ -f "$BASEDIR/roughness_ablation_rmse_bars.png" ] && mv "$BASEDIR/roughness_ablation_rmse_bars.png" results/usd/phase1_modes/hybrid_exact/2025-12-08/ 2>/dev/null || true
RUN_END=$(date +%s)
echo "Run 6 completed in $((RUN_END - RUN_START))s"
echo ''

# === Run 7/12: P1 | USD | two_stage            | 2024-12-09 → 2024-12-10 ===
echo ''
echo '============================================'
echo 'Run 7/12: P1 | USD | two_stage            | 2024-12-09 → 2024-12-10'
echo '============================================'
RUN_START=$(date +%s)
mkdir -p results/usd/phase1_modes/two_stage/2024-12-09
cd $BASEDIR
python results/_run_007.py 2>&1 | tee results/usd/phase1_modes/two_stage/2024-12-09/log.txt

# Move outputs
[ -f "$BASEDIR/amcc_calibration_results.pt" ] && mv "$BASEDIR/amcc_calibration_results.pt" results/usd/phase1_modes/two_stage/2024-12-09/ 2>/dev/null || true
[ -f "$BASEDIR/amcc_convergence.png" ] && mv "$BASEDIR/amcc_convergence.png" results/usd/phase1_modes/two_stage/2024-12-09/ 2>/dev/null || true
[ -f "$BASEDIR/amcc_gradient_norms.png" ] && mv "$BASEDIR/amcc_gradient_norms.png" results/usd/phase1_modes/two_stage/2024-12-09/ 2>/dev/null || true
[ -f "$BASEDIR/amcc_gradient_comparison.png" ] && mv "$BASEDIR/amcc_gradient_comparison.png" results/usd/phase1_modes/two_stage/2024-12-09/ 2>/dev/null || true
[ -f "$BASEDIR/amcc_smile_fits.png" ] && mv "$BASEDIR/amcc_smile_fits.png" results/usd/phase1_modes/two_stage/2024-12-09/ 2>/dev/null || true
[ -f "$BASEDIR/amcc_smile_fits_oos.png" ] && mv "$BASEDIR/amcc_smile_fits_oos.png" results/usd/phase1_modes/two_stage/2024-12-09/ 2>/dev/null || true
[ -f "$BASEDIR/amcc_correlation.png" ] && mv "$BASEDIR/amcc_correlation.png" results/usd/phase1_modes/two_stage/2024-12-09/ 2>/dev/null || true
[ -f "$BASEDIR/roughness_ablation_results.pt" ] && mv "$BASEDIR/roughness_ablation_results.pt" results/usd/phase1_modes/two_stage/2024-12-09/ 2>/dev/null || true
[ -f "$BASEDIR/roughness_ablation_smiles.png" ] && mv "$BASEDIR/roughness_ablation_smiles.png" results/usd/phase1_modes/two_stage/2024-12-09/ 2>/dev/null || true
[ -f "$BASEDIR/roughness_ablation_rmse_bars.png" ] && mv "$BASEDIR/roughness_ablation_rmse_bars.png" results/usd/phase1_modes/two_stage/2024-12-09/ 2>/dev/null || true
RUN_END=$(date +%s)
echo "Run 7 completed in $((RUN_END - RUN_START))s"
echo ''

# === Run 8/12: P1 | USD | two_stage            | 2025-12-08 → 2025-12-09 ===
echo ''
echo '============================================'
echo 'Run 8/12: P1 | USD | two_stage            | 2025-12-08 → 2025-12-09'
echo '============================================'
RUN_START=$(date +%s)
mkdir -p results/usd/phase1_modes/two_stage/2025-12-08
cd $BASEDIR
python results/_run_008.py 2>&1 | tee results/usd/phase1_modes/two_stage/2025-12-08/log.txt

# Move outputs
[ -f "$BASEDIR/amcc_calibration_results.pt" ] && mv "$BASEDIR/amcc_calibration_results.pt" results/usd/phase1_modes/two_stage/2025-12-08/ 2>/dev/null || true
[ -f "$BASEDIR/amcc_convergence.png" ] && mv "$BASEDIR/amcc_convergence.png" results/usd/phase1_modes/two_stage/2025-12-08/ 2>/dev/null || true
[ -f "$BASEDIR/amcc_gradient_norms.png" ] && mv "$BASEDIR/amcc_gradient_norms.png" results/usd/phase1_modes/two_stage/2025-12-08/ 2>/dev/null || true
[ -f "$BASEDIR/amcc_gradient_comparison.png" ] && mv "$BASEDIR/amcc_gradient_comparison.png" results/usd/phase1_modes/two_stage/2025-12-08/ 2>/dev/null || true
[ -f "$BASEDIR/amcc_smile_fits.png" ] && mv "$BASEDIR/amcc_smile_fits.png" results/usd/phase1_modes/two_stage/2025-12-08/ 2>/dev/null || true
[ -f "$BASEDIR/amcc_smile_fits_oos.png" ] && mv "$BASEDIR/amcc_smile_fits_oos.png" results/usd/phase1_modes/two_stage/2025-12-08/ 2>/dev/null || true
[ -f "$BASEDIR/amcc_correlation.png" ] && mv "$BASEDIR/amcc_correlation.png" results/usd/phase1_modes/two_stage/2025-12-08/ 2>/dev/null || true
[ -f "$BASEDIR/roughness_ablation_results.pt" ] && mv "$BASEDIR/roughness_ablation_results.pt" results/usd/phase1_modes/two_stage/2025-12-08/ 2>/dev/null || true
[ -f "$BASEDIR/roughness_ablation_smiles.png" ] && mv "$BASEDIR/roughness_ablation_smiles.png" results/usd/phase1_modes/two_stage/2025-12-08/ 2>/dev/null || true
[ -f "$BASEDIR/roughness_ablation_rmse_bars.png" ] && mv "$BASEDIR/roughness_ablation_rmse_bars.png" results/usd/phase1_modes/two_stage/2025-12-08/ 2>/dev/null || true
RUN_END=$(date +%s)
echo "Run 8 completed in $((RUN_END - RUN_START))s"
echo ''

# === Run 9/12: P2 | USD | roughness           | 2024-12-09 ===
echo ''
echo '============================================'
echo 'Run 9/12: P2 | USD | roughness           | 2024-12-09'
echo '============================================'
RUN_START=$(date +%s)
mkdir -p results/usd/phase2_roughness/roughness/2024-12-09
cd $BASEDIR
python results/_run_009.py 2>&1 | tee results/usd/phase2_roughness/roughness/2024-12-09/log.txt

# Move outputs
[ -f "$BASEDIR/amcc_calibration_results.pt" ] && mv "$BASEDIR/amcc_calibration_results.pt" results/usd/phase2_roughness/roughness/2024-12-09/ 2>/dev/null || true
[ -f "$BASEDIR/amcc_convergence.png" ] && mv "$BASEDIR/amcc_convergence.png" results/usd/phase2_roughness/roughness/2024-12-09/ 2>/dev/null || true
[ -f "$BASEDIR/amcc_gradient_norms.png" ] && mv "$BASEDIR/amcc_gradient_norms.png" results/usd/phase2_roughness/roughness/2024-12-09/ 2>/dev/null || true
[ -f "$BASEDIR/amcc_gradient_comparison.png" ] && mv "$BASEDIR/amcc_gradient_comparison.png" results/usd/phase2_roughness/roughness/2024-12-09/ 2>/dev/null || true
[ -f "$BASEDIR/amcc_smile_fits.png" ] && mv "$BASEDIR/amcc_smile_fits.png" results/usd/phase2_roughness/roughness/2024-12-09/ 2>/dev/null || true
[ -f "$BASEDIR/amcc_smile_fits_oos.png" ] && mv "$BASEDIR/amcc_smile_fits_oos.png" results/usd/phase2_roughness/roughness/2024-12-09/ 2>/dev/null || true
[ -f "$BASEDIR/amcc_correlation.png" ] && mv "$BASEDIR/amcc_correlation.png" results/usd/phase2_roughness/roughness/2024-12-09/ 2>/dev/null || true
[ -f "$BASEDIR/roughness_ablation_results.pt" ] && mv "$BASEDIR/roughness_ablation_results.pt" results/usd/phase2_roughness/roughness/2024-12-09/ 2>/dev/null || true
[ -f "$BASEDIR/roughness_ablation_smiles.png" ] && mv "$BASEDIR/roughness_ablation_smiles.png" results/usd/phase2_roughness/roughness/2024-12-09/ 2>/dev/null || true
[ -f "$BASEDIR/roughness_ablation_rmse_bars.png" ] && mv "$BASEDIR/roughness_ablation_rmse_bars.png" results/usd/phase2_roughness/roughness/2024-12-09/ 2>/dev/null || true
RUN_END=$(date +%s)
echo "Run 9 completed in $((RUN_END - RUN_START))s"
echo ''

# === Run 10/12: P2 | USD | roughness           | 2025-12-08 ===
echo ''
echo '============================================'
echo 'Run 10/12: P2 | USD | roughness           | 2025-12-08'
echo '============================================'
RUN_START=$(date +%s)
mkdir -p results/usd/phase2_roughness/roughness/2025-12-08
cd $BASEDIR
python results/_run_010.py 2>&1 | tee results/usd/phase2_roughness/roughness/2025-12-08/log.txt

# Move outputs
[ -f "$BASEDIR/amcc_calibration_results.pt" ] && mv "$BASEDIR/amcc_calibration_results.pt" results/usd/phase2_roughness/roughness/2025-12-08/ 2>/dev/null || true
[ -f "$BASEDIR/amcc_convergence.png" ] && mv "$BASEDIR/amcc_convergence.png" results/usd/phase2_roughness/roughness/2025-12-08/ 2>/dev/null || true
[ -f "$BASEDIR/amcc_gradient_norms.png" ] && mv "$BASEDIR/amcc_gradient_norms.png" results/usd/phase2_roughness/roughness/2025-12-08/ 2>/dev/null || true
[ -f "$BASEDIR/amcc_gradient_comparison.png" ] && mv "$BASEDIR/amcc_gradient_comparison.png" results/usd/phase2_roughness/roughness/2025-12-08/ 2>/dev/null || true
[ -f "$BASEDIR/amcc_smile_fits.png" ] && mv "$BASEDIR/amcc_smile_fits.png" results/usd/phase2_roughness/roughness/2025-12-08/ 2>/dev/null || true
[ -f "$BASEDIR/amcc_smile_fits_oos.png" ] && mv "$BASEDIR/amcc_smile_fits_oos.png" results/usd/phase2_roughness/roughness/2025-12-08/ 2>/dev/null || true
[ -f "$BASEDIR/amcc_correlation.png" ] && mv "$BASEDIR/amcc_correlation.png" results/usd/phase2_roughness/roughness/2025-12-08/ 2>/dev/null || true
[ -f "$BASEDIR/roughness_ablation_results.pt" ] && mv "$BASEDIR/roughness_ablation_results.pt" results/usd/phase2_roughness/roughness/2025-12-08/ 2>/dev/null || true
[ -f "$BASEDIR/roughness_ablation_smiles.png" ] && mv "$BASEDIR/roughness_ablation_smiles.png" results/usd/phase2_roughness/roughness/2025-12-08/ 2>/dev/null || true
[ -f "$BASEDIR/roughness_ablation_rmse_bars.png" ] && mv "$BASEDIR/roughness_ablation_rmse_bars.png" results/usd/phase2_roughness/roughness/2025-12-08/ 2>/dev/null || true
RUN_END=$(date +%s)
echo "Run 10 completed in $((RUN_END - RUN_START))s"
echo ''

# === Run 11/12: P2 | USD | cross                | 2024-12-09 ===
echo ''
echo '============================================'
echo 'Run 11/12: P2 | USD | cross                | 2024-12-09'
echo '============================================'
RUN_START=$(date +%s)
mkdir -p results/usd/phase2_cross/cross/2024-12-09
cd $BASEDIR
python results/_run_011.py 2>&1 | tee results/usd/phase2_cross/cross/2024-12-09/log.txt

# Move outputs
[ -f "$BASEDIR/amcc_calibration_results.pt" ] && mv "$BASEDIR/amcc_calibration_results.pt" results/usd/phase2_cross/cross/2024-12-09/ 2>/dev/null || true
[ -f "$BASEDIR/amcc_convergence.png" ] && mv "$BASEDIR/amcc_convergence.png" results/usd/phase2_cross/cross/2024-12-09/ 2>/dev/null || true
[ -f "$BASEDIR/amcc_gradient_norms.png" ] && mv "$BASEDIR/amcc_gradient_norms.png" results/usd/phase2_cross/cross/2024-12-09/ 2>/dev/null || true
[ -f "$BASEDIR/amcc_gradient_comparison.png" ] && mv "$BASEDIR/amcc_gradient_comparison.png" results/usd/phase2_cross/cross/2024-12-09/ 2>/dev/null || true
[ -f "$BASEDIR/amcc_smile_fits.png" ] && mv "$BASEDIR/amcc_smile_fits.png" results/usd/phase2_cross/cross/2024-12-09/ 2>/dev/null || true
[ -f "$BASEDIR/amcc_smile_fits_oos.png" ] && mv "$BASEDIR/amcc_smile_fits_oos.png" results/usd/phase2_cross/cross/2024-12-09/ 2>/dev/null || true
[ -f "$BASEDIR/amcc_correlation.png" ] && mv "$BASEDIR/amcc_correlation.png" results/usd/phase2_cross/cross/2024-12-09/ 2>/dev/null || true
[ -f "$BASEDIR/roughness_ablation_results.pt" ] && mv "$BASEDIR/roughness_ablation_results.pt" results/usd/phase2_cross/cross/2024-12-09/ 2>/dev/null || true
[ -f "$BASEDIR/roughness_ablation_smiles.png" ] && mv "$BASEDIR/roughness_ablation_smiles.png" results/usd/phase2_cross/cross/2024-12-09/ 2>/dev/null || true
[ -f "$BASEDIR/roughness_ablation_rmse_bars.png" ] && mv "$BASEDIR/roughness_ablation_rmse_bars.png" results/usd/phase2_cross/cross/2024-12-09/ 2>/dev/null || true
RUN_END=$(date +%s)
echo "Run 11 completed in $((RUN_END - RUN_START))s"
echo ''

# === Run 12/12: P2 | USD | cross                | 2025-12-08 ===
echo ''
echo '============================================'
echo 'Run 12/12: P2 | USD | cross                | 2025-12-08'
echo '============================================'
RUN_START=$(date +%s)
mkdir -p results/usd/phase2_cross/cross/2025-12-08
cd $BASEDIR
python results/_run_012.py 2>&1 | tee results/usd/phase2_cross/cross/2025-12-08/log.txt

# Move outputs
[ -f "$BASEDIR/amcc_calibration_results.pt" ] && mv "$BASEDIR/amcc_calibration_results.pt" results/usd/phase2_cross/cross/2025-12-08/ 2>/dev/null || true
[ -f "$BASEDIR/amcc_convergence.png" ] && mv "$BASEDIR/amcc_convergence.png" results/usd/phase2_cross/cross/2025-12-08/ 2>/dev/null || true
[ -f "$BASEDIR/amcc_gradient_norms.png" ] && mv "$BASEDIR/amcc_gradient_norms.png" results/usd/phase2_cross/cross/2025-12-08/ 2>/dev/null || true
[ -f "$BASEDIR/amcc_gradient_comparison.png" ] && mv "$BASEDIR/amcc_gradient_comparison.png" results/usd/phase2_cross/cross/2025-12-08/ 2>/dev/null || true
[ -f "$BASEDIR/amcc_smile_fits.png" ] && mv "$BASEDIR/amcc_smile_fits.png" results/usd/phase2_cross/cross/2025-12-08/ 2>/dev/null || true
[ -f "$BASEDIR/amcc_smile_fits_oos.png" ] && mv "$BASEDIR/amcc_smile_fits_oos.png" results/usd/phase2_cross/cross/2025-12-08/ 2>/dev/null || true
[ -f "$BASEDIR/amcc_correlation.png" ] && mv "$BASEDIR/amcc_correlation.png" results/usd/phase2_cross/cross/2025-12-08/ 2>/dev/null || true
[ -f "$BASEDIR/roughness_ablation_results.pt" ] && mv "$BASEDIR/roughness_ablation_results.pt" results/usd/phase2_cross/cross/2025-12-08/ 2>/dev/null || true
[ -f "$BASEDIR/roughness_ablation_smiles.png" ] && mv "$BASEDIR/roughness_ablation_smiles.png" results/usd/phase2_cross/cross/2025-12-08/ 2>/dev/null || true
[ -f "$BASEDIR/roughness_ablation_rmse_bars.png" ] && mv "$BASEDIR/roughness_ablation_rmse_bars.png" results/usd/phase2_cross/cross/2025-12-08/ 2>/dev/null || true
RUN_END=$(date +%s)
echo "Run 12 completed in $((RUN_END - RUN_START))s"
echo ''

TOTAL_END=$(date +%s)
TOTAL=$((TOTAL_END - TOTAL_START))
echo '=========================================='
echo "ALL ${0} RUNS COMPLETE in ${TOTAL}s ($(( TOTAL / 60 )) min)"
echo '=========================================='