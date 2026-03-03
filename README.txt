# Rough SABR Forward Market Model — Automatic Monte Carlo Calibration

Implementation and calibration of the Mapped Rough SABR Forward Market Model (FMM) from Adachi et al. (2025), calibrated to USD SOFR swaption volatility surfaces from Bloomberg.

Master thesis, Maximilian Droschl, 2025.

## Model

The Mapped Rough SABR FMM extends the classical SABR model in two directions: rough volatility (fractional Brownian motion with Hurst exponent H < 0.5) and multi-factor forward rate dynamics. The model jointly prices swaptions across the full expiry x tenor grid {1Y, 3Y, 5Y, 7Y, 10Y} x {1Y, 2Y, 3Y, 5Y, 7Y, 10Y} using a single set of global parameters.

**Parameters:**
- H in (0, 0.5) — Hurst exponent (roughness)
- eta in (0, infinity) — vol-of-vol
- alpha_j in (0, infinity) — variance level per forward rate, j = 1, ..., N
- rho_0j in (-1, 0) — spot-vol correlations
- rho_ij — forward-rate correlation matrix (Rapisarda parametrisation)

Calibration is gradient-based (Adam) using differentiable Monte Carlo pricing with common random numbers for variance reduction.

## Setup

```bash
conda create -n amcc python=3.12 -y
conda activate amcc

# PyTorch (CPU — for CUDA see https://pytorch.org/get-started)
pip install torch --index-url https://download.pytorch.org/whl/cpu

# Remaining dependencies
pip install -r requirements.txt
```

## Data

Market data is preprocessed from Bloomberg London Closing quotes (SOFR swap rates + swaption normal vols) into a single pickle:

```bash
python preprocess_usd_swaptions.py
```

This reads Excel files from `data/` and writes `usd_swaption_data.pkl`. The pickle contains discount factors, forward rates, and Bachelier IVs for multiple dates. Black IV conversion happens at load time.

## Usage

### Main calibration

```bash
python calibration.py
```

Two modes are available (set `mode` in the config):

- **hybrid** (recommended) — Single-stage BLP scheme. H is differentiable throughout, giving a fully connected computational graph. All parameters are optimised jointly via Adam with cosine annealing.
- **two_stage** — Stage 1 uses the approximate scheme (H differentiable) with ReduceLROnPlateau for exploration. Stage 2 switches to the exact Cholesky scheme with H frozen for refinement.

Evaluates out-of-sample on the next trading day. Outputs: convergence plot, smile fits (in-sample and OOS), correlation matrix heatmap, and calibrated parameters saved to `amcc_calibration_results.pt`.

### Cross-validation

```bash
python calibration_cross.py
```

Holds out a subset of swaptions (spanning expiries and tenors), calibrates on the rest, and evaluates generalisation. Reports train vs test RMSE.

### Roughness ablation

```bash
python calibration_roughness.py
```

Compares H < 0.5 (rough, from two-stage calibration) against H = 0.5 (classical SABR limit). Quantifies the contribution of the power-law memory kernel to smile fitting accuracy.

## Project structure

```
main.py                         Core library: parameters, simulation, pricing, calibration
calibration.py                  Main calibration driver (two-stage / hybrid)
calibration_cross.py            Train/test cross-validation driver
calibration_roughness.py        H < 0.5 vs H = 0.5 ablation driver
preprocess_usd_swaptions.py     Bloomberg data to pickle
requirements.txt                Python dependencies
data/                           Raw Bloomberg Excel files (not tracked)
usd_swaption_data.pkl           Preprocessed market data (generated)
```

## Simulation schemes

The implementation supports three Monte Carlo schemes for the rough volatility process:

- **Exact** — Full Cholesky decomposition of the fBM covariance matrix. O(M squared) memory, exact in distribution, but H is not differentiable.
- **Approximate** — Euler discretisation of the Volterra integral. H is differentiable, enabling gradient-based optimisation of the Hurst exponent.
- **Hybrid (BLP)** — Bennedsen, Lunde & Pakkanen (2017). Near-field Cholesky + far-field quadrature. H differentiable with accuracy close to the exact scheme.

## References

Adachi, R., Fukasawa, M., Iida, N., Ikeda, M., Nakatsu, Y., Tsurumi, R. & Yamakami, T. (2025). *Rough SABR Forward Market Model*. arXiv:2509.25975.

Bennedsen, M., Lunde, A. & Pakkanen, M. S. (2017). Hybrid scheme for Brownian semistationary processes. *Finance and Stochastics*, 21(4), 931-965.

Rapisarda, F., Brigo, D. & Mercurio, F. (2007). Parameterizing correlations: a geometric interpretation. *IMA Journal of Management Mathematics*, 18(1), 55-73.