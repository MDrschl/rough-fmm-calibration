# Fast Swaption Calibration via Automatic Differentiation in a Mapped Rough SABR FMM

This repository implements the calibration of the **Mapped Rough SABR Forward Market Model** to swaption surfaces via **Automatic Monte Carlo Calibration (AMCC)**. It extends the AMCC framework of [Gonon & Stockinger (2025)](https://doi.org/10.1145/3677052.3698605) from the single-asset equity setting to the multi-rate interest rate setting of the Rough SABR FMM proposed by [Adachi et al. (2025)](https://arxiv.org/abs/2509.25975). To our knowledge, this is the first application of AMCC to a rough stochastic volatility forward market model.

The entire pricing pipeline — from unconstrained parameters through FMM aggregation, Monte Carlo simulation, and swaption pricing — is embedded in a single differentiable computational graph in PyTorch. Gradients of the vega-weighted calibration loss flow back through the aggregation into all 79 model parameters in a single backward pass, replacing the gradient-free optimisation and nested Brent root-finding of the original implementation with an estimated speedup of approximately three orders of magnitude.

---

## Repository structure

```
rough-fmm-calibration/
├── main.py                    # Model core: simulation, pricing, parameter module
├── calibration.py             # Calibration modes, α matching, diagnostics
├── preprocessing.py           # Raw Bloomberg data → calibration-ready .pkl
├── run_all_calibrations.py    # Master orchestration: generates driver scripts + run_all.sh
├── dataUSD/                   # Raw USD SOFR swaption data (Bloomberg London Close)
│   ├── Dez2024IV.xlsx
│   ├── Dez2024SOFR.xlsx
│   ├── Dez2025IV.xlsx
│   └── Dez2025SOFR.xlsx
├── dataEUR/                   # Raw EUR ESTR swaption data
│   ├── Dez2024IV.xlsx
│   ├── Dez2024ESTR.xlsx
│   ├── Dez2025IV.xlsx
│   └── Dez2025ESTR.xlsx
├── usd_swaption_data.pkl      # Preprocessed USD data
├── eur_swaption_data.pkl      # Preprocessed EUR data
└── requirements.txt
```

---

## Model

The model is a **Mapped Rough SABR FMM** on an annual tenor grid $T_0, \ldots, T_N$ with $N = 11$. Each forward term rate $R^j$ follows rough SABR dynamics under an HJM-consistent framework, with variance driven by a Volterra process with Hurst parameter $H \in (0, \tfrac{1}{2})$. Through the FMM aggregation and freezing approximation, each swaption is priced under a mapped single-rate rough Bergomi model with effective parameters $v(0)$, $\rho_{\mathrm{eff}}$, and variance curve $v(t)/v(0)$.

The FMM aggregation determines the forward variance curve as a known, closed-form function of the finite parameter vector $\theta$, involving the Volterra–gamma integral which admits an analytical expression. This eliminates the need for the neural network approximation of the forward variance curve employed by Gonon & Stockinger in the equity setting, reducing the calibration from an infinite-dimensional problem (learn the shape of $v$) to a finite-dimensional one (learn $\theta$) without function-approximation error.

### Parameters

| Symbol | Description | Dim |
|--------|-------------|-----|
| $H$ | Hurst / roughness parameter | 1 |
| $\eta$ | Vol-of-vol (rough Bergomi convention, $\kappa = \eta\sqrt{2H}$) | 1 |
| $\alpha_j$ | Variance levels | $N$ |
| $\rho_{0,j}$ | Spot–vol correlations | $N$ |
| $\rho_{ij}$ | Inter-rate correlations (Rapisarda angles $\omega$) | $N(N-1)/2$ |

Total: $n = 79$ parameters for $N = 11$. Positive semi-definiteness of the correlation matrix is guaranteed throughout optimisation via the Rapisarda angular parametrisation.

---

## Data

### Format

Both `dataUSD/` and `dataEUR/` follow the Bloomberg London Close layout:

- `Dez{Year}IV.xlsx` — implied volatility sheets (ATM + OTM). All IVs are **Bachelier (normal) vol in basis points**.
- `Dez{Year}SOFR.xlsx` (USD) / `Dez{Year}ESTR.xlsx` (EUR) — swap rates for bootstrapping discount factors.

### Swaption grid

**USD:** 19 swaptions per date. Expiries $\{1, 3, 5, 7, 10\}$Y crossed with tenors such that $I + \text{tenor} \leq 11$.

**EUR:** 16 swaptions per date. Expiries $\{1, 2, 3, 5, 7, 10\}$Y with a sparser tenor grid (missing 3Y and 7Y tenor swaptions due to thinner ESTR swaption liquidity).

Strikes: ATM $\pm$ 200/100/50/25 bp (9 strikes per smile).

### Available dates

| Currency | Dates |
|----------|-------|
| USD SOFR | 2024-12-09, 2024-12-10, 2024-12-11, 2025-12-08, 2025-12-09, 2025-12-10 |
| EUR ESTR | same date range |

---

## Setup

```bash
conda create -n amcc python=3.12
conda activate amcc
pip install -r requirements.txt
```

### Requirements

```
numpy>=1.24
torch>=2.0
scipy>=1.10
matplotlib>=3.7
py_lets_be_rational>=1.0
```

---

## Preprocessing

```bash
python preprocessing.py --currency usd    # generates usd_swaption_data.pkl
python preprocessing.py --currency eur    # generates eur_swaption_data.pkl
```

Bootstraps discount factors, converts Bachelier IVs to Black IVs, and computes frozen weights $\Pi_0^j$ and normalised weights $\pi_j$.

---

## Calibration

```bash
python calibration.py
```

Configure via the `CONFIG` dict at the top of `calibration.py`:

```python
CONFIG = {
    "data_file":       "usd_swaption_data.pkl",
    "in_sample_date":  "2024-12-09",
    "out_sample_date": "2024-12-10",
    "mode":            "hybrid_two_stage",
    "device":          "cpu",
    "dtype":           "float64",
    "antithetic":      True,
    "crn_seed":        42,
}
```

### Simulation schemes

Three schemes for the fractional Brownian motion, each at a different point in the accuracy–differentiability trade-off:

| Scheme | $H$ differentiable? | fBm bias | Use case |
|--------|-------------------|----------|----------|
| Exact Cholesky | No | None | Stage 2 with $H$ frozen; diagnostics |
| Approximate Riemann | Yes | High near singularity | Baseline comparison |
| Hybrid (Bennedsen et al.) | Yes | Low | **Recommended default** |

### Calibration modes

| Mode | Description |
|------|-------------|
| `hybrid` | Single-stage hybrid, all parameters free |
| `hybrid_two_stage` | **Primary.** Stage 1: simplified variance curve, $H$ free. Stage 2: full variance curve, $H$ frozen |
| `hybrid_exact` | Stage 1: hybrid. Stage 2: exact Cholesky ($H$ frozen) |
| `two_stage` | Stage 1: approximate Riemann. Stage 2: exact Cholesky |
| `cross` | Train/test split cross-validation |
| `roughness` | Ablation: re-calibrate at fixed $H \in \{0.05, 0.10, \ldots, 0.50\}$ |

### Cross-validation

**USD** (6 held-out): 1Y×3Y, 1Y×7Y, 3Y×2Y, 3Y×7Y, 5Y×3Y, 7Y×2Y.

**EUR:** Auto-selected (5 held-out): 1Y×5Y, 2Y×5Y, 3Y×5Y, 5Y×5Y, 7Y×2Y — since only 2 of the USD test keys exist in the EUR grid.

### Variance reduction

**Common random numbers** and **antithetic variates** are applied throughout. CRN seeds each swaption deterministically per iteration; antithetic variates mirror $N/2$ paths, doubling effective paths at no extra cost.

### Out-of-sample evaluation

All parameters except $\alpha$ are frozen at calibrated values. $\alpha$ is fine-tuned from the calibrated starting point via a short gradient pass (80 iterations). The diagnostic pass uses the same simulation scheme as calibration to avoid scheme-mismatch bias.

---

## Full analysis

```bash
# Preview all 20 runs
python run_all_calibrations.py --device cuda --dtype float64 --dry-run

# Generate and run
python run_all_calibrations.py --device cuda --dtype float64
PYTHONPATH=. PYTHONUNBUFFERED=1 bash run_all.sh
```

`PYTHONPATH=.` is required for the generated driver scripts. `PYTHONUNBUFFERED=1` enables real-time log output.

### Selective runs

```bash
python run_all_calibrations.py --device cpu --dtype float64 --phase 1 --currency usd
PYTHONPATH=. bash run_all.sh

# Individual run
PYTHONPATH=. python results/_run_001.py
```

### Phase 1: Mode comparison (12 runs)
- 3 modes × 2 dates × 2 currencies
- Compares IS RMSE, OOS RMSE, wall time, calibrated parameters

### Phase 2: Deep analysis (8 runs)
- 4 roughness ablations (requires Phase 1 base results)
- 4 cross-validations

**Total: 20 runs.** Estimated ~13 hrs on GPU, ~76 hrs on CPU.

### Output structure

```
results/
  usd/
    phase1_modes/{hybrid_two_stage,hybrid_exact,two_stage}/{2024-12-09,2025-12-08}/
    phase2_roughness/roughness/{2024-12-09,2025-12-08}/
    phase2_cross/cross/{2024-12-09,2025-12-08}/
  eur/
    ...
```

### Monitoring

```bash
tail -f results/usd/phase1_modes/hybrid_two_stage/2024-12-09/log.txt
```

---

## Numerical stability

A `log_S` clamp (`torch.clamp(log_S, min=-20.0, max=20.0)`) is applied in all simulation functions to prevent `exp` overflow on EUR Dec 2024 data, where high $\alpha$ values (~0.39) combined with low rates (~1.7%) produce extreme variance paths. The clamp is a no-op for USD.

---

## Outputs

| File | Contents |
|------|----------|
| `amcc_calibration_results.pt` | Calibrated parameters, loss history, config |
| `amcc_smile_fits.png` | In-sample smile fits |
| `amcc_smile_fits_oos.png` | Out-of-sample smile fits |
| `amcc_convergence.png` | Loss and RMSE convergence |
| `amcc_gradient_norms.png` | Per-parameter gradient norms |
| `amcc_correlation.png` | Calibrated correlation matrix |
| `roughness_ablation_results.pt` | RMSE by fixed $H$ |

```python
import torch
results = torch.load("amcc_calibration_results.pt", weights_only=False)
print(f"H = {results['H']:.4f},  η = {results['eta']:.4f}")
print(f"α = {results['alpha']}")
```

---

## Forward variance curves

**Simplified** (Stage 1):
$$\xi_j(t) = \alpha_j^2 \exp\!\left(\frac{\kappa^2 t^{2H}}{8H}\right)$$

**Full** (Stage 2):
$$\xi_j(t) = \alpha_j^2 \exp\!\left\{\frac{\kappa^2 t^{2H}}{8H} - \sum_{i=j+1}^{N} \frac{\theta_i R_0^i}{1+\theta_i R_0^i}\,\alpha_i\,\rho_{0,i}\,\kappa \int_0^t (t{-}s)^{H-1/2}\gamma_i(s)\,\mathrm{d}s \right\}$$

The full curve lies strictly above the simplified one ($\rho_{0,i} \leq 0$). When Stage 2 activates the full curve, $\alpha$ self-corrects via gradient descent.

---

## References

- Adachi, R., Fukasawa, M., Iida, N., Ikeda, M., Nakatsu, Y., Tsurumi, R. & Yamakami, T. (2025). *Rough SABR Forward Market Model*.
- Gonon, L. & Stockinger, W. (2025). *Leveraging Deep Learning Optimization for Monte Carlo Calibration of (Rough) Stochastic Volatility Models.* Proc. 6th ACM ICAIF.
- Bennedsen, M., Lunde, A. & Pakkanen, M. S. (2017). *Hybrid scheme for Brownian semistationary processes.* Finance and Stochastics.
- Rapisarda, F., Brigo, D. & Mercurio, F. (2007). *Parameterizing correlations: a geometric interpretation.* IMA J. Management Mathematics.
- Lyashenko, A. & Mercurio, F. (2019). *Looking forward to backward-looking rates.* SSRN 3330240.
- Bayer, C., Friz, P. & Gatheral, J. (2016). *Pricing under rough volatility.* Quantitative Finance.
- Hagan, P. S., Kumar, D., Lesniewski, A. S. & Woodward, D. E. (2002). *Managing smile risk.* The Best of Wilmott.