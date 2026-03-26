# Fast Swaption Calibration via Automatic Differentiation in a Mapped Rough SABR FMM

This repository implements the calibration of the **Mapped Rough SABR Forward Market Model** to swaption surfaces via **Automatic Monte Carlo Calibration (AMCC)**. It extends the AMCC framework of [Gonon & Stockinger (2025)](https://doi.org/10.1145/3677052.3698605) from the single-asset equity setting to the multi-rate interest rate setting of the Rough SABR FMM proposed by [Adachi et al. (2025)](https://arxiv.org/abs/2509.25975). To our knowledge, this is the first application of AMCC to a rough stochastic volatility forward market model.

The entire pricing pipeline — from unconstrained parameters through FMM aggregation, Monte Carlo simulation, and swaption pricing — is embedded in a single differentiable computational graph in PyTorch. Gradients of the vega-weighted calibration loss flow back through the aggregation into all 79 model parameters in a single backward pass. Reverse-mode automatic differentiation obtains the full gradient in a single forward-backward pass, whereas a finite-difference approximation would require $\mathcal{O}(n)$ forward passes, reducing the per-iteration cost by roughly two orders of magnitude for the $n = 79$ parameter model.

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
├── dataEUR/                   # Raw EUR ESTR/EURIBOR swaption data
│   ├── Dez2024IV.xlsx
│   ├── Dez2024ESTR.xlsx
│   ├── Dez2024EURIBOR.xlsx
│   ├── Dez2025IV.xlsx
│   ├── Dez2025ESTR.xlsx
│   └── Dez2025EURIBOR.xlsx
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
- `Dez{Year}SOFR.xlsx` (USD) — SOFR swap rates for bootstrapping (single-curve).
- `Dez{Year}ESTR.xlsx` (EUR) — ESTR OIS rates for bootstrapping discount factors.
- `Dez{Year}EURIBOR.xlsx` (EUR) — EURIBOR swap rates for bootstrapping forward rates.

### Swaption grid

**USD:** 19 swaptions per date. Expiries $\{1, 3, 5, 7, 10\}$Y crossed with tenors such that $I + \text{tenor} \leq 11$.

**EUR:** 16 swaptions per date. Expiries $\{1, 2, 3, 5, 7, 10\}$Y with a sparser tenor grid (missing ×3Y and ×7Y tenor columns, plus 1Y×7Y, due to thinner ESTR swaption liquidity). Some swaptions have 8 strikes instead of 9 because the −200bp offset produces a non-positive strike at low EUR swap rates (~2%), and these are filtered during preprocessing.

Strikes: ATM $\pm$ 200/100/50/25 bp (up to 9 strikes per smile).

### Available dates

| Currency | Dates |
|----------|-------|
| USD SOFR | 2024-12-09, 2024-12-10, 2025-12-08, 2025-12-09 |
| EUR ESTR/EURIBOR | 2024-12-09, 2024-12-10, 2025-12-08, 2025-12-09 |

---

## EUR dual-curve preprocessing

EUR swaptions reference 6M EURIBOR as the floating rate but are collateralised and discounted at ESTR, requiring a **dual-curve** setup. The preprocessing handles this as follows:

- **ESTR curve** → discount factors $P^{\text{ESTR}}(T_j)$ → used for annuities $A_0$, the discounting component of the swap rate, and frozen weights $\Pi_0^j$.
- **EURIBOR curve** → forward term rates $R^j$ → used for the FMM dynamics and normalised weights $\pi_j$.

The dual-curve swap rate is computed explicitly as $S_0 = \sum_j \tau_j P^{\text{ESTR}}(T_j) R_j^{\text{EUR}} / A_0^{\text{ESTR}}$, since the single-curve telescoping identity does not hold when the projection and discounting curves differ. For USD (single-curve), the explicit formula reduces to the standard $(P(T_I) - P(T_J))/A_0$ to machine precision.

### Assumptions and approximations

The EUR implementation retains the annual tenor grid ($\tau_j = 1$, $N = 11$) used for USD, which introduces one approximation:

**Annual EURIBOR forwards instead of semi-annual 6M EURIBOR.** EUR swaptions reference 6M EURIBOR with semi-annual floating payments, but the FMM evolves annual forward rates $R^j$ over $[T_{j-1}, T_j]$ with $\tau_j = 1$. This approximates the semi-annual floating leg by an annual one. The resulting error is the difference between compounding two consecutive 6M forwards and a single annual forward, which is a second-order convexity effect — typically fractions of a basis point on the swap rate, well within the calibration noise. Switching to a semi-annual grid ($\tau_j = 0.5$, $N = 22$) would eliminate this approximation but roughly double the parameter count and correlation matrix dimension.

The EURIBOR-ESTR basis (which ranged from −18bp to +62bp across tenors in December 2024) is handled correctly by the dual-curve bootstrapping and does not enter as an approximation.

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

Bootstraps discount factors, converts Bachelier IVs to Black IVs (filtering strikes where the conversion fails), and computes frozen weights $\Pi_0^j$ and normalised weights $\pi_j$. For EUR, bootstraps ESTR and EURIBOR curves separately and computes the dual-curve swap rate.

---

## Calibration

```bash
python calibration.py
```

Configure via the `CONFIG` dict at the top of `calibration.py`:

```python
CONFIG = {
    "data_file":       "eur_swaption_data.pkl",  # or "usd_swaption_data.pkl"
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

All modes optimise the same 79-parameter model via AMCC. They differ in which fBm simulation scheme is used in each stage, whether $H$ is free or frozen, and which forward variance curve is active.

#### Hybrid two-stage (`hybrid_two_stage`) — recommended

| | Stage 1 | Stage 2 |
|--|---------|---------|
| **fBm scheme** | Hybrid | Hybrid |
| **$H$ status** | Free (differentiable) | Frozen at Stage 1 value |
| **Variance curve** | Simplified: $\xi_j(t) = \alpha_j^2 \exp(\eta^2 t^{2H}/4)$ | Full: includes Volterra–gamma drift corrections |
| **Paths / steps** | 20,000 / 50 | 30,000 / 100 |
| **Iterations** | 400 | 600 |

Stage 1 locates the roughness regime $(H, \eta)$ jointly with all other parameters at lower resolution and under the cheaper simplified variance curve. Stage 2 refines $\eta$, $\alpha$, $\rho_0$, and $\rho_{ij}$ under the full variance curve at higher path count, with $H$ frozen. The two-stage structure reduces total computation time compared to the single-stage mode by reserving the expensive full-curve evaluation for Stage 2, which starts from a well-initialised parameter set and converges more quickly.

#### Hybrid single-stage (`hybrid`)

| | Single stage |
|--|-------------|
| **fBm scheme** | Hybrid |
| **$H$ status** | Free (differentiable) |
| **Variance curve** | Full |
| **Paths / steps** | 30,000 / 100 |
| **Iterations** | 800 |

All 79 parameters are optimised jointly in one run with no stage separation. Simpler but slower to converge, because the expensive full variance curve is evaluated from the start, including during the early high-loss phase where precision matters less than speed.

#### Hybrid–exact (`hybrid_exact`)

| | Stage 1 | Stage 2 |
|--|---------|---------|
| **fBm scheme** | Hybrid | Exact Cholesky |
| **$H$ status** | Free (differentiable) | Frozen at Stage 1 value |
| **Variance curve** | Simplified | Full |
| **Paths / steps** | 20,000 / 50 | 30,000 / 100 |
| **Iterations** | 400 | 600 |

Stage 1 is identical to the hybrid two-stage mode. Stage 2 switches to the exact Cholesky scheme, which introduces no fBm discretisation bias but detaches $H$ from the computational graph (already frozen). This mode isolates the effect of the hybrid scheme's residual approximation error in Stage 2: if the hybrid two-stage mode produces lower RMSE, the hybrid scheme interacts more favourably with the Adam optimiser's momentum state than the exact scheme does.

#### Approximate–exact (`two_stage`)

| | Stage 1 | Stage 2 |
|--|---------|---------|
| **fBm scheme** | Approximate Riemann | Exact Cholesky |
| **$H$ status** | Free (differentiable) | Frozen at Stage 1 value |
| **Variance curve** | Simplified | Full |
| **Paths / steps** | 10,000 / 50 | 30,000 / 50 |
| **Iterations** | 800 | 800 |

The cheapest mode per iteration. Stage 1 uses the left-point Riemann discretisation of the Volterra integral, which keeps $H$ differentiable but introduces a downward variance bias near the kernel singularity. This bias pushes $H$ higher toward the Markovian regime and inflates $\alpha$ to compensate for the missing variance. Stage 2 switches to the exact Cholesky scheme. Serves as a baseline for assessing the benefit of the hybrid near-field treatment in the other modes.

#### Additional modes

| Mode | Description |
|------|-------------|
| `cross` | Train/test split cross-validation. Runs hybrid two-stage on training swaptions, evaluates on held-out instruments. |
| `roughness` | Roughness ablation. Re-calibrates at each fixed $H \in \{0.05, 0.10, \ldots, 0.50\}$ using the exact Cholesky scheme (1,200 iterations each). The case $H = 0.50$ uses the Markovian SABR scheme. |

### Cross-validation

**USD** (6 held-out): 1Y×3Y, 1Y×7Y, 3Y×2Y, 3Y×7Y, 5Y×3Y, 7Y×2Y.

**EUR:** Auto-selected (~5 held-out): since only 2 of the USD test keys exist in the EUR grid, the code automatically selects one multi-rate swaption per expiry bucket, choosing the middle tenor where available, and only from swaptions with smile data.

### Variance reduction

**Common random numbers** and **antithetic variates** are applied throughout. CRN seeds each swaption deterministically per iteration; antithetic variates mirror $N/2$ paths, doubling effective paths at no extra cost.

### Out-of-sample evaluation

All parameters except $\alpha$ are frozen at calibrated values. $\alpha$ is fine-tuned from the calibrated starting point via a short gradient pass (80 iterations). The diagnostic pass uses the same simulation scheme as calibration to avoid scheme-mismatch bias.

---

## Full analysis

```bash
# Preview all runs
python run_all_calibrations.py --device cpu --dtype float64 --dry-run

# Generate driver scripts, then execute
python run_all_calibrations.py --device cpu --dtype float64
PYTHONPATH=. PYTHONUNBUFFERED=1 caffeinate -dims bash run_all.sh
```

`PYTHONPATH=.` is required for the generated driver scripts. `PYTHONUNBUFFERED=1` enables real-time log output. `caffeinate -dims` prevents macOS from sleeping during the run.

### Selective runs

```bash
# USD only
python run_all_calibrations.py --device cpu --dtype float64 --currency usd
PYTHONPATH=. PYTHONUNBUFFERED=1 caffeinate -dims bash run_all.sh

# Phase 1 only, USD only
python run_all_calibrations.py --device cpu --dtype float64 --phase 1 --currency usd
PYTHONPATH=. PYTHONUNBUFFERED=1 caffeinate -dims bash run_all.sh

# Individual run
PYTHONPATH=. python results/_run_001.py
```

### Phase 1: Mode comparison (16 runs)
- 4 modes × 2 dates × 2 currencies
- Compares IS RMSE, OOS RMSE, wall time, calibrated parameters

### Phase 2: Deep analysis (8 runs)
- 4 roughness ablations (requires Phase 1 base results)
- 4 cross-validations

**Total: 24 runs.** Estimated ~14 hrs on GPU, ~86 hrs on CPU.

### Output structure

```
results/
  usd/
    phase1_modes/{hybrid_two_stage,hybrid,hybrid_exact,two_stage}/{2024-12-09,2025-12-08}/
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

A `log_S` clamp (`torch.clamp(log_S, min=-20.0, max=20.0)`) is applied in all simulation functions to prevent `exp` overflow. Strikes where the Bachelier-to-Black IV conversion fails (typically deep OTM strikes at low EUR swap rates) are filtered during preprocessing to prevent NaN propagation into the loss function.

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

## References

- Adachi, R., Fukasawa, M., Iida, N., Ikeda, M., Nakatsu, Y., Tsurumi, R. & Yamakami, T. (2025). *Rough SABR Forward Market Model*.
- Gonon, L. & Stockinger, W. (2025). *Leveraging Deep Learning Optimization for Monte Carlo Calibration of (Rough) Stochastic Volatility Models.* Proc. 6th ACM ICAIF.
- Bennedsen, M., Lunde, A. & Pakkanen, M. S. (2017). *Hybrid scheme for Brownian semistationary processes.* Finance and Stochastics.
- Rapisarda, F., Brigo, D. & Mercurio, F. (2007). *Parameterizing correlations: a geometric interpretation.* IMA J. Management Mathematics.
- Lyashenko, A. & Mercurio, F. (2019). *Looking forward to backward-looking rates.* SSRN 3330240.
- Bayer, C., Friz, P. & Gatheral, J. (2016). *Pricing under rough volatility.* Quantitative Finance.
- Hagan, P. S., Kumar, D., Lesniewski, A. S. & Woodward, D. E. (2002). *Managing smile risk.* The Best of Wilmott.