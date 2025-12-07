# separate_tenor_calibration.py

import numpy as np
from math import log, sqrt, exp
from dataclasses import dataclass
from math import log, sqrt, exp, erf


from preprocessing import USDRFRPreprocessor, SwaptionSmile
from roughBergomiFMM import RoughBergomiForwardSwapPricerHybrid


# ---------- Black helper functions ----------

SQRT_2PI = np.sqrt(2.0 * np.pi)


def _phi(x):
    """Standard normal pdf."""
    return np.exp(-0.5 * x * x) / SQRT_2PI


def _Phi(x):
    """Standard normal cdf (using numpy.erf)."""
    return 0.5 * (1.0 + erf(x / np.sqrt(2.0)))


def black_put(F, K, vol, T):
    """
    Black (1976) put price with numeraire 1 (forward price).
    """
    if vol <= 0.0 or T <= 0.0:
        return max(K - F, 0.0)

    sigma_sqrtT = vol * np.sqrt(T)
    lnFK = np.log(F / K)
    d1 = (lnFK + 0.5 * sigma_sqrtT**2) / sigma_sqrtT
    d2 = d1 - sigma_sqrtT
    P = K * _Phi(-d2) - F * _Phi(-d1)
    return float(P)


def black_normalised_put(F, K, vol, T):
    """
    Normalised put p(k,T) = P(F,K,vol,T) / F, where k = log(K/F).
    """
    return black_put(F, K, vol, T) / F


# ---------- Curve → forward swap rate ----------

def forward_par_swap_rate(pre: USDRFRPreprocessor,
                          start: float,
                          tenor: float,
                          pay_freq: int = 1) -> float:
    """
    Forward par swap rate of a swap starting at 'start' with length 'tenor'
    and 'pay_freq' coupon payments per year (here 1 = annual).

    For a 1Y tenor and annual payments, this is the 1Y forward OIS rate over [start, start+1].
    """
    delta = 1.0 / pay_freq
    n_pay = int(round(tenor * pay_freq))

    pay_dates = start + delta * np.arange(1, n_pay + 1)
    dfs = np.array([pre.discount_factor(t) for t in pay_dates])

    df_start = pre.discount_factor(start)
    df_end = dfs[-1]

    annuity = delta * dfs.sum()
    rate = (df_start - df_end) / annuity
    return float(rate)


# ---------- Golden-section search for 1D κ-calibration ----------

def golden_section_minimise(f, a, b, tol=1e-2, max_iter=40):
    """
    Simple, derivative-free 1D minimiser for κ.
    Assumes f is unimodal on [a,b].
    """
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    invphi = 1.0 / phi

    c = b - invphi * (b - a)
    d = a + invphi * (b - a)
    fc = f(c)
    fd = f(d)

    for _ in range(max_iter):
        if abs(b - a) < tol:
            break
        if fc < fd:
            b, fd = d, fc
            d = c
            c = b - invphi * (b - a)
            fc = f(c)
        else:
            a, fc = c, fd
            c = d
            d = a + invphi * (b - a)
            fd = f(d)

    return 0.5 * (a + b)


# ---------- κ-calibration for one smile ----------

def calibrate_kappa_for_smile(pre: USDRFRPreprocessor,
                              smile: SwaptionSmile,
                              H: float,
                              rho: float,
                              v0: float,
                              n_steps: int = 200,
                              n_paths: int = 50_000,
                              kappa_bracket=(0.1, 3.0)) -> float:
    """
    For a given smile (expiry, tenor), fixed H, rho, v0, calibrate κ so that
    the Monte Carlo prices of normalised puts p(k,T) match those implied by
    market Black vols.
    """
    T = smile.expiry
    tenor = smile.tenor

    # 1) Forward par swap rate S0* from SOFR curve
    S0 = forward_par_swap_rate(pre, start=T, tenor=tenor, pay_freq=1)

    # 2) Strikes and log-strikes from moneyness in bps (1bp = 1e-4)
    K = S0 + smile.moneyness_bps * 1e-4
    k_log = np.log(K / S0)

    # 3) Market normalised puts from Black vols
    p_mkt = np.array([
        black_normalised_put(S0, Ki, vol_i, T)
        for Ki, vol_i in zip(K, smile.vol)
    ])

    # 4) Objective: L2 error in normalised put space
    def objective(kappa):
        pricer = RoughBergomiForwardSwapPricerHybrid(
            H=H,
            kappa=kappa,
            rho=rho,
            v0=v0,
            S0=S0,
            T=T,
            n_steps=n_steps,
            n_paths=n_paths,
            gamma=0.5,
            seed=None,  # fresh MC each time; you can fix this for smoother surface
        )

        p_model = np.array([
            pricer.price_put_logstrike(k_val) for k_val in k_log
        ])

        err = np.mean((p_model - p_mkt) ** 2)
        return float(err)

    kappa_star = golden_section_minimise(
        objective,
        a=kappa_bracket[0],
        b=kappa_bracket[1],
        tol=1e-2,
        max_iter=40,
    )

    return kappa_star


# ---------- End-to-end replication of Section 6.1 ----------

def run_separate_tenor_calibration():
    # 0) load data and build preprocessor
    pre = USDRFRPreprocessor(
        sofr_path="Data/SOFRRates.xlsx",
        atm_path="Data/ATMSwaptionIVUSD.xlsx",
        otm_path="Data/OTMSwaptionIVUSD.xlsx",
        use_bootstrap=True,
    )

    tenor_years = 1.0
    expiries = np.array([1.0, 3.0, 5.0, 7.0, 10.0])

    # ---- (only for inspection) annual zero curve on the relevant 1Y grid ----
    T_max = expiries.max() + tenor_years
    year_grid = np.arange(1.0, T_max + 1e-8, 1.0)
    annual_curve = pre.curve_on_grid(year_grid)
    print("Annual zero curve used to compute 1Y forward swap rates:")
    print(annual_curve, "\n")

    # collect smiles for the 1Y tenor
    calib_smiles = {}
    for T in expiries:
        sm = pre.get_smile(expiry_years=T, tenor_years=tenor_years)
        if sm is None:
            print(f"No smile for {T}Y x {tenor_years}Y – skipping.")
        else:
            calib_smiles[T] = sm

    # Model parameters that are *not* calibrated in §6.1:
    rho = -0.7

    # Simple choice: set v0 equal to ATM Black variance of the 1Yx1Y smile
    v0 = pre.atm_vols[(1.0, tenor_years)] ** 2

    # H grid as in Section 6.1
    H_values = [0.45, 0.2]

    results = {}  # H -> { T : kappa }

    for H in H_values:
        kappa_row = {}
        print(f"=== Calibrating κ for H = {H:.2f} ===")
        for T, sm in calib_smiles.items():
            print(f"  T={T:>4.1f}Y ...", end="", flush=True)
            kappa_T = calibrate_kappa_for_smile(
                pre=pre,
                smile=sm,
                H=H,
                rho=rho,
                v0=v0,
                n_steps=200,
                n_paths=10_000,
                kappa_bracket=(0.1, 3.0),
            )
            kappa_row[T] = kappa_T
            print(f" κ ≈ {kappa_T:.3f}")
        results[H] = kappa_row
        print()

    return results


# ---------- main: build curve + run calibration ----------

if __name__ == "__main__":
    res = run_separate_tenor_calibration()

    print("\nCalibration summary (κ by H and T):")
    for H in sorted(res.keys()):
        row = res[H]
        parts = [f"T={T:.1f}Y: κ={row[T]:.3f}" for T in sorted(row.keys())]
        print(f"H={H:.2f}:  " + "  ".join(parts))
