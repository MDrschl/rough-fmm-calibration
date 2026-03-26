"""
preprocessing.py
============================
Preprocessing pipeline for swaption data (USD SOFR / EUR EURIBOR).

Transforms Bloomberg London Closing data — swap rates, ATM and OTM
swaption IVs — into calibration-ready tensors for the Mapped Rough SABR FMM.

KEY CONVENTION
--------------
ALL implied volatilities (both ATM and OTM) are **Bachelier / normal vol
in basis points** as quoted by Bloomberg.  The preprocessing stores them
in decimal (bps / 10_000).  Conversion to Black (lognormal) IV is
performed at load time by `load_market_data()`.

Data layout
-----------
Input files (placed in `dataUSD/` or `dataEUR/` subfolder):

    USD (single-curve):
        Dez2024IV.xlsx      sheets: 0912ATM, 0912OTM, ...
        Dez2024SOFR.xlsx    sheets: 0912SOFR, ...

    EUR (dual-curve):
        Dez2024IV.xlsx      sheets: 0912ATM, 0912OTM, ...
        Dez2024ESTR.xlsx    sheets: 0912ESTR, ...     (discounting)
        Dez2024EURIBOR.xlsx sheets: 0912EURIBOR, ...   (forward rates)

Each sheet prefix (e.g. "0912") encodes the date as DDMM.

Output: {currency}_swaption_data.pkl — a dict keyed by date string.

Usage:
    python preprocessing.py                # USD (default)
    python preprocessing.py --currency eur # EUR
"""

import numpy as np
import pandas as pd
import pickle
import argparse
from scipy.stats import norm
from pathlib import Path
from main import black_iv


# =============================================================================
# 0. Configuration
# =============================================================================

# Annual tenor grid for the FMM (Adachi: T_N = 11)
T_N = 11

# Year fraction for annual payments (tau = 1 for annual)
THETA = 1.0

# OTM strike offsets in basis points
OTM_OFFSETS_BPS = [-200, -100, -50, -25, 25, 50, 100, 200]

# ---------- Per-currency configuration ----------

CURRENCY_CONFIG = {
    "usd": {
        "data_dir": Path("dataUSD"),
        "discount_file_suffix": "SOFR",
        "discount_sheet_suffix": "SOFR",
        "projection_file_suffix": None,       # single-curve: same as discount
        "projection_sheet_suffix": None,
        "output_file": "usd_swaption_data.pkl",
        "label": "USD SOFR",
        "dual_curve": False,
    },
    "eur": {
        "data_dir": Path("dataEUR"),
        "discount_file_suffix": "ESTR",       # OIS discounting curve
        "discount_sheet_suffix": "ESTR",
        "projection_file_suffix": "EURIBOR",  # forward rate projection curve
        "projection_sheet_suffix": "EURIBOR",
        "output_file": "eur_swaption_data.pkl",
        "label": "EUR EURIBOR",
        "dual_curve": True,
    },
}

# Date mapping: (filename_stem, sheet_prefix) -> ISO date string
DATE_MAP = [
    # December 2024 cluster
    ("Dez2024", "0912", "2024-12-09"),
    ("Dez2024", "1012", "2024-12-10"),
    # December 2025 cluster
    ("Dez2025", "0812", "2025-12-08"),
    ("Dez2025", "0912", "2025-12-09"),

]

# Default calibration / evaluation dates
DEFAULT_IN_SAMPLE  = "2024-12-09"
DEFAULT_OUT_SAMPLE = "2024-12-10"


# =============================================================================
# 1. Rate curve: parsing and bootstrapping
# =============================================================================

def parse_rate_curve(path: str, sheet: str) -> pd.DataFrame:
    """Parse swap/OIS rates from a Bloomberg export sheet.

    Works for both SOFR, ESTR, and EURIBOR curves.
    """
    df = pd.read_excel(path, sheet_name=sheet)
    df["mid_rate"] = (df["Final Bid Rate"] + df["Final Ask Rate"]) / 2.0

    def to_years(row):
        t, u = row["Term"], row["Unit"]
        if u == "WK":
            return t / 52.0
        elif u == "MO":
            return t / 12.0
        elif u == "YR":
            return float(t)
        else:
            raise ValueError(f"Unknown unit: {u}")

    df["maturity_years"] = df.apply(to_years, axis=1)
    df = df.sort_values("maturity_years").reset_index(drop=True)
    return df[["maturity_years", "mid_rate"]].copy()


def bootstrap_discount_curve(rate_df: pd.DataFrame, max_T: int = 31) -> dict:
    """
    Bootstrap zero-coupon discount factors from swap rates.

    Assumes annual fixed payments (tau = 1):
        s_T * sum_{i=1}^{T} tau * P(T_i) = 1 - P(T)
        => P(T) = (1 - s_T * tau * sum_{i=1}^{T-1} P(T_i)) / (1 + s_T * tau)

    Works for both OIS (SOFR, ESTR) and IBOR (EURIBOR) swap rates,
    as both have annual fixed legs.
    """
    raw_mats = rate_df["maturity_years"].values
    raw_rates = rate_df["mid_rate"].values / 100.0  # % -> decimal

    # Interpolate onto annual grid
    swap_rates = np.zeros(max_T + 1)
    for j in range(1, max_T + 1):
        swap_rates[j] = float(np.interp(j, raw_mats, raw_rates))

    # Bootstrap
    P = np.ones(max_T + 1)
    for j in range(1, max_T + 1):
        s_j = swap_rates[j]
        sum_prev = np.sum(P[1:j]) * THETA
        P[j] = (1.0 - s_j * sum_prev) / (1.0 + s_j * THETA)

    return {"discount_factors": P, "swap_rates_annual": swap_rates}


def compute_forward_term_rates(P: np.ndarray) -> np.ndarray:
    """R_j = (1/tau) * (P(T_{j-1})/P(T_j) - 1)."""
    N = len(P) - 1
    R = np.zeros(N + 1)
    for j in range(1, N + 1):
        R[j] = (P[j - 1] / P[j] - 1.0) / THETA
    return R


# =============================================================================
# 2. IV parsing
# =============================================================================

def parse_expiry_string(s: str) -> float:
    """Convert '1Mo', '18Mo', '1Yr', '1Wk', etc. to years."""
    s = s.strip()
    if s.endswith("Mo"):
        return int(s[:-2]) / 12.0
    elif s.endswith("Yr"):
        return int(s[:-2])
    elif s.endswith("Wk"):
        return int(s[:-2]) / 52.0
    else:
        raise ValueError(f"Cannot parse expiry: {s}")


def parse_tenor_string(s: str) -> int:
    """Convert '1Yr', '10Yr' to integer years."""
    s = s.strip()
    if s.endswith("Yr"):
        return int(s[:-2])
    else:
        raise ValueError(f"Cannot parse tenor: {s}")


def parse_atm_ivs(path: str, sheet: str) -> pd.DataFrame:
    """
    Parse ATM normal-vol surface.

    Returns DataFrame: expiry_years, tenor_years, atm_iv_normal
    where atm_iv_normal is in DECIMAL (bps / 10_000).
    """
    df = pd.read_excel(path, sheet_name=sheet)
    tenor_cols = [c for c in df.columns if c != "Expiry"]

    records = []
    for _, row in df.iterrows():
        expiry_yrs = parse_expiry_string(row["Expiry"])
        for tc in tenor_cols:
            tenor_yrs = parse_tenor_string(tc)
            iv = row[tc]
            if pd.notna(iv):
                records.append({
                    "expiry_years": expiry_yrs,
                    "tenor_years": tenor_yrs,
                    "atm_iv_normal": iv / 10_000.0,  # bps -> decimal
                })
    return pd.DataFrame(records)


def parse_otm_ivs(path: str, sheet: str) -> pd.DataFrame:
    """
    Parse OTM normal-vol surface.

    Returns DataFrame: expiry_years, tenor_years, offset_bps, iv_normal
    where iv_normal is in DECIMAL (bps / 10_000).
    """
    df = pd.read_excel(path, sheet_name=sheet)
    offset_cols = [c for c in df.columns if c != "Term x Tenor"]
    offset_bps_map = {c: int(c.replace("bps", "")) for c in offset_cols}

    records = []
    for _, row in df.iterrows():
        parts = row["Term x Tenor"].split("X")
        expiry_yrs = parse_expiry_string(parts[0].strip())
        tenor_yrs = parse_tenor_string(parts[1].strip())

        for col_name, offset_bps in offset_bps_map.items():
            iv = row[col_name]
            if pd.notna(iv):
                records.append({
                    "expiry_years": expiry_yrs,
                    "tenor_years": tenor_yrs,
                    "offset_bps": offset_bps,
                    "iv_normal": iv / 10_000.0,  # bps -> decimal
                })
    return pd.DataFrame(records)


# =============================================================================
# 3. Swaption analytics
# =============================================================================

def forward_swap_rate(P_disc, I, J, R_proj=None):
    """Forward swap rate.

    Single-curve (R_proj is None):
        S0 = (P(T_I) - P(T_J)) / A0  [telescoping identity]

    Dual-curve (R_proj provided):
        S0 = sum_j tau_j * P_disc(T_j) * R_proj_j / A0
        where A0 = sum_j tau_j * P_disc(T_j).
        The telescoping identity does not hold when the projection
        and discounting curves differ.
    """
    A0 = THETA * np.sum(P_disc[I + 1:J + 1])
    if R_proj is None:
        return (P_disc[I] - P_disc[J]) / A0
    else:
        float_pv = sum(THETA * P_disc[j] * R_proj[j]
                       for j in range(I + 1, J + 1))
        return float_pv / A0

def forward_annuity(P_disc, I, J):
    """Forward annuity using discount factors P_disc (from discounting curve)."""
    return THETA * np.sum(P_disc[I + 1:J + 1])

def frozen_weights(P_disc, I, J, S0):
    """Frozen annuity weights Pi^0_j (eq. 6 of Adachi).

    Uses discount factors P_disc from the discounting curve and
    the precomputed swap rate S0 (which may be dual-curve).
    """
    A0 = forward_annuity(P_disc, I, J)
    Pi = np.zeros(J - I)
    for idx, j in enumerate(range(I + 1, J + 1)):
        bracket = P_disc[J] + S0 * THETA * np.sum(P_disc[j:J + 1])
        Pi[idx] = (THETA * P_disc[j]) / (A0 * P_disc[j - 1]) * bracket
    return Pi

def normalized_weights(Pi, R, I, J, S0):
    """pi_j = Pi^0_j * R_j / S_0.

    R contains the forward rates (from projection curve in dual-curve).
    S0 is the (possibly dual-curve) swap rate.
    """
    pi = np.zeros(J - I)
    for idx, j in enumerate(range(I + 1, J + 1)):
        pi[idx] = Pi[idx] * R[j] / S0
    return pi

def bachelier_price(S0, K, T, sigma_n, A0, is_call=True):
    """Bachelier (normal model) swaption price."""
    if T <= 0 or sigma_n <= 0:
        return max(0.0, A0 * (S0 - K)) if is_call else max(0.0, A0 * (K - S0))
    sqrt_T = np.sqrt(T)
    d = (S0 - K) / (sigma_n * sqrt_T)
    if is_call:
        return A0 * (sigma_n * sqrt_T * (d * norm.cdf(d) + norm.pdf(d)))
    else:
        return A0 * (sigma_n * sqrt_T * (-d * norm.cdf(-d) + norm.pdf(d)))

def bachelier_vega(S0, K, T, sigma_n, A0):
    """Bachelier vega: dPrice/d(sigma_n)."""
    if T <= 0 or sigma_n <= 0:
        return 0.0
    sqrt_T = np.sqrt(T)
    d = (S0 - K) / (sigma_n * sqrt_T)
    return A0 * sqrt_T * norm.pdf(d)

def bachelier_to_black_iv(S0, K, T, sigma_n, A0, is_call=True):
    """Convert Bachelier normal vol to Black lognormal vol."""
    if S0 <= 0 or K <= 0 or T <= 0 or sigma_n <= 0:
        return np.nan
    target = bachelier_price(S0, K, T, sigma_n, A0, is_call)
    if target <= 0:
        return np.nan
    return black_iv(target, S0, K, T, annuity=A0, is_call=is_call)

def black_price(S0, K, T, sigma, A0, is_call=True):
    """Black (1976) swaption price."""
    if T <= 0 or sigma <= 0 or K <= 0 or S0 <= 0:
        return max(0.0, A0 * (S0 - K)) if is_call else max(0.0, A0 * (K - S0))
    sqrt_T = np.sqrt(T)
    d1 = (np.log(S0 / K) + 0.5 * sigma**2 * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    if is_call:
        return A0 * (S0 * norm.cdf(d1) - K * norm.cdf(d2))
    else:
        return A0 * (K * norm.cdf(-d2) - S0 * norm.cdf(-d1))

def black_vega(S0, K, T, sigma, A0):
    """Black model vega."""
    if T <= 0 or sigma <= 0:
        return 0.0
    sqrt_T = np.sqrt(T)
    d1 = (np.log(S0 / K) + 0.5 * sigma**2 * T) / (sigma * sqrt_T)
    return A0 * S0 * sqrt_T * norm.pdf(d1)


# =============================================================================
# 4. Build swaption data for one date
# =============================================================================

def build_swaption_data(P_disc, R_proj, atm_df, otm_df, T_N):
    """
    Build structured swaption dictionary for a single date.

    In the dual-curve setting (EUR):
      - P_disc: discount factors from the OIS (ESTR) curve
                used for annuities, swap rates, frozen weights
      - R_proj: forward rates from the projection (EURIBOR) curve
                used for the FMM dynamics and normalised weights

    In the single-curve setting (USD):
      - P_disc and R_proj are derived from the same SOFR curve

    ALL IVs are stored in TWO formats:
      - ivs_normal:  Bachelier normal vol in decimal  (as quoted)
      - ivs_black:   Black lognormal vol in decimal   (converted)
    """
    swaptions = {}

    for _, row in atm_df.iterrows():
        T_expiry = row["expiry_years"]
        tenor = row["tenor_years"]

        I = int(round(T_expiry))
        J = int(round(T_expiry + tenor))

        if abs(T_expiry - I) > 0.01:
            continue
        if J > T_N or I < 1:
            continue

        key = (T_expiry, tenor)
        if key in swaptions:
            continue

        # Swap rate: dual-curve if R_proj differs from P_disc-implied rates
        S0 = forward_swap_rate(P_disc, I, J, R_proj=R_proj)
        A0 = forward_annuity(P_disc, I, J)

        # Frozen weights from discounting curve, using dual-curve S0
        Pi = frozen_weights(P_disc, I, J, S0)

        # Normalised weights use projection-curve forward rates and dual-curve S0
        pi = normalized_weights(Pi, R_proj, I, J, S0)

        # ATM
        atm_iv_normal = row["atm_iv_normal"]  # Bachelier vol, decimal
        atm_strike = S0

        strikes_list = [atm_strike]
        ivs_normal_list = [atm_iv_normal]
        offset_bps_list = [0]

        # OTM
        otm_subset = otm_df[
            (np.abs(otm_df["expiry_years"] - T_expiry) < 0.01) &
            (np.abs(otm_df["tenor_years"] - tenor) < 0.01)
        ]

        for _, otm_row in otm_subset.iterrows():
            offset_bps = otm_row["offset_bps"]
            K = S0 + offset_bps / 10000.0
            if K <= 0:
                continue
            strikes_list.append(K)
            ivs_normal_list.append(otm_row["iv_normal"])
            offset_bps_list.append(offset_bps)

        # Sort by strike
        sort_idx = np.argsort(strikes_list)
        strikes = np.array(strikes_list)[sort_idx]
        ivs_normal = np.array(ivs_normal_list)[sort_idx]
        offsets = np.array(offset_bps_list)[sort_idx]

        # OTM convention: put if K < S0, call if K >= S0
        is_call = strikes >= S0

        # Convert Bachelier -> Black for every strike
        ivs_black = np.array([
            bachelier_to_black_iv(S0, K, T_expiry, sig_n, A0, ic)
            for K, sig_n, ic in zip(strikes, ivs_normal, is_call)
        ])

        # Filter out strikes where Black IV conversion failed (NaN)
        valid_mask = ~np.isnan(ivs_black)
        if not np.all(valid_mask):
            strikes = strikes[valid_mask]
            ivs_normal = ivs_normal[valid_mask]
            ivs_black = ivs_black[valid_mask]
            offsets = offsets[valid_mask]
            is_call = is_call[valid_mask]

        # Compute Black prices and vegas (using Black IV)
        prices_black = np.array([
            black_price(S0, K, T_expiry, sig_b, A0, ic)
            for K, sig_b, ic in zip(strikes, ivs_black, is_call)
        ])

        vegas_black = np.array([
            black_vega(S0, K, T_expiry, sig_b, A0)
            for K, sig_b in zip(strikes, ivs_black)
        ])

        # Also compute Bachelier prices for reference
        prices_normal = np.array([
            bachelier_price(S0, K, T_expiry, sig_n, A0, ic)
            for K, sig_n, ic in zip(strikes, ivs_normal, is_call)
        ])

        swaptions[key] = {
            "I": I,
            "J": J,
            "expiry_years": T_expiry,
            "tenor_years": tenor,
            "S0": S0,
            "A0": A0,
            "frozen_weights_Pi": Pi.astype(np.float64),
            "normalized_weights_pi": pi.astype(np.float64),
            "pi_sum": float(np.sum(pi)),
            "strikes": strikes.astype(np.float64),
            "ivs_normal": ivs_normal.astype(np.float64),   # Bachelier, decimal
            "ivs_black": ivs_black.astype(np.float64),     # Black, decimal
            "is_call": is_call,
            "offset_bps": offsets,
            "black_prices": prices_black.astype(np.float64),
            "normal_prices": prices_normal.astype(np.float64),
            "vegas": vegas_black.astype(np.float64),
            "n_strikes": len(strikes),
        }

    return swaptions


def build_calibration_subsets(swaptions):
    """Define calibration subsets following Adachi §6.2."""
    subsets = {
        "stage1_1y_tenor": {},
        "stage1_alpha_expiries": {},
        "stage2_multi_tenor": {},
        "joint_all_smiles": {},
        "joint_all_atm": {},
    }
    alpha_expiries = [1.0, 3.0, 5.0, 10.0]

    for key, swn in swaptions.items():
        T_exp, tenor = key

        if swn["n_strikes"] > 1:
            subsets["joint_all_smiles"][key] = swn
        subsets["joint_all_atm"][key] = swn

        if abs(tenor - 1.0) < 0.01:
            subsets["stage1_1y_tenor"][key] = swn
            if T_exp in alpha_expiries:
                subsets["stage1_alpha_expiries"][key] = swn

        if tenor >= 2.0:
            subsets["stage2_multi_tenor"][key] = swn

    return subsets


# =============================================================================
# 5. Package for one date
# =============================================================================

def process_single_date(iv_path, discount_path, atm_sheet, otm_sheet,
                        discount_sheet, date_str, T_N=11, verbose=True,
                        projection_path=None, projection_sheet=None,
                        dual_curve=False, rate_label="rates"):
    """Process one date: parse, bootstrap, build swaption data.

    For single-curve (USD SOFR):
        discount_path provides both P (discount factors) and R (forward rates).

    For dual-curve (EUR):
        discount_path  -> ESTR curve  -> P (discount factors for annuities)
        projection_path -> EURIBOR curve -> R (forward rates for FMM dynamics)
    """

    if verbose:
        print(f"\n{'='*60}")
        print(f"  Processing {date_str}")
        print(f"  IV:        {iv_path} -> {atm_sheet}, {otm_sheet}")
        print(f"  Discount:  {discount_path} -> {discount_sheet}")
        if dual_curve:
            print(f"  Projection: {projection_path} -> {projection_sheet}")
        print(f"  Mode:      {'dual-curve' if dual_curve else 'single-curve'}")
        print(f"{'='*60}")

    # 1a. Discounting curve (SOFR or ESTR)
    disc_df = parse_rate_curve(str(discount_path), discount_sheet)
    disc_curve = bootstrap_discount_curve(disc_df, max_T=max(T_N, 31))
    P_disc = disc_curve["discount_factors"]

    if dual_curve:
        # 1b. Projection curve (EURIBOR) — bootstrap separately
        proj_df = parse_rate_curve(str(projection_path), projection_sheet)
        proj_curve = bootstrap_discount_curve(proj_df, max_T=max(T_N, 31))
        P_proj = proj_curve["discount_factors"]
        # Forward rates from EURIBOR discount curve
        R_proj = compute_forward_term_rates(P_proj)
    else:
        # Single-curve: forward rates from same curve as discounting
        P_proj = P_disc
        R_proj = compute_forward_term_rates(P_disc)

    if verbose:
        print(f"  Discount curve:   P(1Y)={P_disc[1]:.6f}  "
              f"P(5Y)={P_disc[5]:.6f}  P(10Y)={P_disc[10]:.6f}")
        if dual_curve:
            print(f"  Projection curve: P(1Y)={P_proj[1]:.6f}  "
                  f"P(5Y)={P_proj[5]:.6f}  P(10Y)={P_proj[10]:.6f}")
        print(f"  R_1={R_proj[1]*100:.3f}%  R_5={R_proj[5]*100:.3f}%  "
              f"R_10={R_proj[10]*100:.3f}%")

    # 2. IV data
    atm_df = parse_atm_ivs(str(iv_path), atm_sheet)
    otm_df = parse_otm_ivs(str(iv_path), otm_sheet)

    if verbose:
        print(f"  ATM: {len(atm_df)} (expiry,tenor) pairs")
        print(f"  OTM: {len(otm_df)} observations")

    # 3. Build swaption data
    #    P_disc for discounting (annuities, swap rates, frozen weights)
    #    R_proj for forward rates (normalised weights, FMM dynamics)
    swaptions = build_swaption_data(P_disc, R_proj, atm_df, otm_df, T_N)
    subsets = build_calibration_subsets(swaptions)

    if verbose:
        n_smile = sum(1 for s in swaptions.values() if s["n_strikes"] > 1)
        n_atm = sum(1 for s in swaptions.values() if s["n_strikes"] == 1)
        print(f"  Swaptions: {len(swaptions)} total "
              f"({n_smile} with smiles, {n_atm} ATM-only)")
        for name, sub in subsets.items():
            print(f"    {name}: {len(sub)}")

    # 4. Package
    #    IMPORTANT: discount_factors = P_disc (ESTR for EUR, SOFR for USD)
    #               forward_term_rates = R_proj (EURIBOR for EUR, SOFR for USD)
    #    main.py uses P for annuities and R for FMM dynamics independently.
    date_data = {
        "date": date_str,
        "discount_factors": P_disc[:T_N + 1].astype(np.float64),
        "forward_term_rates": R_proj[:T_N + 1].astype(np.float64),
        "theta": THETA,
        "T_N": T_N,
        "dual_curve": dual_curve,
        "swaptions": swaptions,
        "calibration_subsets": {
            name: sorted(sub.keys()) for name, sub in subsets.items()
        },
        "curve": {
            "discount_factors": P_disc.astype(np.float64),
            "forward_term_rates": R_proj.astype(np.float64),
            "discount_swap_rates": disc_curve["swap_rates_annual"].astype(np.float64),
        },
    }

    if dual_curve:
        date_data["curve"]["projection_discount_factors"] = P_proj.astype(np.float64)
        date_data["curve"]["projection_swap_rates"] = \
            proj_curve["swap_rates_annual"].astype(np.float64)

    return date_data


# =============================================================================
# 6. Main pipeline
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Preprocess Bloomberg swaption data for calibration")
    parser.add_argument("--currency", default="usd",
                        choices=list(CURRENCY_CONFIG.keys()),
                        help="Currency to process (default: usd)")
    args = parser.parse_args()

    ccfg = CURRENCY_CONFIG[args.currency]
    data_dir = ccfg["data_dir"]
    disc_suffix = ccfg["discount_file_suffix"]
    disc_sheet_suffix = ccfg["discount_sheet_suffix"]
    proj_suffix = ccfg["projection_file_suffix"]
    proj_sheet_suffix = ccfg["projection_sheet_suffix"]
    dual_curve = ccfg["dual_curve"]
    output_path = ccfg["output_file"]
    label = ccfg["label"]

    print("=" * 70)
    print(f"{label} Swaption Data Preprocessing (multi-date)")
    if dual_curve:
        print(f"  Dual-curve mode: {disc_suffix} (discounting) + "
              f"{proj_suffix} (projection)")
    print("=" * 70)

    all_data = {}
    dates = []

    for file_stem, sheet_prefix, date_str in DATE_MAP:
        iv_path = data_dir / f"{file_stem}IV.xlsx"
        disc_path = data_dir / f"{file_stem}{disc_suffix}.xlsx"
        atm_sheet = f"{sheet_prefix}ATM"
        otm_sheet = f"{sheet_prefix}OTM"
        disc_sheet = f"{sheet_prefix}{disc_sheet_suffix}"

        # Check files exist
        if not iv_path.exists():
            print(f"  WARNING: {iv_path} not found, skipping {date_str}")
            continue
        if not disc_path.exists():
            print(f"  WARNING: {disc_path} not found, skipping {date_str}")
            continue

        # Projection curve (dual-curve only)
        proj_path = None
        proj_sheet = None
        if dual_curve:
            proj_path = data_dir / f"{file_stem}{proj_suffix}.xlsx"
            proj_sheet = f"{sheet_prefix}{proj_sheet_suffix}"
            if not proj_path.exists():
                print(f"  WARNING: {proj_path} not found, skipping {date_str}")
                continue

        date_data = process_single_date(
            iv_path, disc_path,
            atm_sheet, otm_sheet, disc_sheet,
            date_str, T_N=T_N, verbose=True,
            projection_path=proj_path,
            projection_sheet=proj_sheet,
            dual_curve=dual_curve,
            rate_label=label,
        )
        all_data[date_str] = date_data
        dates.append(date_str)

    # Add global metadata
    all_data["dates"] = sorted(dates)
    all_data["metadata"] = {
        "T_N": T_N,
        "theta": THETA,
        "currency": args.currency.upper(),
        "dates": sorted(dates),
        "default_in_sample": DEFAULT_IN_SAMPLE,
        "default_out_sample": DEFAULT_OUT_SAMPLE,
        "iv_convention": "bachelier_normal_bps",
        "iv_storage": "decimal (bps / 10000)",
        "dual_curve": dual_curve,
    }

    # Save
    with open(output_path, "wb") as f:
        pickle.dump(all_data, f)
    print(f"\nSaved to {output_path}")
    print(f"Dates: {sorted(dates)}")
    print(f"Default in-sample:  {DEFAULT_IN_SAMPLE}")
    print(f"Default out-sample: {DEFAULT_OUT_SAMPLE}")

    # Summary
    summary_path = f"data_summary_{args.currency}.txt"
    with open(summary_path, "w") as f:
        f.write(f"{label} Swaption Data Summary (multi-date)\n")
        f.write(f"T_N = {T_N}, tau = {THETA}\n")
        f.write(f"Dual-curve: {dual_curve}\n")
        f.write(f"All IVs are Bachelier (normal) vol, stored in decimal\n\n")

        for date_str in sorted(dates):
            dd = all_data[date_str]
            P = dd["discount_factors"]
            R = dd["forward_term_rates"]
            f.write(f"{'='*60}\n")
            f.write(f"Date: {date_str}\n")
            f.write(f"{'='*60}\n\n")

            f.write("Discount factors (ESTR/SOFR):\n")
            for j in range(T_N + 1):
                f.write(f"  P(T_{j}) = {P[j]:.8f}\n")

            f.write("\nForward term rates (EURIBOR/SOFR):\n")
            for j in range(1, T_N + 1):
                f.write(f"  R_{j} = {R[j]*100:.4f}%\n")

            f.write(f"\nSwaptions ({len(dd['swaptions'])} total):\n")
            for key in sorted(dd["swaptions"].keys()):
                swn = dd["swaptions"][key]
                f.write(f"\n  {swn['expiry_years']:.0f}Y x "
                        f"{swn['tenor_years']:.0f}Y "
                        f"(I={swn['I']}, J={swn['J']}): "
                        f"S0={swn['S0']*100:.4f}%, "
                        f"{swn['n_strikes']} strikes\n")
                atm_mask = swn['offset_bps'] == 0
                if np.any(atm_mask):
                    f.write(f"    ATM normal vol = "
                            f"{swn['ivs_normal'][atm_mask][0]*10000:.2f} bps, "
                            f"Black vol = "
                            f"{swn['ivs_black'][atm_mask][0]*100:.2f}%\n")
            f.write("\n")

    print(f"Summary saved to {summary_path}")
    print("\nDone.")
    return all_data


if __name__ == "__main__":
    all_data = main()