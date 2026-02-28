"""
preprocess_usd_swaptions.py
============================
Preprocessing pipeline for USD SOFR swaption data (multi-date).

Transforms Bloomberg London Closing data — SOFR swap rates, ATM and OTM
swaption IVs — into calibration-ready tensors for the Mapped Rough SABR FMM.

KEY CONVENTION
--------------
ALL implied volatilities (both ATM and OTM) are **Bachelier / normal vol
in basis points** as quoted by Bloomberg.  The preprocessing stores them
in decimal (bps / 10_000).  Conversion to Black (lognormal) IV is
performed at load time by `load_market_data()`.

Data layout
-----------
Input files (placed in `data/` subfolder):

    Dez2024IV.xlsx      sheets: 0912ATM, 0912OTM, 1012ATM, 1012OTM, 1112ATM, 1112OTM
    Dez2024SOFR.xlsx    sheets: 0912SOFR, 1012SOFR, 1112SOFR
    Dez2025IV.xlsx      sheets: 0812ATM, 0812OTM, 0912ATM, 0912OTM, 1012ATM, 1012OTM
    Dez2025SOFR.xlsx    sheets: 0812SOFR, 0912SOFR, 1012SOFR

Each sheet prefix (e.g. "0912") encodes the date as DDMM.

Output: usd_swaption_data.pkl — a dict keyed by date string, e.g.:
    {
        "2024-12-09": { discount_factors, forward_term_rates, swaptions, ... },
        "2024-12-10": { ... },
        ...
        "dates": ["2024-12-09", ...],
        "metadata": { ... },
    }

Usage:
    python preprocess_usd_swaptions.py
"""

import numpy as np
import pandas as pd
import pickle
from scipy.stats import norm
from pathlib import Path


# =============================================================================
# 0. Configuration
# =============================================================================

# Annual tenor grid for the FMM (Adachi: T_N = 11)
T_N = 11

# Year fraction for annual payments (theta = 1 for annual)
THETA = 1.0

# OTM strike offsets in basis points
OTM_OFFSETS_BPS = [-200, -100, -50, -25, 25, 50, 100, 200]

# Data directory (relative to this script)
DATA_DIR = Path("data")

# Date mapping: (filename_stem, sheet_prefix) -> ISO date string
# Sheet prefix convention: DDMM  (day-month)
DATE_MAP = [
    # December 2024 cluster
    ("Dez2024", "0912", "2024-12-09"),
    ("Dez2024", "1012", "2024-12-10"),
    ("Dez2024", "1112", "2024-12-11"),
    # December 2025 cluster
    ("Dez2025", "0812", "2025-12-08"),
    ("Dez2025", "0912", "2025-12-09"),
    ("Dez2025", "1012", "2025-12-10"),
]

# Default calibration / evaluation dates
DEFAULT_IN_SAMPLE  = "2024-12-09"
DEFAULT_OUT_SAMPLE = "2025-12-10"


# =============================================================================
# 1. SOFR curve: parsing and bootstrapping
# =============================================================================

def parse_sofr_rates(path: str, sheet: str) -> pd.DataFrame:
    """Parse SOFR swap rates from a Bloomberg export sheet."""
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


def bootstrap_discount_curve(sofr_df: pd.DataFrame, max_T: int = 31) -> dict:
    """
    Bootstrap zero-coupon discount factors from SOFR swap rates.

    s_T * sum_{i=1}^{T} theta * P(T_i) = 1 - P(T)
    => P(T) = (1 - s_T * theta * sum_{i=1}^{T-1} P(T_i)) / (1 + s_T * theta)
    """
    raw_mats = sofr_df["maturity_years"].values
    raw_rates = sofr_df["mid_rate"].values / 100.0  # % -> decimal

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
    """R_j = (1/theta) * (P(T_{j-1})/P(T_j) - 1)."""
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

def forward_swap_rate(P, I, J):
    A0 = THETA * np.sum(P[I + 1:J + 1])
    return (P[I] - P[J]) / A0

def forward_annuity(P, I, J):
    return THETA * np.sum(P[I + 1:J + 1])

def frozen_weights(P, I, J):
    """Frozen annuity weights Pi^0_j (eq. 6 of Adachi)."""
    A0 = forward_annuity(P, I, J)
    S0 = forward_swap_rate(P, I, J)
    Pi = np.zeros(J - I)
    for idx, j in enumerate(range(I + 1, J + 1)):
        bracket = P[J] + S0 * THETA * np.sum(P[j:J + 1])
        Pi[idx] = (THETA * P[j]) / (A0 * P[j - 1]) * bracket
    return Pi

def normalized_weights(Pi, R, I, J, S0):
    """pi_j = Pi^0_j * R_j / S_0."""
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
    """Convert Bachelier normal vol to Black lognormal vol via root finding."""
    from scipy.optimize import brentq

    target = bachelier_price(S0, K, T, sigma_n, A0, is_call)
    if target <= 0 or S0 <= 0 or K <= 0 or T <= 0:
        return np.nan

    def black_price_local(sigma_b):
        sqrt_T = np.sqrt(T)
        d1 = (np.log(S0 / K) + 0.5 * sigma_b**2 * T) / (sigma_b * sqrt_T)
        d2 = d1 - sigma_b * sqrt_T
        if is_call:
            return A0 * (S0 * norm.cdf(d1) - K * norm.cdf(d2))
        else:
            return A0 * (K * norm.cdf(-d2) - S0 * norm.cdf(-d1))

    try:
        iv = brentq(lambda s: black_price_local(s) - target, 1e-6, 5.0, xtol=1e-12)
        return iv
    except (ValueError, RuntimeError):
        # Fallback: first-order approximation  sigma_B ≈ sigma_N / S0
        return sigma_n / S0 if S0 > 0 else np.nan

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

def build_swaption_data(P, R, atm_df, otm_df, T_N):
    """
    Build structured swaption dictionary for a single date.

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

        S0 = forward_swap_rate(P, I, J)
        A0 = forward_annuity(P, I, J)
        Pi = frozen_weights(P, I, J)
        pi = normalized_weights(Pi, R, I, J, S0)

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

def process_single_date(iv_path, sofr_path, atm_sheet, otm_sheet, sofr_sheet,
                        date_str, T_N=11, verbose=True):
    """Process one date: parse, bootstrap, build swaption data."""

    if verbose:
        print(f"\n{'='*60}")
        print(f"  Processing {date_str}")
        print(f"  IV:   {iv_path} → {atm_sheet}, {otm_sheet}")
        print(f"  SOFR: {sofr_path} → {sofr_sheet}")
        print(f"{'='*60}")

    # 1. SOFR curve
    sofr_df = parse_sofr_rates(str(sofr_path), sofr_sheet)
    curve = bootstrap_discount_curve(sofr_df, max_T=max(T_N, 31))
    P = curve["discount_factors"]
    R = compute_forward_term_rates(P)

    if verbose:
        print(f"  P(1Y)={P[1]:.6f}  P(5Y)={P[5]:.6f}  P(10Y)={P[10]:.6f}")
        print(f"  R_1={R[1]*100:.3f}%  R_5={R[5]*100:.3f}%  R_10={R[10]*100:.3f}%")

    # 2. IV data
    atm_df = parse_atm_ivs(str(iv_path), atm_sheet)
    otm_df = parse_otm_ivs(str(iv_path), otm_sheet)

    if verbose:
        print(f"  ATM: {len(atm_df)} (expiry,tenor) pairs")
        print(f"  OTM: {len(otm_df)} observations")

    # 3. Build swaption data
    swaptions = build_swaption_data(P, R, atm_df, otm_df, T_N)
    subsets = build_calibration_subsets(swaptions)

    if verbose:
        n_smile = sum(1 for s in swaptions.values() if s["n_strikes"] > 1)
        n_atm = sum(1 for s in swaptions.values() if s["n_strikes"] == 1)
        print(f"  Swaptions: {len(swaptions)} total ({n_smile} with smiles, {n_atm} ATM-only)")
        for name, sub in subsets.items():
            print(f"    {name}: {len(sub)}")

    # 4. Package
    date_data = {
        "date": date_str,
        "discount_factors": P[:T_N + 1].astype(np.float64),
        "forward_term_rates": R[:T_N + 1].astype(np.float64),
        "theta": THETA,
        "T_N": T_N,
        "swaptions": swaptions,
        "calibration_subsets": {
            name: sorted(sub.keys()) for name, sub in subsets.items()
        },
        "curve": {
            "discount_factors": P.astype(np.float64),
            "forward_term_rates": R.astype(np.float64),
            "swap_rates_annual": curve["swap_rates_annual"].astype(np.float64),
        },
    }

    return date_data


# =============================================================================
# 6. Main pipeline
# =============================================================================

def main():
    print("=" * 70)
    print("USD SOFR Swaption Data Preprocessing (multi-date)")
    print("=" * 70)

    all_data = {}
    dates = []

    for file_stem, sheet_prefix, date_str in DATE_MAP:
        iv_path = DATA_DIR / f"{file_stem}IV.xlsx"
        sofr_path = DATA_DIR / f"{file_stem}SOFR.xlsx"
        atm_sheet = f"{sheet_prefix}ATM"
        otm_sheet = f"{sheet_prefix}OTM"
        sofr_sheet = f"{sheet_prefix}SOFR"

        # Check files exist
        if not iv_path.exists():
            print(f"  WARNING: {iv_path} not found, skipping {date_str}")
            continue
        if not sofr_path.exists():
            print(f"  WARNING: {sofr_path} not found, skipping {date_str}")
            continue

        date_data = process_single_date(
            iv_path, sofr_path,
            atm_sheet, otm_sheet, sofr_sheet,
            date_str, T_N=T_N, verbose=True,
        )
        all_data[date_str] = date_data
        dates.append(date_str)

    # Add global metadata
    all_data["dates"] = sorted(dates)
    all_data["metadata"] = {
        "T_N": T_N,
        "theta": THETA,
        "dates": sorted(dates),
        "default_in_sample": DEFAULT_IN_SAMPLE,
        "default_out_sample": DEFAULT_OUT_SAMPLE,
        "iv_convention": "bachelier_normal_bps",
        "iv_storage": "decimal (bps / 10000)",
    }

    # Save
    output_path = "usd_swaption_data.pkl"
    with open(output_path, "wb") as f:
        pickle.dump(all_data, f)
    print(f"\nSaved to {output_path}")
    print(f"Dates: {sorted(dates)}")
    print(f"Default in-sample:  {DEFAULT_IN_SAMPLE}")
    print(f"Default out-sample: {DEFAULT_OUT_SAMPLE}")

    # Summary
    summary_path = "data_summary.txt"
    with open(summary_path, "w") as f:
        f.write("USD SOFR Swaption Data Summary (multi-date)\n")
        f.write(f"T_N = {T_N}, theta = {THETA}\n")
        f.write(f"All IVs are Bachelier (normal) vol, stored in decimal\n\n")

        for date_str in sorted(dates):
            dd = all_data[date_str]
            P = dd["discount_factors"]
            R = dd["forward_term_rates"]
            f.write(f"{'='*60}\n")
            f.write(f"Date: {date_str}\n")
            f.write(f"{'='*60}\n\n")

            f.write("Discount factors:\n")
            for j in range(T_N + 1):
                f.write(f"  P(T_{j}) = {P[j]:.8f}\n")

            f.write("\nForward term rates:\n")
            for j in range(1, T_N + 1):
                f.write(f"  R_{j} = {R[j]*100:.4f}%\n")

            f.write(f"\nSwaptions ({len(dd['swaptions'])} total):\n")
            for key in sorted(dd["swaptions"].keys()):
                swn = dd["swaptions"][key]
                f.write(f"\n  {swn['expiry_years']:.0f}Y x {swn['tenor_years']:.0f}Y "
                        f"(I={swn['I']}, J={swn['J']}): "
                        f"S0={swn['S0']*100:.4f}%, "
                        f"{swn['n_strikes']} strikes\n")
                f.write(f"    ATM normal vol = "
                        f"{swn['ivs_normal'][swn['offset_bps']==0][0]*10000:.2f} bps, "
                        f"Black vol = "
                        f"{swn['ivs_black'][swn['offset_bps']==0][0]*100:.2f}%\n")
            f.write("\n")

    print(f"Summary saved to {summary_path}")
    print("\nDone.")
    return all_data


if __name__ == "__main__":
    all_data = main()