"""
preprocess_usd_swaptions.py
============================
Preprocessing pipeline for USD SOFR swaption data.
Transforms raw Bloomberg data (SOFR swap rates, ATM and OTM swaption IVs)
into calibration-ready tensors for the Mapped Rough SABR FMM.

Reference: Droschl, 2026
           "Fast Swaption Calibration via Automatic Differentiation in a Mapped SABR FMM"

Data date: 9 December 2024

Output: A dictionary (pickled) containing:
    - discount_factors:  P(T_j) for annual grid T_0=0,...,T_N
    - forward_term_rates: R_j^0 for j=1,...,N
    - For each swaption (expiry, tenor):
        - forward_swap_rate S_0
        - annuity A_0
        - strikes (ATM + OTM offsets)
        - lognormal IVs
        - Black model prices
        - frozen weights (Pi_j^0 and pi_j)
    - Metadata and index maps
"""

import numpy as np
import pandas as pd
import pickle
from scipy.stats import norm


# =============================================================================
# 0. Configuration
# =============================================================================

VALUATION_DATE = "2024-12-09"  # Monday, as in Adachi et al.

# Annual tenor grid for the FMM: T_0=0, T_1=1, ..., T_N
# Adachi uses T_N = 11; we extend to 31 to cover the full data range
# but flag which subset is used for calibration
T_N_ADACHI = 11   # for comparison with Adachi et al.
T_N_FULL = 31     # to cover longer tenors in the data

# Year fraction for annual payments (ACT/360 convention, ~365/360)
# Adachi simplifies to theta_j = 1; we do the same for model consistency
THETA = 1.0

# OTM strike offsets in basis points
OTM_OFFSETS_BPS = [-200, -100, -50, -25, 25, 50, 100, 200]

# Paths to data files
SOFR_PATH = "data/SOFRRates.xlsx"
ATM_PATH  = "data/ATMSwaptionIVUSD.xlsx"
OTM_PATH  = "data/OTMSwaptionIVUSD.xlsx"


# =============================================================================
# 1. Parse SOFR swap rates and bootstrap discount curve
# =============================================================================

def parse_sofr_rates(path: str) -> pd.DataFrame:
    """Parse SOFR swap rates from Bloomberg export."""
    df = pd.read_excel(path, sheet_name="Sheet1")

    # Compute mid rate
    df["mid_rate"] = (df["Final Bid Rate"] + df["Final Ask Rate"]) / 2.0

    # Convert Term + Unit to years
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
    return df[["maturity_years", "mid_rate", "Ticker"]].copy()


def bootstrap_discount_curve(sofr_df: pd.DataFrame, max_T: int = 31) -> dict:
    """
    Bootstrap zero-coupon discount factors from SOFR swap rates.

    For a T-year annual swap with rate s_T:
        s_T * sum_{i=1}^{T} theta * P(T_i) = 1 - P(T_T)
    =>  P(T) = (1 - s_T * theta * sum_{i=1}^{T-1} P(T_i)) / (1 + s_T * theta)

    We first interpolate swap rates onto the annual grid using a cubic spline
    on the raw Bloomberg tenors, then bootstrap sequentially.
    """
    # Interpolate swap rates onto annual grid
    raw_mats = sofr_df["maturity_years"].values
    raw_rates = sofr_df["mid_rate"].values / 100.0  # convert from % to decimal

    # Use numpy interpolation (linear is sufficient for integer tenors
    # since most are directly observed; cubic spline requires scipy)
    annual_grid = np.arange(0, max_T + 1, dtype=float)  # T_0=0, T_1=1, ..., T_N
    swap_rates_annual = np.zeros(max_T + 1)
    swap_rates_annual[0] = 0.0
    for j in range(1, max_T + 1):
        swap_rates_annual[j] = float(np.interp(j, raw_mats, raw_rates))

    # Bootstrap discount factors
    P = np.ones(max_T + 1)  # P(T_0) = 1
    theta = THETA

    for j in range(1, max_T + 1):
        s_j = swap_rates_annual[j]
        sum_prev = np.sum(P[1:j]) * theta  # sum_{i=1}^{j-1} theta * P(T_i)
        P[j] = (1.0 - s_j * sum_prev) / (1.0 + s_j * theta)

    # Also store the continuous zero rates for reference
    zero_rates = np.zeros(max_T + 1)
    zero_rates[0] = swap_rates_annual[1]  # short rate ≈ 1Y swap rate
    for j in range(1, max_T + 1):
        zero_rates[j] = -np.log(P[j]) / j

    return {
        "annual_grid": annual_grid,
        "discount_factors": P,
        "swap_rates_annual": swap_rates_annual,
        "zero_rates": zero_rates,
    }


def compute_forward_term_rates(P: np.ndarray) -> np.ndarray:
    """
    Compute forward term rates R_j^0 from discount factors.
    R_j = (1/theta) * (P(T_{j-1})/P(T_j) - 1)
    With theta = 1: R_j = P(T_{j-1})/P(T_j) - 1
    """
    N = len(P) - 1
    R = np.zeros(N + 1)  # R[0] unused, R[j] for j=1,...,N
    for j in range(1, N + 1):
        R[j] = (P[j - 1] / P[j] - 1.0) / THETA
    return R


# =============================================================================
# 2. Parse swaption implied volatility data
# =============================================================================

def parse_expiry_string(s: str) -> float:
    """Convert expiry string (e.g., '1Mo', '18Mo', '1Yr') to years."""
    s = s.strip()
    if s.endswith("Mo"):
        return int(s[:-2]) / 12.0
    elif s.endswith("Yr"):
        return int(s[:-2])
    elif s.endswith("WK") or s.endswith("Wk"):
        return int(s[:-2]) / 52.0
    else:
        raise ValueError(f"Cannot parse expiry: {s}")


def parse_tenor_string(s: str) -> int:
    """Convert tenor column header (e.g., '1Yr', '10Yr') to integer years."""
    s = s.strip()
    if s.endswith("Yr"):
        return int(s[:-2])
    else:
        raise ValueError(f"Cannot parse tenor: {s}")


def parse_atm_ivs(path: str) -> pd.DataFrame:
    """
    Parse ATM swaption IV data.
    Returns a DataFrame with columns: expiry_str, expiry_years, tenor_years, atm_iv
    IVs are in decimal (not percentage).
    """
    df = pd.read_excel(path, sheet_name="Quotes")

    # Tenor columns are all columns except 'Expiry'
    tenor_cols = [c for c in df.columns if c != "Expiry"]

    records = []
    for _, row in df.iterrows():
        expiry_str = row["Expiry"]
        expiry_yrs = parse_expiry_string(expiry_str)
        for tc in tenor_cols:
            tenor_yrs = parse_tenor_string(tc)
            iv = row[tc]
            if pd.notna(iv):
                records.append({
                    "expiry_str": expiry_str,
                    "expiry_years": expiry_yrs,
                    "tenor_years": tenor_yrs,
                    "atm_iv": iv / 100.0,  # convert % to decimal
                })

    return pd.DataFrame(records)


def parse_otm_ivs(path: str) -> pd.DataFrame:
    """
    Parse OTM swaption IV data.
    Returns DataFrame with: expiry_str, expiry_years, tenor_years, offset_bps, iv
    """
    df = pd.read_excel(path, sheet_name="Quotes")

    offset_cols = [c for c in df.columns if c != "Term x Tenor"]
    offset_bps_map = {}
    for c in offset_cols:
        # Parse column name like '-200bps', '25bps'
        val = int(c.replace("bps", ""))
        offset_bps_map[c] = val

    records = []
    for _, row in df.iterrows():
        label = row["Term x Tenor"]  # e.g., '1Mo X 1Yr'
        parts = label.split("X")
        expiry_str = parts[0].strip()
        tenor_str = parts[1].strip()
        expiry_yrs = parse_expiry_string(expiry_str)
        tenor_yrs = parse_tenor_string(tenor_str)

        for col_name, offset_bps in offset_bps_map.items():
            iv = row[col_name]
            if pd.notna(iv):
                records.append({
                    "expiry_str": expiry_str,
                    "expiry_years": expiry_yrs,
                    "tenor_years": tenor_yrs,
                    "offset_bps": offset_bps,
                    "iv": iv / 100.0,  # convert % to decimal
                })

    return pd.DataFrame(records)


# =============================================================================
# 3. Swaption analytics (Black model, frozen weights)
# =============================================================================

def forward_swap_rate(P: np.ndarray, I: int, J: int, theta: float = THETA) -> float:
    """
    Forward swap rate S_0(T_I, T_J) = (P(T_I) - P(T_J)) / A_0
    where A_0 = sum_{j=I+1}^{J} theta_j P(T_j)
    """
    annuity = theta * np.sum(P[I + 1:J + 1])
    return (P[I] - P[J]) / annuity


def forward_annuity(P: np.ndarray, I: int, J: int, theta: float = THETA) -> float:
    """Forward annuity A_0(T_I, T_J) = sum_{j=I+1}^{J} theta_j P(T_j)."""
    return theta * np.sum(P[I + 1:J + 1])


def black_price(S0: float, K: float, T: float, sigma: float,
                annuity: float, is_call: bool = True) -> float:
    """
    Black (1976) swaption price.
    C = A_0 * [S_0 N(d1) - K N(d2)]  (payer)
    P = A_0 * [K N(-d2) - S_0 N(-d1)]  (receiver)
    """
    if T <= 0 or sigma <= 0 or K <= 0 or S0 <= 0:
        return max(0.0, annuity * (S0 - K)) if is_call else max(0.0, annuity * (K - S0))

    sqrt_T = np.sqrt(T)
    d1 = (np.log(S0 / K) + 0.5 * sigma ** 2 * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T

    if is_call:
        return annuity * (S0 * norm.cdf(d1) - K * norm.cdf(d2))
    else:
        return annuity * (K * norm.cdf(-d2) - S0 * norm.cdf(-d1))


def black_vega(S0: float, K: float, T: float, sigma: float,
               annuity: float) -> float:
    """Black model vega (dC/dsigma), same for call and put."""
    if T <= 0 or sigma <= 0:
        return 0.0
    sqrt_T = np.sqrt(T)
    d1 = (np.log(S0 / K) + 0.5 * sigma ** 2 * T) / (sigma * sqrt_T)
    return annuity * S0 * sqrt_T * norm.pdf(d1)


def frozen_weights(P: np.ndarray, I: int, J: int,
                   theta: float = THETA) -> np.ndarray:
    """
    Compute frozen annuity weights Pi^0_j for j = I+1, ..., J.
    From Adachi et al. eq. (6) at t=0:

    Pi^0_j = [theta_j P(T_j)] / [A_0 P(T_{j-1})]
             * [P(T_J) + S_0 sum_{k=j}^{J} theta_k P(T_k)]

    Returns array of length (J - I) with Pi^0_{I+1}, ..., Pi^0_J.
    """
    A0 = forward_annuity(P, I, J, theta)
    S0 = forward_swap_rate(P, I, J, theta)

    Pi = np.zeros(J - I)
    for idx, j in enumerate(range(I + 1, J + 1)):
        bracket = P[J] + S0 * theta * np.sum(P[j:J + 1])
        Pi[idx] = (theta * P[j]) / (A0 * P[j - 1]) * bracket

    return Pi


def normalized_weights(Pi: np.ndarray, R: np.ndarray,
                       I: int, J: int, S0: float) -> np.ndarray:
    """
    Compute normalized weights pi_j = Pi^0_j * eta_j(R^j_0) / S_0.
    With eta_j(r) = r (lognormal case): pi_j = Pi^0_j * R^j_0 / S_0.
    Returns array of length (J - I).
    """
    pi = np.zeros(J - I)
    for idx, j in enumerate(range(I + 1, J + 1)):
        pi[idx] = Pi[idx] * R[j] / S0
    return pi


# =============================================================================
# 4. Build the calibration data structure
# =============================================================================

def build_swaption_data(P: np.ndarray, R: np.ndarray,
                        atm_df: pd.DataFrame, otm_df: pd.DataFrame,
                        T_N: int) -> dict:
    """
    Build a structured dictionary of swaption data for calibration.

    For each (expiry, tenor) pair:
        - Computes S_0, A_0, frozen weights
        - Assembles strikes (ATM + OTM offsets)
        - Looks up IVs, computes Black prices
        - Stores call/put flag (OTM convention)

    Returns dict keyed by (expiry_years, tenor_years) tuples.
    """
    swaptions = {}

    # Get all unique (expiry, tenor) pairs from ATM data
    # Filter to those where expiry + tenor <= T_N and both map to integer indices
    for _, row in atm_df.iterrows():
        T_expiry = row["expiry_years"]
        tenor = row["tenor_years"]
        T_end = T_expiry + tenor

        # Expiry must be an integer year for the annual FMM grid
        # (sub-annual expiries are valid -- the swaption expires at T_expiry
        #  into a swap on the annual grid starting at the next integer)
        # I = index such that T_I = ceil(T_expiry) in annual grid
        # But for simplicity, require integer expiry for now (matching Adachi)
        I = int(round(T_expiry))
        J = int(round(T_end))

        # Skip if expiry is not close to an integer year
        if abs(T_expiry - I) > 0.01:
            continue

        # Skip if beyond our grid
        if J > T_N or I < 1:
            continue

        # Skip if we already have this pair (shouldn't happen with clean data)
        key = (T_expiry, tenor)
        if key in swaptions:
            continue

        # Compute forward swap analytics
        S0 = forward_swap_rate(P, I, J)
        A0 = forward_annuity(P, I, J)
        Pi = frozen_weights(P, I, J)
        pi = normalized_weights(Pi, R, I, J, S0)

        # Sanity check: sum of pi should be ~1
        pi_sum = np.sum(pi)

        # ATM strike and IV
        atm_iv = row["atm_iv"]
        atm_strike = S0

        # Build strike/IV vectors: start with ATM
        strikes_list = [atm_strike]
        ivs_list = [atm_iv]
        offset_bps_list = [0]

        # Add OTM data if available
        otm_subset = otm_df[
            (np.abs(otm_df["expiry_years"] - T_expiry) < 0.01) &
            (np.abs(otm_df["tenor_years"] - tenor) < 0.01)
        ]

        for _, otm_row in otm_subset.iterrows():
            offset_bps = otm_row["offset_bps"]
            K = S0 + offset_bps / 10000.0
            if K <= 0:
                continue  # skip negative strikes
            iv = otm_row["iv"]
            strikes_list.append(K)
            ivs_list.append(iv)
            offset_bps_list.append(offset_bps)

        # Sort by strike
        sort_idx = np.argsort(strikes_list)
        strikes = np.array(strikes_list)[sort_idx]
        ivs = np.array(ivs_list)[sort_idx]
        offsets = np.array(offset_bps_list)[sort_idx]

        # Determine call/put flag (OTM convention: put if K < S0, call if K >= S0)
        is_call = strikes >= S0

        # Compute Black prices
        prices = np.array([
            black_price(S0, K, T_expiry, sig, A0, ic)
            for K, sig, ic in zip(strikes, ivs, is_call)
        ])

        # Compute vegas (for weighting in loss function)
        vegas = np.array([
            black_vega(S0, K, T_expiry, sig, A0)
            for K, sig in zip(strikes, ivs)
        ])

        swaptions[key] = {
            # Indices
            "I": I,
            "J": J,
            "expiry_years": T_expiry,
            "tenor_years": tenor,
            # Swap analytics
            "S0": S0,
            "A0": A0,
            "frozen_weights_Pi": Pi,    # shape (J-I,)
            "normalized_weights_pi": pi,  # shape (J-I,)
            "pi_sum": pi_sum,
            # Strike/IV data
            "strikes": strikes,
            "ivs": ivs,
            "offset_bps": offsets,
            "is_call": is_call,
            # Prices and vegas
            "black_prices": prices,
            "vegas": vegas,
            # Number of strikes
            "n_strikes": len(strikes),
        }

    return swaptions


def build_calibration_subsets(swaptions: dict) -> dict:
    """
    Define calibration subsets following Adachi et al. Section 6.2.

    Stage 1 (first step): 1Y tenor swaptions only
        → calibrates kappa, {alpha_j}, {rho_{0j}}
        Adachi uses expiries at 1Y, 3Y, 5Y, 10Y for alpha estimation

    Stage 2 (second step): multi-tenor ATM swaptions
        → calibrates correlation matrix [rho_{ij}]

    Joint: all swaptions with smile data (for AMCC joint calibration)
    """
    subsets = {
        "stage1_1y_tenor": {},      # 1Y tenor, all expiries (smiles)
        "stage1_alpha_expiries": {}, # 1Y tenor, Adachi's alpha expiries
        "stage2_multi_tenor": {},    # tenor >= 2Y, ATM only
        "joint_all_smiles": {},      # everything with smile data
        "joint_all_atm": {},         # everything, ATM only
    }

    alpha_expiries = [1.0, 3.0, 5.0, 10.0]  # Adachi's chosen expiries

    for key, swn in swaptions.items():
        T_exp, tenor = key

        # Joint: all swaptions with OTM data
        if swn["n_strikes"] > 1:
            subsets["joint_all_smiles"][key] = swn

        # Joint ATM: everything (even single-strike)
        subsets["joint_all_atm"][key] = swn

        # Stage 1: 1Y tenor
        if abs(tenor - 1.0) < 0.01:
            subsets["stage1_1y_tenor"][key] = swn
            if T_exp in alpha_expiries:
                subsets["stage1_alpha_expiries"][key] = swn

        # Stage 2: multi-tenor ATM
        if tenor >= 2.0:
            subsets["stage2_multi_tenor"][key] = swn

    return subsets


# =============================================================================
# 5. Convert to PyTorch tensors
# =============================================================================

def package_data(swaptions: dict, P: np.ndarray, R: np.ndarray) -> dict:
    """
    Package all data into a clean dictionary for serialization.
    All arrays are numpy float64. Convert to torch tensors at load time via:
        tensor = torch.from_numpy(array)  or  torch.tensor(array, dtype=torch.float64)
    """
    packaged = {
        "discount_factors": P.astype(np.float64),
        "forward_term_rates": R.astype(np.float64),
        "theta": THETA,
        "swaptions": {},
    }

    for key, swn in swaptions.items():
        packaged["swaptions"][key] = {
            "I": swn["I"],
            "J": swn["J"],
            "expiry_years": swn["expiry_years"],
            "tenor_years": swn["tenor_years"],
            "S0": np.float64(swn["S0"]),
            "A0": np.float64(swn["A0"]),
            "frozen_weights_Pi": swn["frozen_weights_Pi"].astype(np.float64),
            "normalized_weights_pi": swn["normalized_weights_pi"].astype(np.float64),
            "strikes": swn["strikes"].astype(np.float64),
            "ivs": swn["ivs"].astype(np.float64),
            "is_call": swn["is_call"],
            "black_prices": swn["black_prices"].astype(np.float64),
            "vegas": swn["vegas"].astype(np.float64),
            "n_strikes": swn["n_strikes"],
        }

    return packaged


# =============================================================================
# 6. Main pipeline
# =============================================================================

def main():
    print("=" * 70)
    print("USD SOFR Swaption Data Preprocessing")
    print(f"Valuation date: {VALUATION_DATE}")
    print("=" * 70)

    # --- Step 1: SOFR curve ---
    print("\n[1/5] Parsing SOFR swap rates...")
    sofr_df = parse_sofr_rates(SOFR_PATH)
    print(f"  Loaded {len(sofr_df)} swap rate quotes")
    print(f"  Maturities: {sofr_df['maturity_years'].min():.4f}Y "
          f"to {sofr_df['maturity_years'].max():.1f}Y")
    print(f"  1Y mid rate: {sofr_df[sofr_df['maturity_years']==1.0]['mid_rate'].values[0]:.4f}%")

    # --- Step 2: Bootstrap ---
    print("\n[2/5] Bootstrapping discount curve...")
    curve = bootstrap_discount_curve(sofr_df, max_T=T_N_FULL)
    P = curve["discount_factors"]
    R = compute_forward_term_rates(P)

    print(f"  Annual grid: T_0=0 to T_{T_N_FULL}={T_N_FULL}")
    print(f"  P(1Y) = {P[1]:.6f},  P(5Y) = {P[5]:.6f},  P(10Y) = {P[10]:.6f}")
    print(f"  R_1 = {R[1]*100:.3f}%,  R_5 = {R[5]*100:.3f}%,  R_10 = {R[10]*100:.3f}%")

    # Sanity: recompute 5Y swap rate from discount factors
    S5 = forward_swap_rate(P, 0, 5)
    print(f"  Recomputed 5Y par swap rate: {S5*100:.3f}% "
          f"(market: {curve['swap_rates_annual'][5]*100:.3f}%)")

    # --- Step 3: Parse swaption IVs ---
    print("\n[3/5] Parsing swaption implied volatilities...")
    atm_df = parse_atm_ivs(ATM_PATH)
    otm_df = parse_otm_ivs(OTM_PATH)
    print(f"  ATM: {len(atm_df)} (expiry, tenor) pairs")
    print(f"  OTM: {len(otm_df)} (expiry, tenor, offset) observations")
    print(f"  ATM expiries: {sorted(atm_df['expiry_years'].unique())}")
    print(f"  ATM tenors: {sorted(atm_df['tenor_years'].unique())}")

    # --- Step 4: Build swaption structures ---
    print(f"\n[4/5] Building swaption data (T_N = {T_N_ADACHI} for Adachi comparison)...")
    swaptions = build_swaption_data(P, R, atm_df, otm_df, T_N=T_N_ADACHI)
    print(f"  {len(swaptions)} swaptions on the annual grid")

    # Summary
    for key in sorted(swaptions.keys()):
        swn = swaptions[key]
        print(f"    {swn['expiry_years']:.0f}Y x {swn['tenor_years']:.0f}Y: "
              f"S0={swn['S0']*100:.3f}%, "
              f"ATM IV={swn['ivs'][swn['offset_bps']==0][0]*100:.2f}%, "
              f"{swn['n_strikes']} strikes, "
              f"pi_sum={swn['pi_sum']:.4f}")

    # --- Build calibration subsets ---
    subsets = build_calibration_subsets(swaptions)
    print(f"\n  Calibration subsets:")
    for name, subset in subsets.items():
        print(f"    {name}: {len(subset)} swaptions")

    # --- Step 5: Package and save ---
    print(f"\n[5/5] Packaging data and saving...")
    packaged_data = package_data(swaptions, P, R)

    # Add metadata
    packaged_data["metadata"] = {
        "valuation_date": VALUATION_DATE,
        "T_N": T_N_ADACHI,
        "theta": THETA,
        "n_swaptions": len(swaptions),
        "swaption_keys": sorted(swaptions.keys()),
    }

    # Add calibration subset keys
    packaged_data["calibration_subsets"] = {
        name: sorted(subset.keys()) for name, subset in subsets.items()
    }

    # Add raw curve data for reference
    packaged_data["curve"] = {
        "discount_factors": P.astype(np.float64),
        "forward_term_rates": R.astype(np.float64),
        "swap_rates_annual": curve["swap_rates_annual"].astype(np.float64),
        "zero_rates": curve["zero_rates"].astype(np.float64),
    }

    # Save
    output_path = "/home/claude/usd_swaption_data.pkl"
    with open(output_path, "wb") as f:
        pickle.dump(packaged_data, f)
    print(f"  Saved to {output_path}")

    # --- Also save a human-readable summary ---
    summary_path = "/home/claude/data_summary.txt"
    with open(summary_path, "w") as f:
        f.write(f"USD SOFR Swaption Data Summary\n")
        f.write(f"Valuation date: {VALUATION_DATE}\n")
        f.write(f"T_N = {T_N_ADACHI} (annual grid)\n\n")

        f.write("Discount factors:\n")
        for j in range(T_N_ADACHI + 1):
            f.write(f"  P(T_{j}) = {P[j]:.8f}\n")

        f.write("\nForward term rates:\n")
        for j in range(1, T_N_ADACHI + 1):
            f.write(f"  R_{j} = {R[j]*100:.4f}%\n")

        f.write(f"\nSwaptions ({len(swaptions)} total):\n")
        for key in sorted(swaptions.keys()):
            swn = swaptions[key]
            f.write(f"\n  --- {swn['expiry_years']:.0f}Y x {swn['tenor_years']:.0f}Y ---\n")
            f.write(f"  I={swn['I']}, J={swn['J']}\n")
            f.write(f"  S0 = {swn['S0']*100:.4f}%\n")
            f.write(f"  A0 = {swn['A0']:.6f}\n")
            f.write(f"  Pi_sum = {swn['pi_sum']:.6f}\n")
            f.write(f"  Strikes: {swn['strikes']*100}\n")
            f.write(f"  IVs (%): {swn['ivs']*100}\n")
            f.write(f"  Prices: {swn['black_prices']}\n")

    print(f"  Summary saved to {summary_path}")

    # --- Quick validation ---
    print("\n--- Validation ---")
    # Check: for 1Y tenor swaptions, S0 should match the forward rate
    for key in sorted(swaptions.keys()):
        swn = swaptions[key]
        if swn["tenor_years"] == 1:
            I, J = swn["I"], swn["J"]
            # S0(T_I, T_{I+1}) = R_{I+1} for 1Y tenor
            print(f"  {swn['expiry_years']:.0f}Yx1Y: "
                  f"S0={swn['S0']*100:.4f}%, R_{J}={R[J]*100:.4f}%, "
                  f"match={np.isclose(swn['S0'], R[J], atol=1e-10)}")

    print("\nDone.")
    return packaged_data, swaptions, subsets


if __name__ == "__main__":
    packaged_data, swaptions, subsets = main()
