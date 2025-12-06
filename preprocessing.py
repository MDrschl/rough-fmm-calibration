# preprocessing.py

import numpy as np
import pandas as pd
from dataclasses import dataclass


# ---------- simple container for a smile ----------

@dataclass
class SwaptionSmile:
    expiry: float              # option maturity in years
    tenor: float               # underlying swap tenor in years
    moneyness_bps: np.ndarray  # strike shift in bps relative to ATM (-200, -100, ...)
    vol: np.ndarray            # Black vols at those strikes (decimal, e.g. 0.25)
    atm_vol: float             # ATM Black vol (decimal)


# ---------- main preprocessor ----------

class USDRFRPreprocessor:
    """
    Preprocess USD SOFR curve and swaption IV data (ATM + OTM).

    After construction you get:

    - self.curve: DataFrame with columns ['T','zero_rate','df'].
      * 'T'        : maturity in years
      * 'zero_rate': continuously comp. zero rate in decimals
      * 'df'       : discount factor P(0,T)

    - self.atm_vols: dict[(expiry_years, tenor_years)] -> ATM vol (decimal).
    - self.smiles:    dict[(expiry_years, tenor_years)] -> SwaptionSmile.

    Helper methods:
      - discount_factor(T)
      - curve_on_grid(year_grid)
      - list_available_pairs()
      - get_smile(expiry_years, tenor_years)
    """

    def __init__(self, sofr_path: str, atm_path: str, otm_path: str,
                 use_bootstrap: bool = True):
        self.raw_sofr = pd.read_excel(sofr_path)
        self.raw_atm = pd.read_excel(atm_path)
        self.raw_otm = pd.read_excel(otm_path)

        self.use_bootstrap = use_bootstrap

        self.curve = None      # discount curve
        self.atm_vols = {}     # (expiry, tenor) -> ATM vol (decimal)
        self.smiles = {}       # (expiry, tenor) -> SwaptionSmile

        self._build_zero_curve()
        self._build_atm_surface()
        self._build_smile_surface()

    # -------- basic tenor parsing --------

    @staticmethod
    def _tenor_to_years(term, unit: str) -> float:
        """
        Convert number + unit (from SOFR file) to year fraction.
        """
        unit = unit.strip().upper()
        if unit in ["YR", "Y", "YEAR", "YEARS"]:
            return float(term)
        if unit in ["MO", "MON", "MONTH", "MONTHS"]:
            return float(term) / 12.0
        if unit in ["WK", "WEEK", "WEEKS"]:
            return float(term) / 52.0
        if unit in ["DY", "DAY", "DAYS"]:
            return float(term) / 365.0
        raise ValueError(f"Unknown unit {unit}")

    @staticmethod
    def _tenor_str_to_years(s: str) -> float:
        """
        Convert strings like '1Mo', '3Mo', '1Yr', '10Yr' (from ATM/OTM files)
        to year fractions.
        """
        s = s.strip()
        num, unit = "", ""
        for ch in s:
            if ch.isdigit():
                num += ch
            else:
                unit += ch
        if not num:
            raise ValueError(f"Cannot parse tenor string {s}")

        unit = unit.upper()
        unit = unit.replace("MTH", "MO").replace("MN", "MO")
        unit = unit.replace("YR", "YR").replace("Y", "YR")

        if "MO" in unit:
            return float(num) / 12.0
        else:  # default: years
            return float(num)

    # -------- 0.a build discount curve --------

    def _build_zero_curve(self):
        df = self.raw_sofr.copy()

        # maturity in years
        df["T"] = [
            self._tenor_to_years(term, unit)
            for term, unit in zip(df["Term"], df["Unit"])
        ]

        # mid par OIS rate in *decimal*: 4.52 -> 0.0452
        df["par_rate"] = 0.5 * (df["Final Bid Rate"] + df["Final Ask Rate"]) / 100.0

        df = df.sort_values("T").reset_index(drop=True)

        T = df["T"].to_numpy()
        S = df["par_rate"].to_numpy()

        if self.use_bootstrap:
            # OIS bootstrap of discount factors from par swap rates
            P = np.empty_like(T, dtype=float)

            # first maturity: treat as single-period deposit/short OIS
            alpha0 = T[0]
            P[0] = 1.0 / (1.0 + S[0] * alpha0)

            for n in range(1, len(T)):
                alpha_n = T[n] - T[n - 1]
                # sum of fixed-leg discounted accruals for earlier periods
                acc = 0.0
                for i in range(n):
                    alpha_i = T[i] - (T[i - 1] if i > 0 else 0.0)
                    acc += alpha_i * P[i]
                # par condition: S_n * sum_{i=1}^n alpha_i P(0,T_i) = 1 - P(0,T_n)
                P[n] = (1.0 - S[n] * acc) / (1.0 + S[n] * alpha_n)

            df["df"] = P
            # continuously-compounded zero rate consistent with P(0,T)
            df["zero_rate"] = -np.log(df["df"]) / df["T"]

        else:
            # simple approximation: treat par rate as flat zero rate to that maturity
            df["zero_rate"] = df["par_rate"]
            df["df"] = np.exp(-df["zero_rate"] * df["T"])

        self.curve = df[["T", "zero_rate", "df"]].reset_index(drop=True)

    def discount_factor(self, T: float) -> float:
        """
        Log-linear interpolation of discount factor P(0,T) for any maturity T (years).
        """
        times = self.curve["T"].values
        dfs = self.curve["df"].values
        if T <= times[0]:
            return float(dfs[0])
        if T >= times[-1]:
            return float(dfs[-1])
        log_dfs = np.log(dfs)
        log_df_T = np.interp(T, times, log_dfs)
        return float(np.exp(log_df_T))

    def curve_on_grid(self, year_grid):
        """
        Return curve sampled on a custom year grid (useful for FMM T_1,...,T_N).
        """
        year_grid = np.asarray(year_grid, dtype=float)
        dfs = np.array([self.discount_factor(t) for t in year_grid])
        # avoid division by zero if someone passes T=0
        zero = np.where(year_grid > 0,
                        -np.log(dfs) / year_grid,
                        dfs * 0.0)
        return pd.DataFrame({"T": year_grid, "zero_rate": zero, "df": dfs})

    # -------- 0.b ATM swaption surface --------

    def _build_atm_surface(self):
        atm = self.raw_atm.copy()
        tenor_cols = [c for c in atm.columns if c != "Expiry"]

        for _, row in atm.iterrows():
            expiry_str = row["Expiry"]
            expiry = self._tenor_str_to_years(expiry_str)

            for col in tenor_cols:
                vol = row[col]
                if pd.isna(vol):
                    continue
                tenor = self._tenor_str_to_years(col)

                # ATM vols are quoted in % -> convert to decimal
                self.atm_vols[(expiry, tenor)] = float(vol) / 100.0

    # -------- 0.c OTM smiles (relative to ATM) --------

    @staticmethod
    def _parse_moneyness_col(col: str) -> float:
        """
        Parse column name like '-200bps' or '25bps' into a number of bps.
        Returns [-200, -100, ..., 200].
        """
        col = col.strip()
        sign = -1.0 if col.startswith("-") else 1.0
        num = "".join(ch for ch in col if ch.isdigit())
        if not num:
            raise ValueError(f"Cannot parse moneyness column {col}")
        return sign * float(num)

    def _build_smile_surface(self):
        otm = self.raw_otm.copy()
        moneyness_cols = [c for c in otm.columns if "bps" in c]

        for _, row in otm.iterrows():
            pair = row["Term x Tenor"]
            if not isinstance(pair, str):
                continue

            expiry_str, tenor_str = [p.strip() for p in pair.split("X")]
            expiry = self._tenor_str_to_years(expiry_str)
            tenor = self._tenor_str_to_years(tenor_str)

            atm_vol = self.atm_vols.get((expiry, tenor), np.nan)
            if not np.isfinite(atm_vol):
                # No ATM vol -> we can’t build a smile here
                continue

            moneyness_bps = []
            vols = []

            for col in moneyness_cols:
                skew_val = row[col]
                if pd.isna(skew_val):
                    continue

                m = self._parse_moneyness_col(col)
                moneyness_bps.append(m)

                # ASSUMPTION:
                # OTM file is "Black Vol Skew / Absolute" in *vol bps*.
                # 100 vol-bps = 1% vol = 0.01 in decimal.
                total_vol = atm_vol + float(skew_val) / 1e4
                vols.append(total_vol)

            if moneyness_bps:
                moneyness_bps = np.array(moneyness_bps, dtype=float)
                vols = np.array(vols, dtype=float)

                self.smiles[(expiry, tenor)] = SwaptionSmile(
                    expiry=expiry,
                    tenor=tenor,
                    moneyness_bps=moneyness_bps,
                    vol=vols,
                    atm_vol=atm_vol,
                )

    # -------- convenience accessors --------

    def list_available_pairs(self) -> pd.DataFrame:
        """
        List all (expiry, tenor) combinations where we have ATM vols.
        """
        rows = []
        for (e, t), v in sorted(self.atm_vols.items()):
            rows.append({"expiry": e, "tenor": t, "atm_vol": v})
        return pd.DataFrame(rows)

    def get_smile(self, expiry_years: float, tenor_years: float):
        """
        Return SwaptionSmile object for a given (expiry, tenor) in years,
        or None if no smile is available.
        """
        return self.smiles.get((expiry_years, tenor_years), None)
