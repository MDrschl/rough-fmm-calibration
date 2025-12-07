import numpy as np
from preprocessing import USDRFRPreprocessor

pre = USDRFRPreprocessor(
    sofr_path="Data/SOFRRates.xlsx",
    atm_path="Data/ATMSwaptionIVUSD.xlsx",
    otm_path="Data/OTMSwaptionIVUSD.xlsx",
    use_bootstrap=True,
)


# ---------- 1) zero curve needed for these swaptions ----------

tenor_years = 1.0
expiries = np.array([1.0, 3.0, 5.0, 7.0, 10.0])

T_max = expiries.max() + tenor_years

zero_curve = pre.curve[pre.curve["T"] <= T_max].copy().reset_index(drop=True)

print(zero_curve.head())
print(zero_curve.tail())

calib_smiles = {}

for T in expiries:
    sm = pre.get_smile(expiry_years=T, tenor_years=tenor_years)
    if sm is None:
        print(f"No smile available for {T}Y x {tenor_years}Y – skipping.")
        continue
    calib_smiles[(T, tenor_years)] = sm


zero_curve