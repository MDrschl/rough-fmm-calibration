# main.py

import numpy as np
from preprocessing import SwaptionSmile,USDRFRPreprocessor

pre = USDRFRPreprocessor(
    sofr_path="Data/SOFRRates.xlsx",
    atm_path="Data/ATMSwaptionIVUSD.xlsx",
    otm_path="Data/OTMSwaptionIVUSD.xlsx",
    use_bootstrap=True
)

# 0. Inputs
curve_1y = pre.curve_on_grid(np.arange(1, 11, 1.0))

# ATM vol for 1Y × 1Y (maturity 1Y, tenor 1Y)
atm_1x1 = pre.atm_vols[(1.0, 1.0)]

# Full smile for 1Y × 1Y
smile_1x1 = pre.get_smile(1.0, 1.0)
print(smile_1x1.moneyness_bps)
print(smile_1x1.vol)

# List all expiry/tenor pairs we have
print(pre.list_available_pairs().head())

pre.curve