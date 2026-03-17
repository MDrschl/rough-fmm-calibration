#!/usr/bin/env python3
"""Auto-generated driver: P2 | EUR | cross                | 2025-12-08"""
import sys, os, shutil, json

# Work from the code directory (one level up from results/)
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

patch = {
  "data_file": "eur_swaption_data.pkl",
  "in_sample_date": "2025-12-08",
  "out_sample_date": "2025-12-09",
  "device": "cpu",
  "dtype": "float64",
  "mode": "cross"
}

# Patch CONFIG before __main__ executes
import calibration
for k, v in patch.items():
    if k in calibration.CONFIG:
        calibration.CONFIG[k] = v

# Run by re-executing the module
exec(compile(open('calibration.py').read(), 'calibration.py', 'exec'))
