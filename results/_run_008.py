#!/usr/bin/env python3
"""Auto-generated driver: P1 | USD | two_stage            | 2025-12-08 → 2025-12-09"""
import sys, os, shutil, json

# Work from the code directory (one level up from results/)
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

patch = {
  "data_file": "usd_swaption_data.pkl",
  "in_sample_date": "2025-12-08",
  "out_sample_date": "2025-12-09",
  "device": "cpu",
  "dtype": "float64",
  "mode": "two_stage"
}

# Write config override file (calibration.py reads this at startup)
with open('_config_override.json', 'w') as _f:
    json.dump(patch, _f, indent=2)

# Run calibration (picks up _config_override.json automatically)
try:
    exec(compile(open('calibration.py').read(), 'calibration.py', 'exec'))
finally:
    # Clean up override file
    if os.path.exists('_config_override.json'):
        os.remove('_config_override.json')
