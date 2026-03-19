#!/usr/bin/env python3
"""Auto-generated driver: P2 | USD | roughness           | 2024-12-09"""
import sys, os, shutil, json

# Work from the code directory (one level up from results/)
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

prereq = 'results/usd/phase1_modes/hybrid_two_stage/2024-12-09'
base_pt = os.path.join(prereq, 'amcc_calibration_results.pt')
if os.path.exists(base_pt):
    shutil.copy2(base_pt, 'amcc_calibration_results.pt')
    print(f'Copied base calibration from {base_pt}')
else:
    print(f'WARNING: {base_pt} not found')
    sys.exit(1)

patch = {
  "data_file": "usd_swaption_data.pkl",
  "in_sample_date": "2024-12-09",
  "out_sample_date": "2024-12-10",
  "device": "cpu",
  "dtype": "float64",
  "mode": "roughness"
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
