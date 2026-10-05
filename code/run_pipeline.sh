#!/bin/bash
# Full reproduction pipeline (order matters).
set -e
python3 sw_data.py
python3 tim.py            # calibrate TIM (2011-2019) at 550 and 400 km
python3 validate_tim.py   # out-of-sample storms
python3 calib_sigma.py    # sigma_k from calibration storms
python3 coverage.py       # coverage on test storms
python3 tune_pid.py       # PI tuning on Sep-2017 calibration storm
N_SEEDS=10 NPROC=2 python3 run_all.py
python3 sweeps.py
python3 analysis.py
python3 make_figures.py
python3 make_tables.py
python3 review_extensions.py
python3 operational_calibration.py
python3 resource_campaign.py
python3 paired_resource_campaign.py
python3 common_recovery_campaign.py
python3 resource_sensitivity.py
python3 verify_resource_sensitivity.py
python3 verify_common_recovery.py
