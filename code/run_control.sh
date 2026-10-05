#!/bin/bash
# Control-only part of the pipeline (after density calibration/validation).
set -e
python3 tune_pid.py
N_SEEDS=10 NPROC=2 python3 run_all.py
python3 sweeps.py
python3 analysis.py
python3 make_figures.py
python3 make_tables.py
