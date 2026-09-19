#!/bin/sh
# Reproduce every number in README.md from df_clean.pkl (about 20 seconds).
set -e
mkdir -p out figures
python3 1_extract_events.py
python3 2_fit_size_law.py
python3 3_fit_ceiling_law.py
python3 4_figures.py
