#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
case_path=$(rca data --demo --root "${RCA_DATA:-$HOME/data/rcaeval}")
rca rank "$case_path" --top-k 3
python showcase/reference.py
