#!/usr/bin/env bash
# Benchmark STA / GATv2 sur un export observ-it (CSV ou répertoire dataset_*.csv).
# Prérequis : venv du projet avec torch (voir README).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT/rca_pipeline"
exec "${ROOT}/.venv/bin/python" benchmark_real.py "$@"
