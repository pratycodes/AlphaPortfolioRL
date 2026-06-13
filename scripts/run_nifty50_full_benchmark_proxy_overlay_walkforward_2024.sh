#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

source "${SCRIPT_DIR}/nifty50_full_universe.sh"

export RUN_SLUG="${RUN_SLUG:-nifty50_full_benchmark_proxy_overlay_walkforward_2024}"
export MODEL_DIR="${MODEL_DIR:-models/${RUN_SLUG}}"

export TRAIN_START_DATE="2010-01-01"
export TRAIN_END_DATE="2022-12-31"
export VALID_START_DATE="2023-01-01"
export VALID_END_DATE="2023-12-31"
export TEST_START_DATE="2024-01-01"
export TEST_END_DATE="2024-12-31"

export USE_ACTIVE_OVERLAY="true"
export ACTIVE_OVERLAY_BASE_POLICY="Benchmark Proxy"
export ACTIVE_OVERLAY_BASE_WEIGHT="${ACTIVE_OVERLAY_BASE_WEIGHT:-0.95}"
export ACTIVE_OVERLAY_TILT_WEIGHT="${ACTIVE_OVERLAY_TILT_WEIGHT:-0.05}"
export ACTIVE_OVERLAY_TRACKING_PENALTY="${ACTIVE_OVERLAY_TRACKING_PENALTY:-0.00}"

export USE_TARGET_POLICY_EVAL="true"
export STABLE_POLICY_SELECTION_VERSION="1"

"${PYTHON_BIN}" -m data.bootstrap_paper_data

export ACTIVE_OVERLAY_BASE_WEIGHTS="$(
  "${PYTHON_BIN}" "${SCRIPT_DIR}/fit_benchmark_proxy_weights.py" \
    --start-date "${TRAIN_START_DATE}" \
    --end-date "${TRAIN_END_DATE}"
)"
echo "Benchmark proxy risky weights: ${ACTIVE_OVERLAY_BASE_WEIGHTS}"

exec "${SCRIPT_DIR}/run_nifty50_baseline_cash_long_seed_check.sh"
