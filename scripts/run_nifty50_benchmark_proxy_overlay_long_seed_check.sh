#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

export RUN_SLUG="${RUN_SLUG:-nifty50_benchmark_proxy_overlay_long_seed_check}"
export MODEL_DIR="${MODEL_DIR:-models/${RUN_SLUG}}"

export ASSETS="${ASSETS:-[\"RELIANCE.NS\",\"HDFCBANK.NS\",\"ICICIBANK.NS\",\"INFY.NS\",\"TCS.NS\",\"BHARTIARTL.NS\",\"LT.NS\",\"ITC.NS\",\"SBIN.NS\",\"HINDUNILVR.NS\"]}"
export MARKET_FEATURE_TICKER="${MARKET_FEATURE_TICKER:-^NSEI}"
export BENCHMARK_TICKER="${BENCHMARK_TICKER:-^NSEI}"
export BENCHMARK_NAME="${BENCHMARK_NAME:-NIFTY 50}"
export DATA_CACHE_DIR="${DATA_CACHE_DIR:-data_cache/nifty50_smoke}"

export TRAIN_START_DATE="${TRAIN_START_DATE:-2010-01-01}"
export TRAIN_END_DATE="${TRAIN_END_DATE:-2023-12-31}"
export VALID_START_DATE="${VALID_START_DATE:-2024-01-01}"
export VALID_END_DATE="${VALID_END_DATE:-2024-12-31}"
export TEST_START_DATE="${TEST_START_DATE:-2025-01-01}"
export TEST_END_DATE="${TEST_END_DATE:-2026-05-27}"

export USE_ACTIVE_OVERLAY="true"
export ACTIVE_OVERLAY_BASE_POLICY="Benchmark Proxy"
export ACTIVE_OVERLAY_BASE_WEIGHT="${ACTIVE_OVERLAY_BASE_WEIGHT:-0.90}"
export ACTIVE_OVERLAY_TILT_WEIGHT="${ACTIVE_OVERLAY_TILT_WEIGHT:-0.10}"
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
