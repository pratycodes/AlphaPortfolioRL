#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export RUN_SLUG="${RUN_SLUG:-nifty50_stable_policy_walkforward_2024}"
export MODEL_DIR="${MODEL_DIR:-models/${RUN_SLUG}}"

export TRAIN_START_DATE="2010-01-01"
export TRAIN_END_DATE="2022-12-31"
export VALID_START_DATE="2023-01-01"
export VALID_END_DATE="2023-12-31"
export TEST_START_DATE="2024-01-01"
export TEST_END_DATE="2024-12-31"

export USE_TARGET_POLICY_EVAL="true"
export STABLE_POLICY_SELECTION_VERSION="1"

exec "${SCRIPT_DIR}/run_nifty50_baseline_cash_long_seed_check.sh"
