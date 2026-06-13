#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export RUN_SLUG="${RUN_SLUG:-nifty50_stable_policy_long_seed_check}"
export MODEL_DIR="${MODEL_DIR:-models/${RUN_SLUG}}"
export USE_TARGET_POLICY_EVAL="true"
export STABLE_POLICY_SELECTION_VERSION="1"

exec "${SCRIPT_DIR}/run_nifty50_baseline_cash_long_seed_check.sh"
