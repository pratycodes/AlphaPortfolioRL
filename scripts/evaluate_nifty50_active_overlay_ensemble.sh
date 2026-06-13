#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export RUN_SLUG="${RUN_SLUG:-nifty50_active_overlay_long_seed_check}"
export MODEL_DIR="${MODEL_DIR:-models/${RUN_SLUG}}"

export USE_ACTIVE_OVERLAY="true"
export ACTIVE_OVERLAY_BASE_POLICY="${ACTIVE_OVERLAY_BASE_POLICY:-Equal Weight}"
export ACTIVE_OVERLAY_BASE_WEIGHT="${ACTIVE_OVERLAY_BASE_WEIGHT:-0.80}"
export ACTIVE_OVERLAY_TILT_WEIGHT="${ACTIVE_OVERLAY_TILT_WEIGHT:-0.20}"
export ACTIVE_OVERLAY_TRACKING_PENALTY="${ACTIVE_OVERLAY_TRACKING_PENALTY:-0.05}"
export ENSEMBLE_TOP_K="${ENSEMBLE_TOP_K:-3}"
export ENSEMBLE_WEIGHTING="${ENSEMBLE_WEIGHTING:-softmax}"

exec "${SCRIPT_DIR}/evaluate_nifty50_stable_policy_ensemble.sh"
