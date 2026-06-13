#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export RUN_SLUG="${RUN_SLUG:-champion_arb_relaxed_matrix}"
export CHAMPION_VARIANTS="${CHAMPION_VARIANTS:-champion_arb champion_arb_sparse}"

# Activation-test gates: this is meant to test ARB replay, not just shadow observation.
# Validation is still reported, but it is not used as a hard gate here; otherwise
# weak validation blocks ARB before the ablation can measure replay behavior.
export CHAMPION_ARB_MIN_EPISODE="${CHAMPION_ARB_MIN_EPISODE:-20}"
export CHAMPION_ARB_STABILITY_PATIENCE="${CHAMPION_ARB_STABILITY_PATIENCE:-1}"
export CHAMPION_ARB_POLICY_DRIFT_THRESHOLD="${CHAMPION_ARB_POLICY_DRIFT_THRESHOLD:-0.35}"
export CHAMPION_ARB_MIN_PORTFOLIO_VALUE_RATIO="${CHAMPION_ARB_MIN_PORTFOLIO_VALUE_RATIO:-0.90}"
export CHAMPION_ARB_MAX_ACTIVATION_TURNOVER="${CHAMPION_ARB_MAX_ACTIVATION_TURNOVER:-0.20}"
export CHAMPION_ARB_MIN_VALIDATION_SCORE="${CHAMPION_ARB_MIN_VALIDATION_SCORE:--999.0}"
export CHAMPION_ARB_MAX_MIX="${CHAMPION_ARB_MAX_MIX:-0.20}"
export CHAMPION_ARB_RAMP_EPISODES="${CHAMPION_ARB_RAMP_EPISODES:-60}"
export CHAMPION_ARB_TEMPERATURE="${CHAMPION_ARB_TEMPERATURE:-0.70}"
export CHAMPION_ARB_MIN_PROBABILITY="${CHAMPION_ARB_MIN_PROBABILITY:-0.002}"

exec "${SCRIPT_DIR}/run_champion_arb_sparse_matrix.sh" "$@"
