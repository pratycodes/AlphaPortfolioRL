#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export RUN_SLUG="${RUN_SLUG:-champion_base_multiseed}"
export CHAMPION_VARIANTS="champion_base"
export RUN_SEEDS="${RUN_SEEDS:-42 7 123 2024 2025}"
export CHAMPION_MODEL_DIR="${CHAMPION_MODEL_DIR:-models/${RUN_SLUG}}"

exec "${SCRIPT_DIR}/run_champion_arb_sparse_matrix.sh" "$@"
