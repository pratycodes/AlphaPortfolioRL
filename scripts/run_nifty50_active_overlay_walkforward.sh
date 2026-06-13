#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

"${SCRIPT_DIR}/run_nifty50_active_overlay_walkforward_2024.sh"
"${SCRIPT_DIR}/run_nifty50_active_overlay_long_seed_check.sh"
"${SCRIPT_DIR}/select_walkforward_champion.sh"
