#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

"${SCRIPT_DIR}/run_nifty50_benchmark_proxy_overlay_walkforward_2024.sh"
"${SCRIPT_DIR}/run_nifty50_benchmark_proxy_overlay_long_seed_check.sh"
