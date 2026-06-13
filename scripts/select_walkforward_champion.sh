#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/private/tmp/matplotlib-alpha-portfolio-rl}"

"${PYTHON_BIN}" -m evaluation.walkforward_champion_selector --output-dir champion
