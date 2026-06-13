#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

"${SCRIPT_DIR}/run_nifty50_stable_policy_walkforward_2024.sh"
"${SCRIPT_DIR}/evaluate_nifty50_stable_policy_walkforward_2024_ensemble.sh"

"${PYTHON_BIN}" - <<'PY'
from pathlib import Path

import pandas as pd

current_path = Path("runs/nifty50_stable_policy_long_seed_check_ensemble_metrics.csv")
walkforward_path = Path("runs/nifty50_stable_policy_walkforward_2024_ensemble_metrics.csv")
output_path = Path("champion/walkforward_comparison.csv")

if not current_path.exists():
    raise FileNotFoundError(f"Missing current champion metrics: {current_path}")
if not walkforward_path.exists():
    raise FileNotFoundError(f"Missing 2024 walk-forward metrics: {walkforward_path}")

rows = []
for window, path in [
    ("2025-01-01_to_2026-05-27", current_path),
    ("2024-01-01_to_2024-12-31", walkforward_path),
]:
    metrics = pd.read_csv(path)
    selected = metrics.loc[metrics["Strategy"] == "RL Ensemble"].iloc[0]
    benchmark = metrics.loc[metrics["Strategy"] == "NIFTY 50"].iloc[0]
    equal_weight = metrics.loc[metrics["Strategy"] == "Equal Weight"].iloc[0]
    rows.append(
        {
            "window": window,
            "rl_return": selected["Total Return"],
            "rl_sharpe": selected["Sharpe Ratio"],
            "rl_max_drawdown": selected["Max Drawdown"],
            "nifty_return": benchmark["Total Return"],
            "nifty_sharpe": benchmark["Sharpe Ratio"],
            "nifty_max_drawdown": benchmark["Max Drawdown"],
            "equal_weight_return": equal_weight["Total Return"],
            "equal_weight_sharpe": equal_weight["Sharpe Ratio"],
            "return_gap_vs_nifty": selected["Total Return"] - benchmark["Total Return"],
            "sharpe_gap_vs_nifty": selected["Sharpe Ratio"] - benchmark["Sharpe Ratio"],
            "return_gap_vs_equal_weight": selected["Total Return"] - equal_weight["Total Return"],
            "sharpe_gap_vs_equal_weight": selected["Sharpe Ratio"] - equal_weight["Sharpe Ratio"],
        }
    )

output_path.parent.mkdir(parents=True, exist_ok=True)
comparison = pd.DataFrame(rows)
comparison.to_csv(output_path, index=False)
print(comparison.round(4).to_string(index=False))
print(f"Wrote comparison to {output_path}")
PY
