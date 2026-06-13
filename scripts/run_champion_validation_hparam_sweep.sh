#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

SWEEP_SLUG="${SWEEP_SLUG:-champion_validation_hparam_sweep}"
SWEEP_SEEDS="${SWEEP_SEEDS:-42 7 123}"

# name dropout lr_actor bcm_lambda param_noise_init_std return_reward_scale
CANDIDATES=(
  "baseline 0.50 1e-5 0.10 0.010 1000.0"
  "dropout_030 0.30 1e-5 0.10 0.010 1000.0"
  "bcm_005 0.50 1e-5 0.05 0.010 1000.0"
  "actor_lr_005 0.50 5e-6 0.10 0.010 1000.0"
)

for candidate in "${CANDIDATES[@]}"; do
  read -r name dropout lr_actor bcm_lambda param_noise_init return_scale <<<"${candidate}"

  export RUN_SLUG="${SWEEP_SLUG}_${name}"
  export CHAMPION_VARIANTS="champion_base"
  export RUN_SEEDS="${SWEEP_SEEDS}"
  export CHAMPION_MODEL_DIR="models/${RUN_SLUG}"

  export CHAMPION_BASE_DROPOUT="${dropout}"
  export LR_ACTOR="${lr_actor}"
  export BCM_LAMBDA="${bcm_lambda}"
  export PARAM_NOISE_INIT_STD="${param_noise_init}"
  export CHAMPION_RETURN_REWARD_SCALE="${return_scale}"

  echo "Running validation sweep candidate=${name} seeds=${SWEEP_SEEDS}"
  "${SCRIPT_DIR}/run_champion_arb_sparse_matrix.sh" --skip-test-eval "$@"
done

"${PYTHON_BIN}" - <<'PY'
import os
from pathlib import Path

import pandas as pd

slug = os.environ.get("SWEEP_SLUG", "champion_validation_hparam_sweep")
candidate_files = sorted(Path("runs").glob(f"{slug}_*_summary.csv"))
frames = []
for path in candidate_files:
    candidate = path.name.removeprefix(f"{slug}_").removesuffix("_summary.csv")
    frame = pd.read_csv(path)
    frame.insert(0, "candidate", candidate)
    frames.append(frame)

if not frames:
    raise SystemExit(f"No sweep summaries found for {slug}")

raw = pd.concat(frames, ignore_index=True)
raw_path = Path(f"runs/{slug}_raw_summary.csv")
raw.to_csv(raw_path, index=False)

grouped = raw.groupby("candidate", as_index=False).agg(
    seeds=("seed", "nunique"),
    val_score_mean=("best_val_selection_score", "mean"),
    val_score_std=("best_val_selection_score", "std"),
    val_return_mean=("best_val_total_return", "mean"),
    val_return_std=("best_val_total_return", "std"),
    val_sharpe_mean=("best_val_sharpe_ratio", "mean"),
    val_sharpe_std=("best_val_sharpe_ratio", "std"),
    val_drawdown_mean=("best_val_max_drawdown", "mean"),
    val_turnover_mean=("best_val_average_turnover", "mean"),
    lr_actor=("lr_actor", "first"),
    bcm_lambda=("bcm_lambda", "first"),
    dropout=("dropout", "first"),
)
grouped = grouped.sort_values("val_score_mean", ascending=False)
summary_path = Path(f"runs/{slug}_validation_summary.csv")
grouped.to_csv(summary_path, index=False)

print(f"Wrote raw sweep summary to {raw_path}")
print(f"Wrote grouped validation summary to {summary_path}")
print(grouped.round(4).to_string(index=False))
PY
