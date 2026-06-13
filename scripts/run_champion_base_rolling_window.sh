#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

ROLLING_SLUG="${ROLLING_SLUG:-champion_base_rolling_window}"
ROLLING_SEEDS="${ROLLING_SEEDS:-123}"
ROLLING_VARIANTS="champion_base"

# Rolling windows use an 8-year train window, 1-year validation window, and
# 1-year test window where possible. The last fold uses the current extended
# OOS window through 2026-05-27.
FOLDS=(
  "test_2019 2010-01-01 2017-12-31 2018-01-01 2018-12-31 2019-01-01 2019-12-31"
  "test_2020 2011-01-01 2018-12-31 2019-01-01 2019-12-31 2020-01-01 2020-12-31"
  "test_2021 2012-01-01 2019-12-31 2020-01-01 2020-12-31 2021-01-01 2021-12-31"
  "test_2022 2013-01-01 2020-12-31 2021-01-01 2021-12-31 2022-01-01 2022-12-31"
  "test_2023 2014-01-01 2021-12-31 2022-01-01 2022-12-31 2023-01-01 2023-12-31"
  "test_2024 2015-01-01 2022-12-31 2023-01-01 2023-12-31 2024-01-01 2024-12-31"
  "test_2025_2026 2016-01-01 2023-12-31 2024-01-01 2024-12-31 2025-01-01 2026-05-27"
)

mkdir -p runs

for fold in "${FOLDS[@]}"; do
  read -r fold_name train_start train_end valid_start valid_end test_start test_end <<<"${fold}"

  export RUN_SLUG="${ROLLING_SLUG}_${fold_name}"
  export RUN_SEEDS="${ROLLING_SEEDS}"
  export CHAMPION_VARIANTS="${ROLLING_VARIANTS}"
  export CHAMPION_MODEL_DIR="models/${RUN_SLUG}"

  export CHAMPION_TRAIN_START_DATE="${train_start}"
  export CHAMPION_TRAIN_END_DATE="${train_end}"
  export CHAMPION_VALID_START_DATE="${valid_start}"
  export CHAMPION_VALID_END_DATE="${valid_end}"
  export CHAMPION_TEST_START_DATE="${test_start}"
  export CHAMPION_TEST_END_DATE="${test_end}"

  echo "Running fold=${fold_name}"
  echo "  train=${train_start} -> ${train_end}"
  echo "  valid=${valid_start} -> ${valid_end}"
  echo "  test =${test_start} -> ${test_end}"

  "${SCRIPT_DIR}/run_champion_arb_sparse_matrix.sh" "$@"
done

"${PYTHON_BIN}" - <<'PY'
import os
from pathlib import Path

import pandas as pd

rolling_slug = os.environ.get("ROLLING_SLUG", "champion_base_rolling_window")
folds = [
    ("test_2019", "2010-01-01", "2017-12-31", "2018-01-01", "2018-12-31", "2019-01-01", "2019-12-31"),
    ("test_2020", "2011-01-01", "2018-12-31", "2019-01-01", "2019-12-31", "2020-01-01", "2020-12-31"),
    ("test_2021", "2012-01-01", "2019-12-31", "2020-01-01", "2020-12-31", "2021-01-01", "2021-12-31"),
    ("test_2022", "2013-01-01", "2020-12-31", "2021-01-01", "2021-12-31", "2022-01-01", "2022-12-31"),
    ("test_2023", "2014-01-01", "2021-12-31", "2022-01-01", "2022-12-31", "2023-01-01", "2023-12-31"),
    ("test_2024", "2015-01-01", "2022-12-31", "2023-01-01", "2023-12-31", "2024-01-01", "2024-12-31"),
    ("test_2025_2026", "2016-01-01", "2023-12-31", "2024-01-01", "2024-12-31", "2025-01-01", "2026-05-27"),
]


def load_fold_csv(fold_name, suffix):
    path = Path("runs") / f"{rolling_slug}_{fold_name}_{suffix}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing fold output: {path}")
    return pd.read_csv(path)


validation_frames = []
test_frames = []
grouped_frames = []

for fold_name, train_start, train_end, valid_start, valid_end, test_start, test_end in folds:
    fold_metadata = {
        "fold": fold_name,
        "train_start_date": train_start,
        "train_end_date": train_end,
        "valid_start_date": valid_start,
        "valid_end_date": valid_end,
        "test_start_date": test_start,
        "test_end_date": test_end,
    }

    validation = load_fold_csv(fold_name, "summary")
    test = load_fold_csv(fold_name, "test_summary")
    grouped = load_fold_csv(fold_name, "test_grouped_summary")

    for key, value in fold_metadata.items():
        validation[key] = value
        test[key] = value
        grouped[key] = value

    validation_frames.append(validation)
    test_frames.append(test)
    grouped_frames.append(grouped)

validation_all = pd.concat(validation_frames, ignore_index=True)
test_all = pd.concat(test_frames, ignore_index=True)
grouped_all = pd.concat(grouped_frames, ignore_index=True)

validation_path = Path("runs") / f"{rolling_slug}_validation_all_folds.csv"
test_path = Path("runs") / f"{rolling_slug}_test_all_folds.csv"
grouped_path = Path("runs") / f"{rolling_slug}_test_grouped_all_folds.csv"
summary_path = Path("runs") / f"{rolling_slug}_rolling_summary.csv"

validation_all.to_csv(validation_path, index=False)
test_all.to_csv(test_path, index=False)
grouped_all.to_csv(grouped_path, index=False)

summary = test_all.groupby("ablation", as_index=False).agg(
    folds=("fold", "nunique"),
    seeds=("seed", "nunique"),
    rl_return_mean=("rl_return", "mean"),
    rl_return_std=("rl_return", "std"),
    rl_sharpe_mean=("rl_sharpe", "mean"),
    rl_sharpe_std=("rl_sharpe", "std"),
    rl_max_drawdown_mean=("rl_max_drawdown", "mean"),
    rl_max_drawdown_worst=("rl_max_drawdown", "min"),
    rl_avg_turnover_mean=("rl_avg_turnover", "mean"),
    return_gap_vs_benchmark_mean=("return_gap_vs_benchmark", "mean"),
    return_gap_vs_crp_mean=("return_gap_vs_crp", "mean"),
    return_gap_vs_equal_weight_mean=("return_gap_vs_equal_weight", "mean"),
    return_gap_vs_buy_hold_ew_mean=("return_gap_vs_buy_hold_ew", "mean"),
    sharpe_gap_vs_benchmark_mean=("sharpe_gap_vs_benchmark", "mean"),
    sharpe_gap_vs_crp_mean=("sharpe_gap_vs_crp", "mean"),
    sharpe_gap_vs_equal_weight_mean=("sharpe_gap_vs_equal_weight", "mean"),
    sharpe_gap_vs_buy_hold_ew_mean=("sharpe_gap_vs_buy_hold_ew", "mean"),
)

for baseline in ["benchmark", "crp", "equal_weight", "buy_hold_ew"]:
    summary[f"beats_{baseline}_return_mean"] = summary[f"return_gap_vs_{baseline}_mean"] > 0.0
    summary[f"beats_{baseline}_sharpe_mean"] = summary[f"sharpe_gap_vs_{baseline}_mean"] > 0.0

summary.to_csv(summary_path, index=False)

print(f"Wrote validation folds to {validation_path}")
print(f"Wrote test folds to {test_path}")
print(f"Wrote grouped fold summaries to {grouped_path}")
print(f"Wrote rolling summary to {summary_path}")
print(summary.round(4).to_string(index=False))
PY
