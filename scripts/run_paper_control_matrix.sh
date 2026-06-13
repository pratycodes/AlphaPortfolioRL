#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_SLUG="${RUN_SLUG:-paper_control_matrix}"
RUN_SEEDS="${RUN_SEEDS:-42 7 123}"
PAPER_CONTROL_ABLATIONS="${PAPER_CONTROL_ABLATIONS:-paper_baseline paper_ipm paper_dam paper_bcm paper_ipm_bcm paper_all}"

MANIFEST="runs/${RUN_SLUG}_matrix.json"
SUMMARY="runs/${RUN_SLUG}_summary.csv"
TEST_SUMMARY="runs/${RUN_SLUG}_test_summary.csv"
RUNS_DIR="runs/${RUN_SLUG}_runs"

export RUN_SLUG
export RUN_SEEDS
export PAPER_CONTROL_ABLATIONS

# Paper-control universe from Yu et al. (2019): six U.S. equities + cash.
# The market index is an observed feature and benchmark, not a tradable asset.
export ASSETS='["COST","CSCO","F","GS","AIG","CAT"]'
export USE_MARKET_FEATURE="true"
export MARKET_FEATURE_TICKER="^GSPC"
export BENCHMARK_TICKER="^GSPC"
export BENCHMARK_NAME="S&P 500"

export DATA_CACHE_DIR="${PAPER_CONTROL_DATA_CACHE_DIR:-data_cache/paper_control}"
export USE_DATA_CACHE="true"
export REFRESH_DATA_CACHE="${REFRESH_DATA_CACHE:-false}"
export REQUIRE_DATA_CACHE="true"
export EXPERIMENT_DIR="${RUNS_DIR}"
export MODEL_DIR="${PAPER_CONTROL_MODEL_DIR:-models/${RUN_SLUG}}"

export TRAIN_START_DATE="${PAPER_CONTROL_TRAIN_START_DATE:-2010-01-01}"
export TRAIN_END_DATE="${PAPER_CONTROL_TRAIN_END_DATE:-2023-12-31}"
export VALID_START_DATE="${PAPER_CONTROL_VALID_START_DATE:-2024-01-01}"
export VALID_END_DATE="${PAPER_CONTROL_VALID_END_DATE:-2024-12-31}"
export TEST_START_DATE="${PAPER_CONTROL_TEST_START_DATE:-2025-01-01}"
export TEST_END_DATE="${PAPER_CONTROL_TEST_END_DATE:-2026-05-27}"

export EPISODES="${PAPER_CONTROL_EPISODES:-200}"
export EPISODE_LENGTH="${PAPER_CONTROL_EPISODE_LENGTH:-650}"
export BATCH_SIZE="${PAPER_CONTROL_BATCH_SIZE:-128}"
export BUFFER_SIZE="${PAPER_CONTROL_BUFFER_SIZE:-1000}"
export IPM_PRETRAIN_EPOCHS="${PAPER_CONTROL_IPM_PRETRAIN_EPOCHS:-50}"
export GAN_EPOCHS="${PAPER_CONTROL_GAN_EPOCHS:-200}"
export CHECKPOINT_FREQ="${PAPER_CONTROL_CHECKPOINT_FREQ:-10}"
export VALIDATION_FREQ="${PAPER_CONTROL_VALIDATION_FREQ:-10}"
export MIN_SAVE_EPISODE="${PAPER_CONTROL_MIN_SAVE_EPISODE:-40}"
export EARLY_STOPPING_PATIENCE="${PAPER_CONTROL_EARLY_STOPPING_PATIENCE:-4}"
export EARLY_STOPPING_MIN_DELTA="${PAPER_CONTROL_EARLY_STOPPING_MIN_DELTA:-0.001}"

export REBALANCE_FREQ="1"
export REWARD_MODE="log_return"
export TRAINING_BENCHMARK_POLICY="CRP"
export RETURN_REWARD_SCALE="${PAPER_CONTROL_RETURN_REWARD_SCALE:-1000.0}"
export TRADING_COST_BPS="${PAPER_CONTROL_TRADING_COST_BPS:-20.0}"
export SLIPPAGE_BPS="0.0"
export SPREAD_BPS="0.0"
export MARKET_IMPACT_BPS="0.0"
export FIXED_COMMISSION="0.0"
export TURNOVER_PENALTY="0.0"
export CONCENTRATION_PENALTY="0.0"
export DRAWDOWN_PENALTY="0.0"
export CASH_PENALTY="0.0"
export MAX_WEIGHT="none"
export MAX_CASH_WEIGHT="none"

export USE_PARAMETER_NOISE="true"
export USE_ONLINE_IPM="true"
export USE_ARB="false"
export USE_SPARSE_NETWORK="false"
export USE_PRIORITIZED_REPLAY="true"
export USE_ACTIVE_OVERLAY="false"
unset ACTIVE_OVERLAY_BASE_WEIGHTS

export MODEL_SELECTION_METRIC="Benchmark Relative Score"
export MODEL_SELECTION_HURDLE="CRP"
export MODEL_SELECTION_HURDLES='["CRP","Equal Weight","Buy & Hold EW"]'
export USE_TARGET_POLICY_EVAL="true"
export STABLE_POLICY_SELECTION_VERSION="1"

mkdir -p runs "${RUNS_DIR}"

"${PYTHON_BIN}" - <<'PY'
import json
import os
from pathlib import Path

ablations = os.environ["PAPER_CONTROL_ABLATIONS"].split()
seeds = [int(seed) for seed in os.environ["RUN_SEEDS"].split()]
split = {
    "train_start_date": os.environ["TRAIN_START_DATE"],
    "train_end_date": os.environ["TRAIN_END_DATE"],
    "valid_start_date": os.environ["VALID_START_DATE"],
    "valid_end_date": os.environ["VALID_END_DATE"],
    "test_start_date": os.environ["TEST_START_DATE"],
    "test_end_date": os.environ["TEST_END_DATE"],
}
runs = [
    {
        "seed": seed,
        "ablation": ablation,
        "extra_overrides": {},
        **split,
    }
    for ablation in ablations
    for seed in seeds
]
path = Path(f"runs/{os.environ['RUN_SLUG']}_matrix.json")
path.write_text(json.dumps(runs, indent=2, sort_keys=True), encoding="utf-8")
print(f"Wrote {len(runs)} planned runs to {path}")
PY

"${PYTHON_BIN}" -m data.bootstrap_paper_data

set_paper_ablation_flags() {
  local ablation="$1"
  export USE_IPM="false"
  export USE_BCM="false"
  export USE_DAM="false"

  case "${ablation}" in
    paper_baseline)
      ;;
    paper_ipm)
      export USE_IPM="true"
      ;;
    paper_dam)
      export USE_DAM="true"
      ;;
    paper_bcm)
      export USE_BCM="true"
      ;;
    paper_ipm_bcm)
      export USE_IPM="true"
      export USE_BCM="true"
      ;;
    paper_all)
      export USE_IPM="true"
      export USE_BCM="true"
      export USE_DAM="true"
      ;;
    *)
      echo "Unsupported paper-control ablation: ${ablation}" >&2
      exit 1
      ;;
  esac
}

for ablation in ${PAPER_CONTROL_ABLATIONS}; do
  set_paper_ablation_flags "${ablation}"
  export ABLATION="${ablation}"

  for seed in ${RUN_SEEDS}; do
    export SEED="${seed}"
    echo "Running paper-control ablation=${ABLATION} seed=${SEED}"
    "${PYTHON_BIN}" main.py
  done
done

"${PYTHON_BIN}" -m experiments.aggregate \
  --runs-dir "${RUNS_DIR}" \
  --manifest "${MANIFEST}" \
  --output "${SUMMARY}"

"${PYTHON_BIN}" - <<'PY'
import os

import pandas as pd

run_slug = os.environ["RUN_SLUG"]
summary = pd.read_csv(f"runs/{run_slug}_summary.csv")
columns = [
    "ablation",
    "seed",
    "best_val_step",
    "best_val_selection_score",
    "best_val_total_return",
    "best_val_sharpe_ratio",
    "best_val_max_drawdown",
    "best_val_average_cash",
    "best_val_average_turnover",
]
columns = [column for column in columns if column in summary.columns]
ranked = summary[columns].sort_values(["ablation", "best_val_selection_score"], ascending=[True, False])
print(ranked.to_string(index=False))
PY

for ablation in ${PAPER_CONTROL_ABLATIONS}; do
  set_paper_ablation_flags "${ablation}"
  export ABLATION="${ablation}"

  for seed in ${RUN_SEEDS}; do
    export SEED="${seed}"
    echo "Evaluating paper-control ablation=${ABLATION} seed=${SEED}"
    "${PYTHON_BIN}" -m evaluation.dashboard
    cp assets/dashboard_metrics.csv "runs/${RUN_SLUG}_dashboard_${ablation}_seed_${seed}.csv"
    cp assets/dashboard_cost_scenarios.csv "runs/${RUN_SLUG}_costs_${ablation}_seed_${seed}.csv"
  done
done

"${PYTHON_BIN}" - <<'PY'
import os
from pathlib import Path

import pandas as pd

run_slug = os.environ["RUN_SLUG"]
benchmark_name = os.environ["BENCHMARK_NAME"]
ablations = os.environ["PAPER_CONTROL_ABLATIONS"].split()
seeds = [int(seed) for seed in os.environ["RUN_SEEDS"].split()]


def strategy(metrics, name):
    matches = metrics.loc[metrics["Strategy"] == name]
    if matches.empty:
        raise ValueError(f"Missing strategy '{name}' in dashboard metrics")
    return matches.iloc[0].to_dict()


rows = []
for ablation in ablations:
    for seed in seeds:
        path = Path(f"runs/{run_slug}_dashboard_{ablation}_seed_{seed}.csv")
        metrics = pd.read_csv(path)
        rl = strategy(metrics, "RL Agent")
        benchmark = strategy(metrics, benchmark_name)
        crp = strategy(metrics, "CRP")
        equal_weight = strategy(metrics, "Equal Weight")
        buy_hold = strategy(metrics, "Buy & Hold EW")
        rows.append(
            {
                "ablation": ablation,
                "seed": seed,
                "rl_return": rl["Total Return"],
                "rl_sharpe": rl["Sharpe Ratio"],
                "rl_max_drawdown": rl["Max Drawdown"],
                "rl_final_value": rl["Final Value"],
                "rl_avg_turnover": rl["Average Turnover"],
                "rl_avg_cash": rl["Average Cash"],
                "return_gap_vs_benchmark": rl["Total Return"] - benchmark["Total Return"],
                "return_gap_vs_crp": rl["Total Return"] - crp["Total Return"],
                "return_gap_vs_equal_weight": rl["Total Return"] - equal_weight["Total Return"],
                "return_gap_vs_buy_hold_ew": rl["Total Return"] - buy_hold["Total Return"],
                "sharpe_gap_vs_benchmark": rl["Sharpe Ratio"] - benchmark["Sharpe Ratio"],
                "sharpe_gap_vs_crp": rl["Sharpe Ratio"] - crp["Sharpe Ratio"],
                "sharpe_gap_vs_equal_weight": rl["Sharpe Ratio"] - equal_weight["Sharpe Ratio"],
                "sharpe_gap_vs_buy_hold_ew": rl["Sharpe Ratio"] - buy_hold["Sharpe Ratio"],
            }
        )

summary = pd.DataFrame(rows)
summary.to_csv(f"runs/{run_slug}_test_summary.csv", index=False)
print(summary.round(4).to_string(index=False))

grouped = summary.groupby("ablation", as_index=False).agg(
    rl_return_mean=("rl_return", "mean"),
    rl_return_std=("rl_return", "std"),
    rl_sharpe_mean=("rl_sharpe", "mean"),
    rl_sharpe_std=("rl_sharpe", "std"),
    max_drawdown_mean=("rl_max_drawdown", "mean"),
    return_gap_vs_crp_mean=("return_gap_vs_crp", "mean"),
    sharpe_gap_vs_crp_mean=("sharpe_gap_vs_crp", "mean"),
    return_gap_vs_equal_weight_mean=("return_gap_vs_equal_weight", "mean"),
    sharpe_gap_vs_equal_weight_mean=("sharpe_gap_vs_equal_weight", "mean"),
)
grouped.to_csv(f"runs/{run_slug}_test_grouped_summary.csv", index=False)
print("\nGrouped test summary")
print(grouped.round(4).to_string(index=False))
PY

echo "Validation summary: ${SUMMARY}"
echo "Test summary: ${TEST_SUMMARY}"
echo "Grouped test summary: runs/${RUN_SLUG}_test_grouped_summary.csv"
