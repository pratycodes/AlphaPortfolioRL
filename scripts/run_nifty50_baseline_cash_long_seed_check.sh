#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_SLUG="${RUN_SLUG:-nifty50_baseline_cash_long_seed_check}"
RUN_SEEDS="${RUN_SEEDS:-42 7 123}"

MANIFEST="runs/${RUN_SLUG}_matrix.json"
SUMMARY="runs/${RUN_SLUG}_summary.csv"
TEST_SUMMARY="runs/${RUN_SLUG}_test_summary.csv"
RUNS_DIR="runs/${RUN_SLUG}_runs"
export RUN_SLUG
export RUN_SEEDS

export ASSETS="${ASSETS:-[\"RELIANCE.NS\",\"HDFCBANK.NS\",\"ICICIBANK.NS\",\"INFY.NS\",\"TCS.NS\",\"BHARTIARTL.NS\",\"LT.NS\",\"ITC.NS\",\"SBIN.NS\",\"HINDUNILVR.NS\"]}"
export MARKET_FEATURE_TICKER="${MARKET_FEATURE_TICKER:-^NSEI}"
export BENCHMARK_TICKER="${BENCHMARK_TICKER:-^NSEI}"
export BENCHMARK_NAME="${BENCHMARK_NAME:-NIFTY 50}"

export DATA_CACHE_DIR="${DATA_CACHE_DIR:-data_cache/nifty50_smoke}"
export EXPERIMENT_DIR="${RUNS_DIR}"
export MODEL_DIR="${MODEL_DIR:-models/${RUN_SLUG}}"

export TRAIN_START_DATE="${TRAIN_START_DATE:-2010-01-01}"
export TRAIN_END_DATE="${TRAIN_END_DATE:-2023-12-31}"
export VALID_START_DATE="${VALID_START_DATE:-2024-01-01}"
export VALID_END_DATE="${VALID_END_DATE:-2024-12-31}"
export TEST_START_DATE="${TEST_START_DATE:-2025-01-01}"
export TEST_END_DATE="${TEST_END_DATE:-2026-05-27}"

export EPISODES="${EPISODES:-100}"
export EPISODE_LENGTH="${EPISODE_LENGTH:-650}"
export BATCH_SIZE="${BATCH_SIZE:-128}"
export BUFFER_SIZE="${BUFFER_SIZE:-1500}"
export IPM_PRETRAIN_EPOCHS="${IPM_PRETRAIN_EPOCHS:-50}"
export MIN_SAVE_EPISODE="${MIN_SAVE_EPISODE:-20}"
export EARLY_STOPPING_PATIENCE="${EARLY_STOPPING_PATIENCE:-4}"
export VALIDATION_FREQ="${VALIDATION_FREQ:-10}"

export ABLATION="${ABLATION:-current_baseline}"
export USE_IPM="${USE_IPM:-true}"
export USE_ONLINE_IPM="${USE_ONLINE_IPM:-true}"
export USE_BCM="${USE_BCM:-true}"
export USE_DAM="${USE_DAM:-false}"
export USE_ARB="${USE_ARB:-false}"
export USE_SPARSE_NETWORK="${USE_SPARSE_NETWORK:-false}"
export USE_PRIORITIZED_REPLAY="${USE_PRIORITIZED_REPLAY:-true}"

export REBALANCE_FREQ="${REBALANCE_FREQ:-5}"
export TRADING_COST_BPS="${TRADING_COST_BPS:-5}"
export SLIPPAGE_BPS="${SLIPPAGE_BPS:-5}"
export SPREAD_BPS="${SPREAD_BPS:-2}"
export MARKET_IMPACT_BPS="${MARKET_IMPACT_BPS:-0}"
export MAX_CASH_WEIGHT="${MAX_CASH_WEIGHT:-0.05}"
export CASH_PENALTY="${CASH_PENALTY:-0.20}"
export DRAWDOWN_PENALTY="${DRAWDOWN_PENALTY:-1.0}"
export SELECTION_DRAWDOWN_WEIGHT="${SELECTION_DRAWDOWN_WEIGHT:-1.0}"
export USE_ACTIVE_OVERLAY="${USE_ACTIVE_OVERLAY:-false}"
export ACTIVE_OVERLAY_BASE_POLICY="${ACTIVE_OVERLAY_BASE_POLICY:-Equal Weight}"
export ACTIVE_OVERLAY_BASE_WEIGHT="${ACTIVE_OVERLAY_BASE_WEIGHT:-0.80}"
export ACTIVE_OVERLAY_TILT_WEIGHT="${ACTIVE_OVERLAY_TILT_WEIGHT:-0.20}"
export ACTIVE_OVERLAY_TRACKING_PENALTY="${ACTIVE_OVERLAY_TRACKING_PENALTY:-0.05}"

mkdir -p runs "${RUNS_DIR}"

"${PYTHON_BIN}" - <<'PY'
import json
import os
from pathlib import Path

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
        "ablation": "current_baseline",
        "extra_overrides": {},
        **split,
    }
    for seed in seeds
]
path = Path(f"runs/{os.environ['RUN_SLUG']}_matrix.json")
path.write_text(json.dumps(runs, indent=2, sort_keys=True), encoding="utf-8")
print(f"Wrote {len(runs)} planned runs to {path}")
PY

"${PYTHON_BIN}" -m data.bootstrap_paper_data

for seed in ${RUN_SEEDS}; do
  export SEED="${seed}"
  "${PYTHON_BIN}" main.py
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
    "best_val_sharpe_ratio",
    "best_val_total_return",
    "best_val_max_drawdown",
    "best_val_average_cash",
    "best_val_average_turnover",
]
columns = [column for column in columns if column in summary.columns]
ranked = summary[columns].sort_values("best_val_selection_score", ascending=False)
print(ranked.to_string(index=False))
PY

for seed in ${RUN_SEEDS}; do
  export SEED="${seed}"
  "${PYTHON_BIN}" -m evaluation.dashboard
  cp assets/dashboard_metrics.csv "runs/${RUN_SLUG}_dashboard_seed_${seed}.csv"
  cp assets/dashboard_cost_scenarios.csv "runs/${RUN_SLUG}_costs_seed_${seed}.csv"
done

"${PYTHON_BIN}" - <<'PY'
import os
from pathlib import Path

import pandas as pd

run_slug = os.environ["RUN_SLUG"]
seeds = [int(seed) for seed in os.environ["RUN_SEEDS"].split()]
rows = []
for seed in seeds:
    path = Path(f"runs/{run_slug}_dashboard_seed_{seed}.csv")
    metrics = pd.read_csv(path)
    rl = metrics.loc[metrics["Strategy"] == "RL Agent"].iloc[0].to_dict()
    nifty = metrics.loc[metrics["Strategy"] == "NIFTY 50"].iloc[0].to_dict()
    crp = metrics.loc[metrics["Strategy"] == "CRP"].iloc[0].to_dict()
    equal_weight = metrics.loc[metrics["Strategy"] == "Equal Weight"].iloc[0].to_dict()
    buy_hold = metrics.loc[metrics["Strategy"] == "Buy & Hold EW"].iloc[0].to_dict()
    rows.append(
        {
            "seed": seed,
            "rl_return": rl["Total Return"],
            "rl_sharpe": rl["Sharpe Ratio"],
            "rl_max_drawdown": rl["Max Drawdown"],
            "rl_final_value": rl["Final Value"],
            "rl_avg_turnover": rl["Average Turnover"],
            "rl_avg_cash": rl["Average Cash"],
            "return_gap_vs_nifty": rl["Total Return"] - nifty["Total Return"],
            "return_gap_vs_crp": rl["Total Return"] - crp["Total Return"],
            "return_gap_vs_equal_weight": rl["Total Return"] - equal_weight["Total Return"],
            "return_gap_vs_buy_hold_ew": rl["Total Return"] - buy_hold["Total Return"],
            "sharpe_gap_vs_nifty": rl["Sharpe Ratio"] - nifty["Sharpe Ratio"],
            "sharpe_gap_vs_crp": rl["Sharpe Ratio"] - crp["Sharpe Ratio"],
            "sharpe_gap_vs_equal_weight": rl["Sharpe Ratio"] - equal_weight["Sharpe Ratio"],
            "sharpe_gap_vs_buy_hold_ew": rl["Sharpe Ratio"] - buy_hold["Sharpe Ratio"],
        }
    )

summary = pd.DataFrame(rows)
summary.to_csv(f"runs/{run_slug}_test_summary.csv", index=False)
print(summary.round(4).to_string(index=False))
PY
