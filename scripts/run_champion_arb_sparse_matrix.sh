#!/usr/bin/env bash
set -euo pipefail

SMOKE="false"
EVALUATE_ONLY="false"
SKIP_TEST_EVAL="false"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --smoke)
      SMOKE="true"
      shift
      ;;
    --evaluate-only)
      EVALUATE_ONLY="true"
      shift
      ;;
    --skip-test-eval)
      SKIP_TEST_EVAL="true"
      shift
      ;;
    *)
      echo "Usage: $0 [--smoke] [--evaluate-only] [--skip-test-eval]" >&2
      exit 2
      ;;
  esac
done

PYTHON_BIN="${PYTHON_BIN:-python}"
DEFAULT_RUN_SLUG="champion_arb_sparse_matrix"
if [[ "${SMOKE}" == "true" ]]; then
  DEFAULT_RUN_SLUG="champion_arb_sparse_smoke"
fi
RUN_SLUG="${RUN_SLUG:-${DEFAULT_RUN_SLUG}}"
RUN_SEEDS="${RUN_SEEDS:-123}"
CHAMPION_VARIANTS="${CHAMPION_VARIANTS:-champion_base champion_arb champion_sparse champion_arb_sparse}"

if [[ "${SMOKE}" == "true" ]]; then
  export CHAMPION_EPISODES="${CHAMPION_EPISODES:-30}"
  export CHAMPION_EPISODE_LENGTH="${CHAMPION_EPISODE_LENGTH:-260}"
  export CHAMPION_BATCH_SIZE="${CHAMPION_BATCH_SIZE:-64}"
  export CHAMPION_BUFFER_SIZE="${CHAMPION_BUFFER_SIZE:-500}"
  export CHAMPION_IPM_PRETRAIN_EPOCHS="${CHAMPION_IPM_PRETRAIN_EPOCHS:-10}"
  export CHAMPION_GAN_EPOCHS="${CHAMPION_GAN_EPOCHS:-20}"
  export CHAMPION_MIN_SAVE_EPISODE="${CHAMPION_MIN_SAVE_EPISODE:-5}"
  export CHAMPION_EARLY_STOPPING_PATIENCE="${CHAMPION_EARLY_STOPPING_PATIENCE:-2}"
fi

MANIFEST="runs/${RUN_SLUG}_matrix.json"
SUMMARY="runs/${RUN_SLUG}_summary.csv"
TEST_SUMMARY="runs/${RUN_SLUG}_test_summary.csv"
RUNS_DIR="runs/${RUN_SLUG}_runs"

export RUN_SLUG
export RUN_SEEDS
export CHAMPION_VARIANTS

# Champion base: the current paper-control best configuration.
export ASSETS='["COST","CSCO","F","GS","AIG","CAT"]'
export USE_MARKET_FEATURE="true"
export MARKET_FEATURE_TICKER="^GSPC"
export BENCHMARK_TICKER="^GSPC"
export BENCHMARK_NAME="S&P 500"

export DATA_CACHE_DIR="${CHAMPION_DATA_CACHE_DIR:-data_cache/paper_control}"
export USE_DATA_CACHE="true"
export REFRESH_DATA_CACHE="${REFRESH_DATA_CACHE:-false}"
export REQUIRE_DATA_CACHE="true"
export EXPERIMENT_DIR="${RUNS_DIR}"
export MODEL_DIR="${CHAMPION_MODEL_DIR:-models/${RUN_SLUG}}"

export TRAIN_START_DATE="${CHAMPION_TRAIN_START_DATE:-2010-01-01}"
export TRAIN_END_DATE="${CHAMPION_TRAIN_END_DATE:-2023-12-31}"
export VALID_START_DATE="${CHAMPION_VALID_START_DATE:-2024-01-01}"
export VALID_END_DATE="${CHAMPION_VALID_END_DATE:-2024-12-31}"
export TEST_START_DATE="${CHAMPION_TEST_START_DATE:-2025-01-01}"
export TEST_END_DATE="${CHAMPION_TEST_END_DATE:-2026-05-27}"

export EPISODES="${CHAMPION_EPISODES:-200}"
export EPISODE_LENGTH="${CHAMPION_EPISODE_LENGTH:-650}"
export BATCH_SIZE="${CHAMPION_BATCH_SIZE:-128}"
export BUFFER_SIZE="${CHAMPION_BUFFER_SIZE:-1000}"
export IPM_PRETRAIN_EPOCHS="${CHAMPION_IPM_PRETRAIN_EPOCHS:-50}"
export GAN_EPOCHS="${CHAMPION_GAN_EPOCHS:-200}"
export CHECKPOINT_FREQ="${CHAMPION_CHECKPOINT_FREQ:-10}"
export VALIDATION_FREQ="${CHAMPION_VALIDATION_FREQ:-10}"
export MIN_SAVE_EPISODE="${CHAMPION_MIN_SAVE_EPISODE:-40}"
export EARLY_STOPPING_PATIENCE="${CHAMPION_EARLY_STOPPING_PATIENCE:-4}"
export EARLY_STOPPING_MIN_DELTA="${CHAMPION_EARLY_STOPPING_MIN_DELTA:-0.001}"

export REBALANCE_FREQ="1"
export REWARD_MODE="log_return"
export TRAINING_BENCHMARK_POLICY="CRP"
export RETURN_REWARD_SCALE="${CHAMPION_RETURN_REWARD_SCALE:-1000.0}"
export TRADING_COST_BPS="${CHAMPION_TRADING_COST_BPS:-20.0}"
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

variants = os.environ["CHAMPION_VARIANTS"].split()
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
        "ablation": variant,
        "extra_overrides": {},
        **split,
    }
    for variant in variants
    for seed in seeds
]
path = Path(f"runs/{os.environ['RUN_SLUG']}_matrix.json")
path.write_text(json.dumps(runs, indent=2, sort_keys=True), encoding="utf-8")
print(f"Wrote {len(runs)} planned runs to {path}")
PY

if [[ "${EVALUATE_ONLY}" != "true" ]]; then
  "${PYTHON_BIN}" -m data.bootstrap_paper_data
fi

set_default_arb_flags() {
  export USE_ADAPTIVE_ARB_ACTIVATION="${CHAMPION_DEFAULT_USE_ADAPTIVE_ARB_ACTIVATION:-true}"
  export ARB_MIN_EPISODE="${CHAMPION_DEFAULT_ARB_MIN_EPISODE:-20}"
  export ARB_STABILITY_PATIENCE="${CHAMPION_DEFAULT_ARB_STABILITY_PATIENCE:-3}"
  export ARB_POLICY_DRIFT_THRESHOLD="${CHAMPION_DEFAULT_ARB_POLICY_DRIFT_THRESHOLD:-0.075}"
  export ARB_INSTABILITY_DECAY="${CHAMPION_DEFAULT_ARB_INSTABILITY_DECAY:-0.50}"
  export ARB_STABILITY_RECOVERY="${CHAMPION_DEFAULT_ARB_STABILITY_RECOVERY:-0.20}"
  export ARB_MIN_PORTFOLIO_VALUE_RATIO="${CHAMPION_DEFAULT_ARB_MIN_PORTFOLIO_VALUE_RATIO:-0.90}"
  export ARB_MAX_ACTIVATION_TURNOVER="${CHAMPION_DEFAULT_ARB_MAX_ACTIVATION_TURNOVER:-0.18}"
  export ARB_MIN_VALIDATION_SCORE="${CHAMPION_DEFAULT_ARB_MIN_VALIDATION_SCORE:-0.0}"
  export ARB_PROBE_SIZE="${CHAMPION_DEFAULT_ARB_PROBE_SIZE:-256}"
  export ARB_RAMP_EPISODES="${CHAMPION_DEFAULT_ARB_RAMP_EPISODES:-30}"
  export ARB_CACHE_REFRESH_INTERVAL="${CHAMPION_DEFAULT_ARB_CACHE_REFRESH_INTERVAL:-256}"
  export ARB_START_EPISODE="${CHAMPION_DEFAULT_ARB_START_EPISODE:-30}"
  export ARB_FULL_EPISODE="${CHAMPION_DEFAULT_ARB_FULL_EPISODE:-80}"
  export ARB_MAX_MIX="${CHAMPION_DEFAULT_ARB_MAX_MIX:-0.80}"
  export ARB_TEMPERATURE="${CHAMPION_DEFAULT_ARB_TEMPERATURE:-0.25}"
  export ARB_MIN_PROBABILITY="${CHAMPION_DEFAULT_ARB_MIN_PROBABILITY:-1e-4}"
  export ARB_RECENCY_TAU="${CHAMPION_DEFAULT_ARB_RECENCY_TAU:-5000.0}"
  export ARB_REWARD_WEIGHT="${CHAMPION_DEFAULT_ARB_REWARD_WEIGHT:-0.35}"
  export ARB_UNCERTAINTY_WEIGHT="${CHAMPION_DEFAULT_ARB_UNCERTAINTY_WEIGHT:-0.25}"
  export ARB_ON_POLICY_WEIGHT="${CHAMPION_DEFAULT_ARB_ON_POLICY_WEIGHT:-0.25}"
  export ARB_RECENCY_WEIGHT="${CHAMPION_DEFAULT_ARB_RECENCY_WEIGHT:-0.15}"
}

set_conservative_arb_flags() {
  export USE_ADAPTIVE_ARB_ACTIVATION="true"
  export ARB_MIN_EPISODE="${CHAMPION_ARB_MIN_EPISODE:-40}"
  export ARB_STABILITY_PATIENCE="${CHAMPION_ARB_STABILITY_PATIENCE:-4}"
  export ARB_POLICY_DRIFT_THRESHOLD="${CHAMPION_ARB_POLICY_DRIFT_THRESHOLD:-0.05}"
  export ARB_INSTABILITY_DECAY="${CHAMPION_ARB_INSTABILITY_DECAY:-0.50}"
  export ARB_STABILITY_RECOVERY="${CHAMPION_ARB_STABILITY_RECOVERY:-0.20}"
  export ARB_MIN_PORTFOLIO_VALUE_RATIO="${CHAMPION_ARB_MIN_PORTFOLIO_VALUE_RATIO:-1.00}"
  export ARB_MAX_ACTIVATION_TURNOVER="${CHAMPION_ARB_MAX_ACTIVATION_TURNOVER:-0.05}"
  export ARB_MIN_VALIDATION_SCORE="${CHAMPION_ARB_MIN_VALIDATION_SCORE:-0.20}"
  export ARB_PROBE_SIZE="${CHAMPION_ARB_PROBE_SIZE:-256}"
  export ARB_RAMP_EPISODES="${CHAMPION_ARB_RAMP_EPISODES:-50}"
  export ARB_CACHE_REFRESH_INTERVAL="${CHAMPION_ARB_CACHE_REFRESH_INTERVAL:-256}"
  export ARB_START_EPISODE="${CHAMPION_ARB_START_EPISODE:-30}"
  export ARB_FULL_EPISODE="${CHAMPION_ARB_FULL_EPISODE:-80}"
  export ARB_MAX_MIX="${CHAMPION_ARB_MAX_MIX:-0.35}"
  export ARB_TEMPERATURE="${CHAMPION_ARB_TEMPERATURE:-0.50}"
  export ARB_MIN_PROBABILITY="${CHAMPION_ARB_MIN_PROBABILITY:-0.001}"
  export ARB_RECENCY_TAU="${CHAMPION_ARB_RECENCY_TAU:-1000.0}"
  export ARB_REWARD_WEIGHT="${CHAMPION_ARB_REWARD_WEIGHT:-0.25}"
  export ARB_UNCERTAINTY_WEIGHT="${CHAMPION_ARB_UNCERTAINTY_WEIGHT:-0.25}"
  export ARB_ON_POLICY_WEIGHT="${CHAMPION_ARB_ON_POLICY_WEIGHT:-0.30}"
  export ARB_RECENCY_WEIGHT="${CHAMPION_ARB_RECENCY_WEIGHT:-0.20}"
}

set_champion_variant_flags() {
  local variant="$1"

  export USE_IPM="true"
  export USE_BCM="true"
  export USE_DAM="true"
  export USE_ARB="false"
  export USE_SPARSE_NETWORK="false"
  export USE_PRIORITIZED_REPLAY="true"
  export DROPOUT="${CHAMPION_BASE_DROPOUT:-0.5}"
  export SPARSE_TOPOLOGY="${CHAMPION_SPARSE_TOPOLOGY:-ER}"
  export SPARSE_WIDTH_MULTIPLIER="${CHAMPION_SPARSE_WIDTH_MULTIPLIER:-3.0}"
  export SPARSE_DENSITY="${CHAMPION_SPARSE_DENSITY:-0.60}"
  set_default_arb_flags

  case "${variant}" in
    champion_base)
      ;;
    champion_arb)
      set_conservative_arb_flags
      export USE_ARB="true"
      export USE_PRIORITIZED_REPLAY="false"
      ;;
    champion_sparse)
      set_conservative_arb_flags
      export USE_SPARSE_NETWORK="true"
      export DROPOUT="${CHAMPION_SPARSE_DROPOUT:-0.25}"
      ;;
    champion_arb_sparse)
      set_conservative_arb_flags
      export USE_ARB="true"
      export USE_SPARSE_NETWORK="true"
      export USE_PRIORITIZED_REPLAY="false"
      export DROPOUT="${CHAMPION_SPARSE_DROPOUT:-0.25}"
      ;;
    *)
      echo "Unsupported champion variant: ${variant}" >&2
      exit 1
      ;;
  esac
}

if [[ "${EVALUATE_ONLY}" != "true" ]]; then
  for variant in ${CHAMPION_VARIANTS}; do
    set_champion_variant_flags "${variant}"
    export ABLATION="${variant}"

    for seed in ${RUN_SEEDS}; do
      export SEED="${seed}"
      echo "Running champion variant=${ABLATION} seed=${SEED}"
      "${PYTHON_BIN}" main.py
    done
  done
else
  echo "Skipping training because --evaluate-only was supplied."
fi

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
    "final_train_arb_mix",
    "final_train_arb_activation_episode",
]
columns = [column for column in columns if column in summary.columns]
ranked = summary[columns].sort_values(["ablation", "best_val_selection_score"], ascending=[True, False])
print(ranked.to_string(index=False))
PY

if [[ "${SKIP_TEST_EVAL}" != "true" ]]; then
  for variant in ${CHAMPION_VARIANTS}; do
    set_champion_variant_flags "${variant}"
    export ABLATION="${variant}"

    for seed in ${RUN_SEEDS}; do
      export SEED="${seed}"
      echo "Evaluating champion variant=${ABLATION} seed=${SEED}"
      "${PYTHON_BIN}" -m evaluation.dashboard
      cp assets/dashboard_metrics.csv "runs/${RUN_SLUG}_dashboard_${variant}_seed_${seed}.csv"
      cp assets/dashboard_cost_scenarios.csv "runs/${RUN_SLUG}_costs_${variant}_seed_${seed}.csv"
    done
  done

  "${PYTHON_BIN}" - <<'PY'
import os
from pathlib import Path

import pandas as pd

run_slug = os.environ["RUN_SLUG"]
benchmark_name = os.environ["BENCHMARK_NAME"]
variants = os.environ["CHAMPION_VARIANTS"].split()
seeds = [int(seed) for seed in os.environ["RUN_SEEDS"].split()]


def strategy(metrics, name):
    matches = metrics.loc[metrics["Strategy"] == name]
    if matches.empty:
        raise ValueError(f"Missing strategy '{name}' in dashboard metrics")
    return matches.iloc[0].to_dict()


rows = []
for variant in variants:
    for seed in seeds:
        path = Path(f"runs/{run_slug}_dashboard_{variant}_seed_{seed}.csv")
        metrics = pd.read_csv(path)
        rl = strategy(metrics, "RL Agent")
        benchmark = strategy(metrics, benchmark_name)
        crp = strategy(metrics, "CRP")
        equal_weight = strategy(metrics, "Equal Weight")
        buy_hold = strategy(metrics, "Buy & Hold EW")
        rows.append(
            {
                "ablation": variant,
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
    avg_turnover_mean=("rl_avg_turnover", "mean"),
    avg_cash_mean=("rl_avg_cash", "mean"),
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
    grouped[f"beats_{baseline}_return"] = grouped[f"return_gap_vs_{baseline}_mean"] > 0.0
    grouped[f"beats_{baseline}_sharpe"] = grouped[f"sharpe_gap_vs_{baseline}_mean"] > 0.0

grouped.to_csv(f"runs/{run_slug}_test_grouped_summary.csv", index=False)
print("\nGrouped test summary")
print(grouped.round(4).to_string(index=False))
PY

  echo "Test summary: ${TEST_SUMMARY}"
  echo "Grouped test summary: runs/${RUN_SLUG}_test_grouped_summary.csv"
else
  echo "Skipped test evaluation because --skip-test-eval was supplied."
fi

echo "Validation summary: ${SUMMARY}"
