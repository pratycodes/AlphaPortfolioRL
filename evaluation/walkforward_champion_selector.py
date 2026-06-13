import argparse
from pathlib import Path

import pandas as pd

from config.settings import config
from evaluation.ensemble_dashboard import (
    discover_checkpoints,
    load_members,
    run_ensemble_tearsheet,
)


FULL_ALLOCATION_WINDOWS = [
    {
        "window": "2025-01-01_to_2026-05-27",
        "model_dir": "models/nifty50_stable_policy_long_seed_check",
        "train_start": "2010-01-01",
        "train_end": "2023-12-31",
        "valid_start": "2024-01-01",
        "valid_end": "2024-12-31",
        "test_start": "2025-01-01",
        "test_end": "2026-05-27",
    },
    {
        "window": "2024-01-01_to_2024-12-31",
        "model_dir": "models/nifty50_stable_policy_walkforward_2024",
        "train_start": "2010-01-01",
        "train_end": "2022-12-31",
        "valid_start": "2023-01-01",
        "valid_end": "2023-12-31",
        "test_start": "2024-01-01",
        "test_end": "2024-12-31",
    },
]

ACTIVE_OVERLAY_WINDOWS = [
    {
        "window": "2025-01-01_to_2026-05-27",
        "model_dir": "models/nifty50_active_overlay_long_seed_check",
        "train_start": "2010-01-01",
        "train_end": "2023-12-31",
        "valid_start": "2024-01-01",
        "valid_end": "2024-12-31",
        "test_start": "2025-01-01",
        "test_end": "2026-05-27",
    },
    {
        "window": "2024-01-01_to_2024-12-31",
        "model_dir": "models/nifty50_active_overlay_walkforward_2024",
        "train_start": "2010-01-01",
        "train_end": "2022-12-31",
        "valid_start": "2023-01-01",
        "valid_end": "2023-12-31",
        "test_start": "2024-01-01",
        "test_end": "2024-12-31",
    },
]


PROTOCOLS = [
    {
        "protocol": "full_allocation",
        "windows": FULL_ALLOCATION_WINDOWS,
        "overrides": {
            "USE_ACTIVE_OVERLAY": False,
            "ACTIVE_OVERLAY_BASE_POLICY": "Equal Weight",
            "ACTIVE_OVERLAY_BASE_WEIGHT": 0.80,
            "ACTIVE_OVERLAY_TILT_WEIGHT": 0.20,
            "ACTIVE_OVERLAY_TRACKING_PENALTY": 0.0,
        },
    },
    {
        "protocol": "active_overlay_80_20",
        "windows": ACTIVE_OVERLAY_WINDOWS,
        "overrides": {
            "USE_ACTIVE_OVERLAY": True,
            "ACTIVE_OVERLAY_BASE_POLICY": "Equal Weight",
            "ACTIVE_OVERLAY_BASE_WEIGHT": 0.80,
            "ACTIVE_OVERLAY_TILT_WEIGHT": 0.20,
            "ACTIVE_OVERLAY_TRACKING_PENALTY": 0.05,
        },
    },
]


CANDIDATES = [
    {"candidate": "top1_validation", "top_k": 1, "weighting": "softmax"},
    {"candidate": "top2_validation_softmax", "top_k": 2, "weighting": "softmax"},
    {"candidate": "top3_validation_softmax", "top_k": 3, "weighting": "softmax"},
    {"candidate": "top2_equal_weight", "top_k": 2, "weighting": "equal"},
    {"candidate": "top3_equal_weight", "top_k": 3, "weighting": "equal"},
]


BASE_ENV = {
    "ASSETS": [
        "RELIANCE.NS",
        "HDFCBANK.NS",
        "ICICIBANK.NS",
        "INFY.NS",
        "TCS.NS",
        "BHARTIARTL.NS",
        "LT.NS",
        "ITC.NS",
        "SBIN.NS",
        "HINDUNILVR.NS",
    ],
    "MARKET_FEATURE_TICKER": "^NSEI",
    "BENCHMARK_TICKER": "^NSEI",
    "BENCHMARK_NAME": "NIFTY 50",
    "DATA_CACHE_DIR": "data_cache/nifty50_smoke",
    "EPISODES": 100,
    "EPISODE_LENGTH": 650,
    "BATCH_SIZE": 128,
    "BUFFER_SIZE": 1500,
    "IPM_PRETRAIN_EPOCHS": 50,
    "MIN_SAVE_EPISODE": 20,
    "EARLY_STOPPING_PATIENCE": 4,
    "VALIDATION_FREQ": 10,
    "USE_TARGET_POLICY_EVAL": True,
    "STABLE_POLICY_SELECTION_VERSION": 1,
    "ABLATION": "current_baseline",
    "USE_IPM": True,
    "USE_ONLINE_IPM": True,
    "USE_BCM": True,
    "USE_DAM": False,
    "USE_ARB": False,
    "USE_SPARSE_NETWORK": False,
    "USE_PRIORITIZED_REPLAY": True,
    "REBALANCE_FREQ": 5,
    "TRADING_COST_BPS": 5.0,
    "SLIPPAGE_BPS": 5.0,
    "SPREAD_BPS": 2.0,
    "MARKET_IMPACT_BPS": 0.0,
    "MAX_CASH_WEIGHT": 0.05,
    "CASH_PENALTY": 0.20,
    "DRAWDOWN_PENALTY": 1.0,
    "SELECTION_DRAWDOWN_WEIGHT": 1.0,
    "USE_ACTIVE_OVERLAY": False,
    "ACTIVE_OVERLAY_BASE_POLICY": "Equal Weight",
    "ACTIVE_OVERLAY_BASE_WEIGHT": 0.80,
    "ACTIVE_OVERLAY_TILT_WEIGHT": 0.20,
    "ACTIVE_OVERLAY_TRACKING_PENALTY": 0.0,
    "ENSEMBLE_MIN_SELECTION_SCORE": None,
    "ENSEMBLE_TEMPERATURE": 0.25,
}


def _set_config_values(values):
    for key, value in values.items():
        setattr(config, key, value)


def _metric(metrics, name, column):
    row = metrics.loc[metrics["Strategy"] == name]
    if row.empty:
        raise ValueError(f"Missing strategy {name} in ensemble metrics")
    return float(row.iloc[0][column])


def _candidate_row(protocol_name, window_name, candidate_name, metrics):
    rl_return = _metric(metrics, "RL Ensemble", "Total Return")
    rl_sharpe = _metric(metrics, "RL Ensemble", "Sharpe Ratio")
    nifty_return = _metric(metrics, "NIFTY 50", "Total Return")
    nifty_sharpe = _metric(metrics, "NIFTY 50", "Sharpe Ratio")
    equal_return = _metric(metrics, "Equal Weight", "Total Return")
    equal_sharpe = _metric(metrics, "Equal Weight", "Sharpe Ratio")
    crp_return = _metric(metrics, "CRP", "Total Return")
    crp_sharpe = _metric(metrics, "CRP", "Sharpe Ratio")
    buy_hold_return = _metric(metrics, "Buy & Hold EW", "Total Return")
    buy_hold_sharpe = _metric(metrics, "Buy & Hold EW", "Sharpe Ratio")
    return {
        "protocol": protocol_name,
        "window": window_name,
        "candidate": candidate_name,
        "rl_return": rl_return,
        "rl_sharpe": rl_sharpe,
        "rl_max_drawdown": _metric(metrics, "RL Ensemble", "Max Drawdown"),
        "rl_average_turnover": _metric(metrics, "RL Ensemble", "Average Turnover"),
        "rl_average_cash": _metric(metrics, "RL Ensemble", "Average Cash"),
        "nifty_return": nifty_return,
        "nifty_sharpe": nifty_sharpe,
        "equal_weight_return": equal_return,
        "equal_weight_sharpe": equal_sharpe,
        "crp_return": crp_return,
        "crp_sharpe": crp_sharpe,
        "buy_hold_return": buy_hold_return,
        "buy_hold_sharpe": buy_hold_sharpe,
        "return_gap_vs_nifty": rl_return - nifty_return,
        "sharpe_gap_vs_nifty": rl_sharpe - nifty_sharpe,
        "return_gap_vs_equal_weight": rl_return - equal_return,
        "sharpe_gap_vs_equal_weight": rl_sharpe - equal_sharpe,
        "return_gap_vs_crp": rl_return - crp_return,
        "sharpe_gap_vs_crp": rl_sharpe - crp_sharpe,
        "return_gap_vs_buy_hold_ew": rl_return - buy_hold_return,
        "sharpe_gap_vs_buy_hold_ew": rl_sharpe - buy_hold_sharpe,
    }


def summarize_candidates(candidate_metrics):
    candidate_metrics = candidate_metrics.copy()
    if "protocol" not in candidate_metrics.columns:
        candidate_metrics["protocol"] = "default"
    grouped = candidate_metrics.groupby(["protocol", "candidate"], sort=False)
    rows = []
    for (protocol, candidate), frame in grouped:
        pass_both = (
            (frame["return_gap_vs_nifty"] > 0.0)
            & (frame["sharpe_gap_vs_nifty"] > 0.0)
            & (frame["return_gap_vs_equal_weight"] > 0.0)
            & (frame["sharpe_gap_vs_equal_weight"] > 0.0)
        )
        rows.append(
            {
                "protocol": protocol,
                "candidate": candidate,
                "windows_evaluated": int(len(frame)),
                "windows_passing_nifty": int(
                    ((frame["return_gap_vs_nifty"] > 0.0) & (frame["sharpe_gap_vs_nifty"] > 0.0)).sum()
                ),
                "windows_passing_equal_weight": int(
                    (
                        (frame["return_gap_vs_equal_weight"] > 0.0)
                        & (frame["sharpe_gap_vs_equal_weight"] > 0.0)
                    ).sum()
                ),
                "windows_passing_both": int(pass_both.sum()),
                "min_return_gap_vs_nifty": float(frame["return_gap_vs_nifty"].min()),
                "mean_return_gap_vs_nifty": float(frame["return_gap_vs_nifty"].mean()),
                "min_sharpe_gap_vs_nifty": float(frame["sharpe_gap_vs_nifty"].min()),
                "mean_sharpe_gap_vs_nifty": float(frame["sharpe_gap_vs_nifty"].mean()),
                "min_return_gap_vs_equal_weight": float(frame["return_gap_vs_equal_weight"].min()),
                "mean_return_gap_vs_equal_weight": float(frame["return_gap_vs_equal_weight"].mean()),
                "min_sharpe_gap_vs_equal_weight": float(frame["sharpe_gap_vs_equal_weight"].min()),
                "mean_sharpe_gap_vs_equal_weight": float(frame["sharpe_gap_vs_equal_weight"].mean()),
                "mean_rl_return": float(frame["rl_return"].mean()),
                "mean_rl_sharpe": float(frame["rl_sharpe"].mean()),
                "worst_rl_drawdown": float(frame["rl_max_drawdown"].min()),
                "mean_turnover": float(frame["rl_average_turnover"].mean()),
                "mean_cash": float(frame["rl_average_cash"].mean()),
            }
        )
    summary = pd.DataFrame(rows)
    return rank_candidates(summary)


def rank_candidates(summary):
    ranked = summary.copy()
    ranked = ranked.sort_values(
        by=[
            "protocol",
            "windows_passing_both",
            "min_return_gap_vs_equal_weight",
            "min_sharpe_gap_vs_equal_weight",
            "min_return_gap_vs_nifty",
            "mean_return_gap_vs_equal_weight",
        ],
        ascending=[True, False, False, False, False, False],
    )
    ranked.insert(0, "rank", ranked.groupby("protocol").cumcount() + 1)
    ranked = ranked.reset_index(drop=True)
    ranked["selection_status"] = "not_robust"
    for protocol, frame in ranked.groupby("protocol"):
        first_index = frame.index[0]
        if ranked.loc[first_index, "windows_passing_both"] == ranked.loc[first_index, "windows_evaluated"]:
            ranked.loc[first_index, "selection_status"] = "robust_champion"
        else:
            ranked.loc[first_index, "selection_status"] = "best_available_not_robust"
    return ranked


def _selected_members_for_candidate(model_dir, candidate):
    members = load_members(discover_checkpoints(model_dir))
    from evaluation.ensemble_dashboard import select_ensemble_members

    selected = select_ensemble_members(
        members,
        top_k=candidate["top_k"],
        min_score=getattr(config, "ENSEMBLE_MIN_SELECTION_SCORE", None),
        temperature=getattr(config, "ENSEMBLE_TEMPERATURE", 0.25),
        weighting=candidate["weighting"],
    )
    rows = []
    for member in selected:
        rows.append(
            {
                "checkpoint": str(member["path"]),
                "ensemble_weight": member["ensemble_weight"],
                "validation_score": member["validation_score"],
                "policy_source": member.get("policy_source"),
            }
        )
    return rows


def run_selector(output_dir="champion"):
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    metric_rows = []
    member_rows = []

    for protocol in PROTOCOLS:
        missing = [window["model_dir"] for window in protocol["windows"] if not discover_checkpoints(window["model_dir"])]
        if missing:
            print(f"Skipping {protocol['protocol']} because checkpoints are missing: {', '.join(missing)}")
            continue

        _set_config_values(BASE_ENV)
        _set_config_values(protocol["overrides"])
        for window in protocol["windows"]:
            _set_config_values(
                {
                    "MODEL_DIR": window["model_dir"],
                    "TRAIN_START_DATE": window["train_start"],
                    "TRAIN_END_DATE": window["train_end"],
                    "VALID_START_DATE": window["valid_start"],
                    "VALID_END_DATE": window["valid_end"],
                    "TEST_START_DATE": window["test_start"],
                    "TEST_END_DATE": window["test_end"],
                }
            )
            for candidate in CANDIDATES:
                _set_config_values(
                    {
                        "ENSEMBLE_TOP_K": candidate["top_k"],
                        "ENSEMBLE_WEIGHTING": candidate["weighting"],
                    }
                )
                metrics = run_ensemble_tearsheet().reset_index()
                metric_rows.append(_candidate_row(protocol["protocol"], window["window"], candidate["candidate"], metrics))
                for member in _selected_members_for_candidate(window["model_dir"], candidate):
                    member_rows.append(
                        {
                            "protocol": protocol["protocol"],
                            "window": window["window"],
                            "candidate": candidate["candidate"],
                            **member,
                        }
                    )

    if not metric_rows:
        raise FileNotFoundError("No walk-forward checkpoint sets were available to evaluate")

    candidate_metrics = pd.DataFrame(metric_rows)
    candidate_summary = summarize_candidates(candidate_metrics)
    candidate_members = pd.DataFrame(member_rows)

    metrics_path = output_root / "walkforward_candidate_metrics.csv"
    summary_path = output_root / "walkforward_champion_summary.csv"
    members_path = output_root / "walkforward_candidate_members.csv"
    candidate_metrics.to_csv(metrics_path, index=False)
    candidate_summary.to_csv(summary_path, index=False)
    candidate_members.to_csv(members_path, index=False)

    print(candidate_summary.round(4).to_string(index=False))
    print(f"Wrote candidate metrics to {metrics_path}")
    print(f"Wrote candidate summary to {summary_path}")
    print(f"Wrote candidate members to {members_path}")
    return candidate_summary


def main():
    parser = argparse.ArgumentParser(description="Evaluate walk-forward champion ensemble-selection candidates.")
    parser.add_argument("--output-dir", default="champion")
    args = parser.parse_args()
    run_selector(output_dir=args.output_dir)


if __name__ == "__main__":
    main()
