from pathlib import Path

import numpy as np
import pandas as pd
import torch

from agent.ddpg_agent import DDPGAgent
from config.settings import config
from data.fetcher import fetch_data
from data.splits import validate_research_dates
from data.universe import data_tickers, ipm_feature_dim
from env.portfolio_env import PortfolioEnv
from evaluation.baselines import run_baselines
from evaluation.dashboard import _make_evaluation_ipm, _normalize_index, fetch_benchmark
from evaluation.metrics import FinancialMetrics
from models.ipm import IPM
from utils.checkpoints import load_training_checkpoint
from utils.logger import setup_logger

logger = setup_logger(name="ensemble_tearsheet")


def discover_checkpoints(model_dir=None):
    root = Path(model_dir or config.MODEL_DIR)
    return sorted(root.glob("*/best.pt"))


def load_members(checkpoint_paths):
    members = []
    for path in checkpoint_paths:
        agent = DDPGAgent(len(config.ASSETS), config.WINDOW_SIZE)
        ipm = IPM(len(config.ASSETS), config.WINDOW_SIZE).to(agent.device)
        checkpoint = load_training_checkpoint(path, agent, ipm, agent.device, config)
        agent.actor.eval()
        ipm.eval()
        members.append(
            {
                "path": Path(path),
                "agent": agent,
                "ipm": ipm,
                "policy_source": checkpoint.get("policy_source"),
                "validation_metrics": checkpoint.get("validation_metrics", {}),
                "validation_score": checkpoint.get("validation_metrics", {}).get("Selection Score", float("-inf")),
            }
        )
    return members


def select_ensemble_members(members, top_k=None, min_score=None, temperature=0.25, weighting="softmax"):
    if not members:
        raise ValueError("Cannot select ensemble members from an empty member list")

    ranked = sorted(members, key=lambda member: member.get("validation_score", float("-inf")), reverse=True)
    if min_score is not None:
        ranked = [member for member in ranked if member.get("validation_score", float("-inf")) >= min_score]
    top_k = int(top_k or 0)
    if top_k > 0:
        ranked = ranked[:top_k]
    if not ranked:
        ranked = sorted(members, key=lambda member: member.get("validation_score", float("-inf")), reverse=True)[:1]

    weighting = str(weighting or "softmax").lower()
    scores = np.asarray([member.get("validation_score", float("-inf")) for member in ranked], dtype=float)
    if weighting == "equal":
        weights = np.full(len(ranked), 1.0 / len(ranked))
    elif weighting not in {"softmax", "validation_softmax"}:
        raise ValueError(f"Unsupported ensemble weighting mode: {weighting}")
    elif len(ranked) == 1 or not np.isfinite(scores).all():
        weights = np.full(len(ranked), 1.0 / len(ranked))
    else:
        temperature = max(float(temperature), 1e-8)
        shifted = (scores - scores.max()) / temperature
        weights = np.exp(shifted)
        weights = weights / (weights.sum() + 1e-8)

    selected = []
    for member, weight in zip(ranked, weights):
        selected_member = dict(member)
        selected_member["ensemble_weight"] = float(weight)
        selected.append(selected_member)
    return selected


def ensemble_action(members, obs, prev_weights, eval_ipms=None):
    if not members:
        raise ValueError("Cannot compute an ensemble action without members")

    actions = []
    weights = []
    for index, member in enumerate(members):
        agent = member["agent"]
        ipm = eval_ipms[index] if eval_ipms is not None else member["ipm"]
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(agent.device)
        prev_w_tensor = torch.FloatTensor(prev_weights).unsqueeze(0).to(agent.device)
        with torch.no_grad():
            if config.USE_IPM:
                ipm_pred = ipm(obs_tensor)
            else:
                ipm_pred = torch.zeros(1, ipm_feature_dim(config), device=agent.device)
            actions.append(agent.actor(obs_tensor, prev_w_tensor, ipm_pred).cpu().numpy()[0])
            weights.append(member.get("ensemble_weight", 1.0))
    return _average_simplex_actions(actions, weights=weights)


def _average_simplex_actions(actions, weights=None):
    actions = np.asarray(actions, dtype=float)
    if weights is None:
        action = np.mean(actions, axis=0)
    else:
        weights = np.asarray(weights, dtype=float)
        weights = weights / (weights.sum() + 1e-8)
        action = np.sum(actions * weights[:, None], axis=0)
    action = np.clip(action, 0.0, 1.0)
    return action / (action.sum() + 1e-8)


def evaluate_ensemble(members, df):
    env = PortfolioEnv(df, config)
    obs, _ = env.reset()
    done = False
    prev_weights = np.zeros(len(config.ASSETS) + 1)
    prev_weights[0] = 1.0

    eval_ipms = []
    ipm_optimizers = []
    for member in members:
        eval_ipm, _ = _make_evaluation_ipm(member["ipm"], member["agent"].device, config)
        eval_ipm.eval()
        eval_ipms.append(eval_ipm)
        if config.USE_IPM and getattr(config, "USE_ONLINE_IPM", False):
            ipm_optimizers.append(torch.optim.RMSprop(eval_ipm.parameters(), lr=config.IPM_ONLINE_LR))
        else:
            ipm_optimizers.append(None)

    values = [env.portfolio_value]
    dates = [_normalize_index([df.index[env.current_step]])[0]]
    weights = [prev_weights]
    turnovers = []
    costs = []

    while not done:
        action = ensemble_action(members, obs, prev_weights, eval_ipms=eval_ipms)
        current_obs = obs
        obs, _, done, _, info = env.step(action)
        for index, optimizer in enumerate(ipm_optimizers):
            if optimizer is not None:
                IPM.online_update(
                    eval_ipms[index],
                    optimizer,
                    current_obs,
                    env.get_ipm_target(info["period_start_step"]),
                    members[index]["agent"].device,
                )
        prev_weights = info["weights"]
        values.append(info["portfolio_value"])
        weights.append(info["weights"])
        dates.append(_normalize_index([df.index[info["period_end_step"]]])[0])
        turnovers.append(info["turnover"])
        costs.append(info["cost_rate"])

    values = pd.Series(values, index=pd.DatetimeIndex(dates), name="value")
    weights = pd.DataFrame(weights, index=values.index, columns=["Cash"] + list(config.ASSETS))
    returns = values.pct_change().dropna()
    metrics = FinancialMetrics.get_metrics(returns)
    metrics["Final Value"] = float(values.iloc[-1])
    metrics["Average Turnover"] = float(np.mean(turnovers)) if turnovers else 0.0
    metrics["Total Cost Rate"] = float(np.sum(costs)) if costs else 0.0
    metrics["Average Cash"] = float(weights["Cash"].mean())
    return {"values": values, "weights": weights, "metrics": metrics}


def run_ensemble_tearsheet(checkpoint_paths=None):
    validate_research_dates(config)
    paths = [Path(path) for path in checkpoint_paths] if checkpoint_paths else discover_checkpoints(config.MODEL_DIR)
    if not paths:
        raise FileNotFoundError(f"No best.pt checkpoints found under {config.MODEL_DIR}")

    logger.info(f"Loading {len(paths)} ensemble checkpoints")
    for path in paths:
        logger.info(f"  {path}")

    df = fetch_data(data_tickers(config), config.TEST_START_DATE, config.TEST_END_DATE)
    benchmark_df = fetch_benchmark(config.TEST_START_DATE, config.TEST_END_DATE)
    members = load_members(paths)
    members = select_ensemble_members(
        members,
        top_k=getattr(config, "ENSEMBLE_TOP_K", 0),
        min_score=getattr(config, "ENSEMBLE_MIN_SELECTION_SCORE", None),
        temperature=getattr(config, "ENSEMBLE_TEMPERATURE", 0.25),
        weighting=getattr(config, "ENSEMBLE_WEIGHTING", "softmax"),
    )
    logger.info("Selected ensemble members")
    for member in members:
        logger.info(
            f"  weight={member['ensemble_weight']:.4f} | "
            f"score={member['validation_score']:.4f} | "
            f"policy={member.get('policy_source')} | "
            f"{member['path']}"
        )
    ensemble = evaluate_ensemble(members, df)

    baseline_results = run_baselines(df, config)
    values = pd.DataFrame(
        {
            "RL Ensemble": ensemble["values"],
            config.BENCHMARK_NAME: benchmark_df["Value"],
        }
    )
    for name, result in baseline_results.items():
        values[name] = result["values"]
    values = values.dropna(how="any")
    if len(values) < 2:
        raise ValueError("Not enough date-overlapping data to compute ensemble metrics")

    returns = values.pct_change().dropna()
    metrics_by_name = {name: FinancialMetrics.get_metrics(returns[name]) for name in values.columns}
    metrics_by_name["RL Ensemble"].update(ensemble["metrics"])
    for name, result in baseline_results.items():
        metrics_by_name[name]["Final Value"] = float(values[name].iloc[-1])
        metrics_by_name[name]["Average Turnover"] = result["metrics"].get("Average Turnover", 0.0)
        metrics_by_name[name]["Total Cost Rate"] = result["metrics"].get("Total Cost Rate", 0.0)
        metrics_by_name[name]["Average Cash"] = float(result["weights"]["Cash"].mean())
    metrics_by_name[config.BENCHMARK_NAME]["Final Value"] = float(values[config.BENCHMARK_NAME].iloc[-1])
    metrics_by_name[config.BENCHMARK_NAME]["Average Turnover"] = 0.0
    metrics_by_name[config.BENCHMARK_NAME]["Total Cost Rate"] = 0.0
    metrics_by_name[config.BENCHMARK_NAME]["Average Cash"] = 0.0

    benchmark_metrics = metrics_by_name[config.BENCHMARK_NAME]
    configured_hurdles = list(getattr(config, "MODEL_SELECTION_HURDLES", ["CRP"]))
    hurdle_names = []
    for name in [config.BENCHMARK_NAME] + configured_hurdles:
        if name in metrics_by_name and name not in hurdle_names:
            hurdle_names.append(name)

    rows = []
    for name, metrics in metrics_by_name.items():
        primary_hurdle = metrics_by_name["CRP"] if "CRP" in metrics_by_name else benchmark_metrics
        row = {
            "Strategy": name,
            "Total Return": metrics["Total Return"],
            "Excess Return vs Benchmark": metrics["Total Return"] - benchmark_metrics["Total Return"],
            "Excess Return vs CRP": metrics["Total Return"] - primary_hurdle["Total Return"],
            "Sharpe Ratio": metrics["Sharpe Ratio"],
            "Sharpe Gap vs Benchmark": metrics["Sharpe Ratio"] - benchmark_metrics["Sharpe Ratio"],
            "Sharpe Gap vs CRP": metrics["Sharpe Ratio"] - primary_hurdle["Sharpe Ratio"],
            "Max Drawdown": metrics["Max Drawdown"],
            "Final Value": metrics["Final Value"],
            "Average Turnover": metrics["Average Turnover"],
            "Total Cost Rate": metrics["Total Cost Rate"],
            "Average Cash": metrics["Average Cash"],
        }
        for hurdle in hurdle_names:
            row[f"Return Gap vs {hurdle}"] = metrics["Total Return"] - metrics_by_name[hurdle]["Total Return"]
            row[f"Sharpe Gap vs {hurdle}"] = metrics["Sharpe Ratio"] - metrics_by_name[hurdle]["Sharpe Ratio"]
        rows.append(row)

    summary = pd.DataFrame(rows).set_index("Strategy")
    output_path = Path("assets/ensemble_dashboard_metrics.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_path)
    member_path = Path("assets/ensemble_members.csv")
    pd.DataFrame(
        [
            {
                "checkpoint": str(member["path"]),
                "ensemble_weight": member["ensemble_weight"],
                "validation_score": member["validation_score"],
                "policy_source": member.get("policy_source"),
            }
            for member in members
        ]
    ).to_csv(member_path, index=False)
    logger.info("Ensemble metrics")
    logger.info("-" * 75)
    logger.info(summary.round(4).to_string())
    logger.info(f"Saved ensemble metrics to {output_path}")
    logger.info(f"Saved ensemble members to {member_path}")
    return summary


if __name__ == "__main__":
    run_ensemble_tearsheet()
