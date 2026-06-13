import os
import copy
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from agent.ddpg_agent import DDPGAgent
from config.settings import config
from data.fetcher import fetch_close_series, fetch_data
from data.splits import validate_research_dates
from data.universe import data_tickers, ipm_feature_dim
from env.portfolio_env import PortfolioEnv
from evaluation.baselines import run_baselines
from evaluation.metrics import FinancialMetrics
from models.ipm import IPM
from utils.checkpoints import best_checkpoint_path, checkpoint_dir, load_training_checkpoint
from utils.logger import setup_logger

logger = setup_logger(name="tearsheet")


def _normalize_index(index):
    return pd.to_datetime(index).tz_localize(None).normalize()


def fetch_benchmark(start_date, end_date):
    logger.info(f"Fetching {config.BENCHMARK_NAME} Benchmark Data")
    close = fetch_close_series(config.BENCHMARK_TICKER, start_date, end_date)
    benchmark = pd.DataFrame(index=_normalize_index(close.index))
    benchmark["Return"] = close.pct_change().fillna(0).to_numpy()
    initial_capital = float(getattr(config, "INITIAL_CAPITAL", 10000.0))
    benchmark["Value"] = initial_capital * (1 + benchmark["Return"]).cumprod()
    return benchmark


def _zero_cost_config(base_config):
    payload = base_config.model_dump() if hasattr(base_config, "model_dump") else dict(base_config.__dict__)
    for key in ("TRADING_COST_BPS", "SLIPPAGE_BPS", "SPREAD_BPS", "MARKET_IMPACT_BPS", "FIXED_COMMISSION"):
        payload[key] = 0.0
    return SimpleNamespace(**payload)


def _make_evaluation_ipm(ipm, device, eval_config, update_ipm_online=True):
    if eval_config.USE_IPM and update_ipm_online and getattr(eval_config, "USE_ONLINE_IPM", False):
        return copy.deepcopy(ipm).to(device), True
    return ipm, False


def _evaluate_loaded_agent(df, eval_config, agent, ipm):
    env = PortfolioEnv(df, eval_config)
    obs, _ = env.reset()
    done = False
    prev_weights = np.zeros(len(eval_config.ASSETS) + 1)
    prev_weights[0] = 1.0

    values = [env.portfolio_value]
    weights = [prev_weights]
    dates = [_normalize_index([df.index[env.current_step]])[0]]
    turnovers = []
    costs = []
    actor_was_training = agent.actor.training
    eval_ipm, cloned_ipm = _make_evaluation_ipm(ipm, agent.device, eval_config)
    ipm_was_training = eval_ipm.training
    agent.actor.eval()
    eval_ipm.eval()
    ipm_optimizer = None
    if eval_config.USE_IPM and getattr(eval_config, "USE_ONLINE_IPM", False):
        ipm_optimizer = torch.optim.RMSprop(eval_ipm.parameters(), lr=eval_config.IPM_ONLINE_LR)

    while not done:
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(agent.device)
        prev_w_tensor = torch.FloatTensor(prev_weights).unsqueeze(0).to(agent.device)

        with torch.no_grad():
            if eval_config.USE_IPM:
                ipm_pred = eval_ipm(obs_tensor)
            else:
                ipm_pred = torch.zeros(1, ipm_feature_dim(eval_config), device=agent.device)
            action = agent.actor(obs_tensor, prev_w_tensor, ipm_pred).cpu().numpy()[0]

        current_obs = obs
        obs, _, done, _, info = env.step(action)
        if ipm_optimizer is not None:
            IPM.online_update(
                eval_ipm,
                ipm_optimizer,
                current_obs,
                env.get_ipm_target(info["period_start_step"]),
                agent.device,
            )
        prev_weights = info["weights"]
        values.append(info["portfolio_value"])
        weights.append(info["weights"])
        dates.append(_normalize_index([df.index[info["period_end_step"]]])[0])
        turnovers.append(info["turnover"])
        costs.append(info["cost_rate"])

    if actor_was_training:
        agent.actor.train()
    if ipm_was_training and not cloned_ipm:
        eval_ipm.train()

    values = pd.Series(values, index=pd.DatetimeIndex(dates), name="value")
    weights = pd.DataFrame(weights, index=values.index, columns=["Cash"] + list(eval_config.ASSETS))
    returns = values.pct_change().dropna()
    metrics = FinancialMetrics.get_metrics(returns)
    metrics["Final Value"] = float(values.iloc[-1])
    metrics["Average Turnover"] = float(np.mean(turnovers)) if turnovers else 0.0
    metrics["Total Cost Rate"] = float(np.sum(costs)) if costs else 0.0
    metrics["Average Cash"] = float(weights["Cash"].mean())

    return {
        "values": values,
        "weights": weights,
        "metrics": metrics,
    }


def run_tearsheet(checkpoint_path: str | None = None):
    checkpoint_path = Path(checkpoint_path or best_checkpoint_path(config))
    if not checkpoint_path.exists():
        available = sorted(Path(config.MODEL_DIR).glob("*/best.pt"))
        available_text = "\n".join(f"  - {path}" for path in available[-10:]) or "  - none"
        raise FileNotFoundError(
            f"No checkpoint exists for the current config at {checkpoint_path}.\n"
            f"Run `python main.py` first; it will write checkpoints under {checkpoint_dir(config)}.\n"
            "Existing best checkpoints are for older config hashes and may be metadata-incompatible:\n"
            f"{available_text}"
        )

    validate_research_dates(config)
    df = fetch_data(data_tickers(config), config.TEST_START_DATE, config.TEST_END_DATE)
    benchmark_df = fetch_benchmark(config.TEST_START_DATE, config.TEST_END_DATE)

    env = PortfolioEnv(df, config)
    agent = DDPGAgent(len(config.ASSETS), config.WINDOW_SIZE)
    ipm = IPM(len(config.ASSETS), config.WINDOW_SIZE).to(agent.device)

    logger.info(f"Loading checkpoint: {checkpoint_path}")
    load_training_checkpoint(checkpoint_path, agent, ipm, agent.device, config)

    agent.actor.eval()
    eval_ipm, cloned_ipm = _make_evaluation_ipm(ipm, agent.device, config)
    ipm_was_training = eval_ipm.training
    eval_ipm.eval()
    ipm_optimizer = None
    if config.USE_IPM and getattr(config, "USE_ONLINE_IPM", False):
        ipm_optimizer = torch.optim.RMSprop(eval_ipm.parameters(), lr=config.IPM_ONLINE_LR)

    obs, _ = env.reset()
    done = False
    prev_weights = np.zeros(len(config.ASSETS) + 1)
    prev_weights[0] = 1.0

    rl_values = [env.portfolio_value]
    rl_weights = [prev_weights]
    value_dates = [_normalize_index([df.index[env.current_step]])[0]]
    rl_turnovers = []
    rl_costs = []

    while not done:
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(agent.device)
        prev_w_tensor = torch.FloatTensor(prev_weights).unsqueeze(0).to(agent.device)

        with torch.no_grad():
            if config.USE_IPM:
                ipm_pred = eval_ipm(obs_tensor)
            else:
                ipm_pred = torch.zeros(1, ipm_feature_dim(config), device=agent.device)
            action = agent.actor(obs_tensor, prev_w_tensor, ipm_pred).cpu().numpy()[0]

        current_obs = obs
        next_obs, _, done, _, info = env.step(action)
        if ipm_optimizer is not None:
            IPM.online_update(
                eval_ipm,
                ipm_optimizer,
                current_obs,
                env.get_ipm_target(info["period_start_step"]),
                agent.device,
            )
        rl_values.append(info["portfolio_value"])
        rl_weights.append(info["weights"])
        value_dates.append(_normalize_index([df.index[info["period_end_step"]]])[0])
        rl_turnovers.append(info["turnover"])
        rl_costs.append(info["cost_rate"])

        obs = next_obs
        prev_weights = info["weights"]

    if ipm_was_training and not cloned_ipm:
        eval_ipm.train()

    value_index = pd.DatetimeIndex(value_dates)
    baseline_results = run_baselines(df, config)
    values = pd.DataFrame(
        {
            "RL Agent": rl_values,
            config.BENCHMARK_NAME: benchmark_df["Value"],
        },
        index=value_index,
    )
    for name, result in baseline_results.items():
        values[name] = result["values"]

    values[config.BENCHMARK_NAME] = benchmark_df["Value"]
    values = values.dropna(how="any")
    if len(values) < 2:
        raise ValueError("Not enough date-overlapping data to compute benchmark metrics")

    returns = values.pct_change().dropna()
    metrics_by_name = {name: FinancialMetrics.get_metrics(returns[name]) for name in values.columns}
    metrics_by_name["RL Agent"]["Final Value"] = float(values["RL Agent"].iloc[-1])
    metrics_by_name["RL Agent"]["Average Turnover"] = float(np.mean(rl_turnovers)) if rl_turnovers else 0.0
    metrics_by_name["RL Agent"]["Total Cost Rate"] = float(np.sum(rl_costs)) if rl_costs else 0.0
    metrics_by_name["RL Agent"]["Average Cash"] = float(np.mean(np.asarray(rl_weights)[:, 0]))

    for name, result in baseline_results.items():
        metrics_by_name[name]["Final Value"] = float(values[name].iloc[-1])
        metrics_by_name[name]["Average Turnover"] = result["metrics"].get("Average Turnover", 0.0)
        metrics_by_name[name]["Total Cost Rate"] = result["metrics"].get("Total Cost Rate", 0.0)
        metrics_by_name[name]["Average Cash"] = float(result["weights"]["Cash"].mean())

    metrics_by_name[config.BENCHMARK_NAME]["Final Value"] = float(values[config.BENCHMARK_NAME].iloc[-1])
    metrics_by_name[config.BENCHMARK_NAME]["Average Turnover"] = 0.0
    metrics_by_name[config.BENCHMARK_NAME]["Total Cost Rate"] = 0.0
    metrics_by_name[config.BENCHMARK_NAME]["Average Cash"] = 0.0

    logger.info("Metrics by strategy")
    logger.info("-" * 75)
    for name, strategy_metrics in metrics_by_name.items():
        logger.info(name)
        for key, value in strategy_metrics.items():
            logger.info(f"  {key:<20}: {value:.4f}")

    benchmark_metrics = metrics_by_name[config.BENCHMARK_NAME]
    configured_hurdles = list(getattr(config, "MODEL_SELECTION_HURDLES", ["CRP"]))
    hurdle_names = []
    for name in [config.BENCHMARK_NAME] + configured_hurdles:
        if name in metrics_by_name and name not in hurdle_names:
            hurdle_names.append(name)
    summary_rows = []
    for name, strategy_metrics in metrics_by_name.items():
        primary_hurdle = metrics_by_name["CRP"] if "CRP" in metrics_by_name else benchmark_metrics
        row = {
            "Strategy": name,
            "Total Return": strategy_metrics["Total Return"],
            "Excess Return vs Benchmark": strategy_metrics["Total Return"] - benchmark_metrics["Total Return"],
            "Excess Return vs CRP": strategy_metrics["Total Return"] - primary_hurdle["Total Return"],
            "Sharpe Ratio": strategy_metrics["Sharpe Ratio"],
            "Sharpe Gap vs Benchmark": strategy_metrics["Sharpe Ratio"] - benchmark_metrics["Sharpe Ratio"],
            "Sharpe Gap vs CRP": strategy_metrics["Sharpe Ratio"] - primary_hurdle["Sharpe Ratio"],
            "Max Drawdown": strategy_metrics["Max Drawdown"],
            "Final Value": strategy_metrics["Final Value"],
            "Average Turnover": strategy_metrics["Average Turnover"],
            "Total Cost Rate": strategy_metrics["Total Cost Rate"],
            "Average Cash": strategy_metrics["Average Cash"],
        }
        for hurdle in hurdle_names:
            row[f"Return Gap vs {hurdle}"] = strategy_metrics["Total Return"] - metrics_by_name[hurdle]["Total Return"]
            row[f"Sharpe Gap vs {hurdle}"] = strategy_metrics["Sharpe Ratio"] - metrics_by_name[hurdle]["Sharpe Ratio"]
        summary_rows.append(row)

    summary = pd.DataFrame(summary_rows).set_index("Strategy")
    logger.info("Benchmark-relative summary")
    logger.info("-" * 75)
    logger.info(summary.round(4).to_string())

    rl_metrics = metrics_by_name["RL Agent"]
    passed_benchmark = all(
        rl_metrics["Total Return"] > metrics_by_name[hurdle]["Total Return"]
        and rl_metrics["Sharpe Ratio"] > metrics_by_name[hurdle]["Sharpe Ratio"]
        and rl_metrics["Max Drawdown"] >= metrics_by_name[hurdle]["Max Drawdown"]
        for hurdle in hurdle_names
    )
    verdict = "PASS" if passed_benchmark else "FAIL"
    logger.info(f"Out-of-sample benchmark gate vs {', '.join(hurdle_names)}: {verdict}")

    zero_cost_config = _zero_cost_config(config)
    zero_cost_agent = _evaluate_loaded_agent(df, zero_cost_config, agent, ipm)
    zero_cost_baselines = run_baselines(df, zero_cost_config)
    scenario_rows = []
    for name, strategy_metrics in metrics_by_name.items():
        row = {"Scenario": "realistic_costs", "Strategy": name}
        row.update(strategy_metrics)
        scenario_rows.append(row)

    zero_metrics = {"RL Agent": zero_cost_agent["metrics"]}
    zero_metrics[config.BENCHMARK_NAME] = metrics_by_name[config.BENCHMARK_NAME]
    for name, result in zero_cost_baselines.items():
        zero_metrics[name] = result["metrics"]

    for name, strategy_metrics in zero_metrics.items():
        row = {"Scenario": "zero_costs", "Strategy": name}
        row.update(strategy_metrics)
        scenario_rows.append(row)

    os.makedirs("assets", exist_ok=True)
    cost_scenarios = pd.DataFrame(scenario_rows)
    cost_scenario_path = "assets/dashboard_cost_scenarios.csv"
    cost_scenarios.to_csv(cost_scenario_path, index=False)
    logger.info(f"Saved cost scenario table to {cost_scenario_path}")

    fig, axes = plt.subplots(3, 1, figsize=(12, 16))

    ax1 = axes[0]
    normalized_values = values / values.iloc[0]
    for column in normalized_values.columns:
        linewidth = 2 if column == "RL Agent" else 1.2
        ax1.plot(normalized_values.index, normalized_values[column], label=column, linewidth=linewidth, alpha=0.9)

    ax1.set_title(f"Normalized Wealth: Agent vs Benchmarks ({config.TEST_START_DATE} to {config.TEST_END_DATE})")
    ax1.set_ylabel("Value / Initial Value")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2 = axes[1]
    relative_hurdles = [name for name in configured_hurdles if name in values.columns]
    for hurdle in relative_hurdles:
        ax2.plot(values.index, values["RL Agent"] / values[hurdle], label=f"RL / {hurdle}", linewidth=1.8)
    ax2.axhline(1.0, color="black", linewidth=1.0, linestyle="--", alpha=0.6)
    ax2.set_title("Relative Wealth vs Hurdles")
    ax2.set_ylabel("Relative Wealth")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    ax3 = axes[2]
    w_array = np.array(rl_weights).T
    labels = ["Cash"] + config.ASSETS
    ax3.stackplot(value_index, w_array, labels=labels, alpha=0.8)
    ax3.set_title("Agent Asset Allocation Over Time")
    ax3.set_ylabel("Weight")
    ax3.set_xlabel("Trading Steps")
    ax3.legend(loc="upper left", bbox_to_anchor=(1, 1))

    plt.tight_layout()
    metrics_path = "assets/dashboard_metrics.csv"
    summary.to_csv(metrics_path)
    logger.info(f"Saved metrics table to {metrics_path}")

    output_path = "assets/dashboard_benchmark.png"
    plt.savefig(output_path)
    logger.info(f"Saved comparison analysis to {output_path}")


if __name__ == "__main__":
    run_tearsheet()
