import copy
import random

import numpy as np
import pandas as pd
import torch

from agent.ddpg_agent import DDPGAgent
from agent.replay_buffer import ReplayBuffer
from config.settings import config
from data.dam_gan import RGAN_Generator
from data.fetcher import fetch_data
from data.splits import validate_research_dates
from data.train_dam import train_gan
from data.universe import data_tickers, feature_assets, ipm_feature_dim
from env.portfolio_env import PortfolioEnv
from evaluation.baselines import run_baselines
from evaluation.metrics import FinancialMetrics
from models.ipm import IPM
from optimization.oracle import PortfolioOracle
from utils.checkpoints import checkpoint_dir, save_training_checkpoint
from utils.experiment_tracker import ExperimentTracker
from utils.logger import setup_logger

logger = setup_logger()


def _make_evaluation_ipm(ipm, device, update_ipm_online=True, cfg=config):
    if getattr(cfg, "USE_IPM", False) and update_ipm_online and getattr(cfg, "USE_ONLINE_IPM", False):
        return copy.deepcopy(ipm).to(device), True
    return ipm, False


def evaluate_agent(agent, ipm, df, return_trace=False, update_ipm_online=True, actor=None):
    env = PortfolioEnv(df, config)
    obs, _ = env.reset()
    done = False
    values = [env.portfolio_value]
    turnovers = []
    costs = []
    cash_weights = [1.0]
    trace_rows = []
    prev_weights = np.zeros(len(config.ASSETS) + 1)
    prev_weights[0] = 1.0

    actor_module = actor or agent.actor
    eval_ipm, cloned_ipm = _make_evaluation_ipm(ipm, agent.device, update_ipm_online, config)
    actor_was_training = actor_module.training
    ipm_was_training = eval_ipm.training
    actor_module.eval()
    eval_ipm.eval()
    ipm_optimizer = None
    if config.USE_IPM and update_ipm_online and getattr(config, "USE_ONLINE_IPM", False):
        ipm_optimizer = torch.optim.RMSprop(eval_ipm.parameters(), lr=config.IPM_ONLINE_LR)

    while not done:
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(agent.device)
        prev_w_tensor = torch.FloatTensor(prev_weights).unsqueeze(0).to(agent.device)

        with torch.no_grad():
            if config.USE_IPM:
                ipm_pred = eval_ipm(obs_tensor)
            else:
                ipm_pred = torch.zeros(1, ipm_feature_dim(config), device=agent.device)
            action = actor_module(obs_tensor, prev_w_tensor, ipm_pred).cpu().numpy()[0]

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
        turnovers.append(info["turnover"])
        costs.append(info["cost_rate"])
        cash_weights.append(info["weights"][0])
        if return_trace:
            row = {
                "date": pd.to_datetime(df.index[info["period_end_step"]]).normalize(),
                "portfolio_value": info["portfolio_value"],
                "turnover": info["turnover"],
                "cost_rate": info["cost_rate"],
                "raw_return": info["raw_return"],
                "net_return": info["net_return"],
                "rebalanced": info["rebalanced"],
            }
            for index, label in enumerate(["Cash"] + list(config.ASSETS)):
                row[f"weight_{label}"] = info["weights"][index]
                row[f"action_{label}"] = info["executed_action"][index]
            trace_rows.append(row)

    if actor_was_training:
        actor_module.train()
    if ipm_was_training and not cloned_ipm:
        eval_ipm.train()

    returns = np.diff(values) / values[:-1]
    metrics = FinancialMetrics.get_metrics(returns)
    metrics["Final Value"] = values[-1]
    metrics["Average Turnover"] = float(np.mean(turnovers)) if turnovers else 0.0
    metrics["Total Cost Rate"] = float(np.sum(costs)) if costs else 0.0
    metrics["Average Cash"] = float(np.mean(cash_weights)) if cash_weights else 0.0
    if return_trace:
        return metrics, pd.DataFrame(trace_rows)
    return metrics


def _active_share(action, num_assets):
    benchmark = np.zeros(num_assets + 1, dtype=float)
    benchmark[1:] = 1.0 / num_assets
    return 0.5 * float(np.sum(np.abs(np.asarray(action, dtype=float) - benchmark)))


def _benchmark_action(num_assets, policy_name):
    normalized_name = str(policy_name).strip().lower()
    if normalized_name in {"equal weight", "equal_weight", "ew", "buy & hold ew", "buy and hold ew", "buy_hold_ew"}:
        action = np.zeros(num_assets + 1, dtype=float)
        action[1:] = 1.0 / num_assets
        return action
    if normalized_name == "crp":
        return np.full(num_assets + 1, 1.0 / (num_assets + 1), dtype=float)
    raise ValueError(f"Unsupported TRAINING_BENCHMARK_POLICY: {policy_name}")


def selection_score(metrics, hurdle_metrics=None, cfg=config):
    metric_name = cfg.MODEL_SELECTION_METRIC
    if metric_name == "Benchmark Relative Score":
        if hurdle_metrics is None:
            raise ValueError("Benchmark Relative Score requires validation hurdle metrics")

        if "Sharpe Ratio" not in hurdle_metrics:
            if not hurdle_metrics:
                raise ValueError("Benchmark Relative Score requires at least one validation hurdle")
            return min(
                _single_hurdle_selection_score(metrics, hurdle_metric, cfg)
                for hurdle_metric in hurdle_metrics.values()
            )

        return _single_hurdle_selection_score(metrics, hurdle_metrics, cfg)

    if metric_name not in metrics:
        raise ValueError(f"Unknown model selection metric: {metric_name}")
    return metrics[metric_name]


def _single_hurdle_selection_score(metrics, hurdle_metrics, cfg=config):
    turnover_gap = metrics.get("Average Turnover", 0.0) - hurdle_metrics.get("Average Turnover", 0.0)
    return (
        metrics["Sharpe Ratio"] - hurdle_metrics["Sharpe Ratio"]
        + cfg.SELECTION_RETURN_WEIGHT * (metrics["Total Return"] - hurdle_metrics["Total Return"])
        + cfg.SELECTION_DRAWDOWN_WEIGHT * (metrics["Max Drawdown"] - hurdle_metrics["Max Drawdown"])
        - cfg.SELECTION_TURNOVER_WEIGHT * turnover_gap
    )


def selection_hurdle_names(cfg=config):
    hurdles = getattr(cfg, "MODEL_SELECTION_HURDLES", None)
    if hurdles:
        return list(hurdles)
    return [getattr(cfg, "MODEL_SELECTION_HURDLE", config.MODEL_SELECTION_HURDLE)]


def _validation_policy_candidates(agent, ipm, valid_df, validation_hurdle_metrics):
    candidates = []
    policy_specs = [("online", agent.actor)]
    if getattr(config, "USE_TARGET_POLICY_EVAL", True):
        policy_specs.append(("target_ema", agent.actor_target))

    for policy_source, actor in policy_specs:
        metrics, trace = evaluate_agent(agent, ipm, valid_df, return_trace=True, actor=actor)
        score = selection_score(metrics, validation_hurdle_metrics)
        metrics["Selection Score"] = score
        metrics["Policy Source"] = policy_source
        candidates.append(
            {
                "policy_source": policy_source,
                "actor": actor,
                "metrics": metrics,
                "trace": trace,
                "score": score,
            }
        )

    selected = max(candidates, key=lambda item: item["score"])
    for candidate in candidates:
        prefix = candidate["policy_source"]
        selected["metrics"][f"{prefix} Selection Score"] = candidate["score"]
        selected["metrics"][f"{prefix} Sharpe Ratio"] = candidate["metrics"]["Sharpe Ratio"]
        selected["metrics"][f"{prefix} Total Return"] = candidate["metrics"]["Total Return"]
        selected["metrics"][f"{prefix} Max Drawdown"] = candidate["metrics"]["Max Drawdown"]
    return selected, candidates


def _sample_episode_frame(train_df, gan=None, gan_device=torch.device("cpu")):
    required_rows = int(config.EPISODE_LENGTH) + int(config.WINDOW_SIZE) + 1
    if len(train_df) <= required_rows:
        episode_df = train_df.copy()
    else:
        start = np.random.randint(0, len(train_df) - required_rows)
        episode_df = train_df.iloc[start : start + required_rows].copy()

    if gan is None:
        return episode_df

    with torch.no_grad():
        noise = torch.randn(1, config.GAN_SEQ_LEN, config.GAN_NOISE_DIM, device=gan_device)
        synthetic_changes = gan(noise).squeeze(0).cpu().numpy()
    synthetic_changes = np.clip(synthetic_changes, -0.20, 0.20)
    synthetic_df = _synthetic_hlc_frame(episode_df, synthetic_changes)
    return pd.concat([episode_df, synthetic_df]).sort_index()


def _synthetic_hlc_frame(base_df, percentage_changes):
    columns = base_df.columns
    feature_list = feature_assets(config)
    last = base_df.iloc[-1].copy()
    rows = []
    dates = pd.bdate_range(base_df.index[-1], periods=len(percentage_changes) + 1)[1:]

    for date, changes in zip(dates, percentage_changes):
        row = last.copy()
        for asset_index, asset in enumerate(feature_list):
            offset = asset_index * 3
            high_change, low_change, close_change = changes[offset : offset + 3]
            previous_close = float(last[(asset, "Close")])
            close = max(previous_close * (1.0 + close_change), 1e-6)
            high = max(float(last[(asset, "High")]) * (1.0 + high_change), close)
            low = min(float(last[(asset, "Low")]) * (1.0 + low_change), close, high)

            row[(asset, "Open")] = previous_close
            row[(asset, "High")] = high
            row[(asset, "Low")] = max(low, 1e-6)
            row[(asset, "Close")] = close
            if (asset, "Volume") in columns:
                row[(asset, "Volume")] = float(last[(asset, "Volume")])

        rows.append(row)
        last = row

    return pd.DataFrame(rows, index=dates, columns=columns)


def train():
    validate_research_dates(config)
    random.seed(config.SEED)
    np.random.seed(config.SEED)
    torch.manual_seed(config.SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.SEED)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    run_checkpoint_dir = checkpoint_dir(config)
    run_checkpoint_dir.mkdir(parents=True, exist_ok=True)
    tracker = ExperimentTracker(config)
    tracker.write_manifest(config)

    if config.USE_DAM:
        dam_path = run_checkpoint_dir / "dam_generator.pth"
        if not dam_path.exists():
            train_gan(dam_path)

    logger.info(f"Fetching Training Data ({config.TRAIN_START_DATE} -> {config.TRAIN_END_DATE})")
    train_df = fetch_data(data_tickers(config), config.TRAIN_START_DATE, config.TRAIN_END_DATE)
    logger.info(f"Fetching Validation Data ({config.VALID_START_DATE} -> {config.VALID_END_DATE})")
    valid_df = fetch_data(data_tickers(config), config.VALID_START_DATE, config.VALID_END_DATE)
    hurdle_names = selection_hurdle_names(config)
    validation_baselines = run_baselines(valid_df, config, names=hurdle_names)
    validation_hurdle_metrics = {name: result["metrics"] for name, result in validation_baselines.items()}
    for name, hurdle_metrics in validation_hurdle_metrics.items():
        logger.info(
            f"Validation hurdle ({name}) | "
            f"Sharpe: {hurdle_metrics['Sharpe Ratio']:.4f} | "
            f"Return: {hurdle_metrics['Total Return']:.4f}"
        )

    agent = DDPGAgent(num_assets=len(config.ASSETS), window_size=config.WINDOW_SIZE)
    buffer = ReplayBuffer(capacity=config.BUFFER_SIZE, batch_size=config.BATCH_SIZE, device=agent.device, config=config)
    ipm = IPM(num_assets=len(config.ASSETS), window_size=config.WINDOW_SIZE).to(agent.device)
    oracle_cost_bps = config.TRADING_COST_BPS + getattr(config, "SLIPPAGE_BPS", 0.0)
    oracle = PortfolioOracle(num_assets=len(config.ASSETS) + 1, transaction_cost_bps=oracle_cost_bps)

    gan = None
    gan_device = torch.device("cpu")
    if config.USE_DAM:
        # Keep DAM generation on CPU. PyTorch MPS LSTM can fail with placeholder storage errors.
        gan = RGAN_Generator(
            config.GAN_NOISE_DIM,
            config.GAN_HIDDEN_DIM,
            config.GAN_SEQ_LEN,
            len(feature_assets(config)) * 3,
        ).to(gan_device)
        try:
            gan.load_state_dict(torch.load(run_checkpoint_dir / "dam_generator.pth", map_location=gan_device))
            gan.eval()
            logger.info("DAM Generator Loaded on CPU.")
        except Exception:
            gan = None

    if config.USE_IPM:
        IPM.pretrain_ipm(ipm, train_df, agent.device, epochs=config.IPM_PRETRAIN_EPOCHS)
        ipm_optimizer = (
            torch.optim.Adam(ipm.parameters(), lr=config.IPM_ONLINE_LR)
            if getattr(config, "USE_ONLINE_IPM", False)
            else None
        )
    else:
        ipm.eval()
        ipm_optimizer = None

    logger.info("Starting RL Training.")
    if getattr(config, "USE_ACTIVE_OVERLAY", False):
        logger.info(
            f"Active overlay enabled. Executed weights = "
            f"{config.ACTIVE_OVERLAY_BASE_WEIGHT:.2f} * {config.ACTIVE_OVERLAY_BASE_POLICY} + "
            f"{config.ACTIVE_OVERLAY_TILT_WEIGHT:.2f} * RL policy; "
            f"tracking penalty={config.ACTIVE_OVERLAY_TRACKING_PENALTY:.4f}."
        )
    if getattr(config, "USE_PARAMETER_NOISE", False):
        logger.info(
            f"Parameter-space exploration enabled. Initial std={config.PARAM_NOISE_INIT_STD:.4f}, "
            f"target action distance={config.PARAM_NOISE_TARGET_ACTION_STD:.4f}."
        )
    if config.USE_ARB:
        if config.USE_ADAPTIVE_ARB_ACTIVATION:
            logger.info(
                f"Shadow ARB enabled. Replay stays uniform until policy drift is <= "
                f"{config.ARB_POLICY_DRIFT_THRESHOLD:.3f} for {config.ARB_STABILITY_PATIENCE} checks "
                f"after episode {config.ARB_MIN_EPISODE}, portfolio value is >= "
                f"{config.ARB_MIN_PORTFOLIO_VALUE_RATIO:.2f}x initial capital, and turnover is <= "
                f"{config.ARB_MAX_ACTIVATION_TURNOVER:.2f}, and latest validation score is >= "
                f"{config.ARB_MIN_VALIDATION_SCORE:.2f}; then ramps to {config.ARB_MAX_MIX:.2f} "
                f"over {config.ARB_RAMP_EPISODES} episodes."
            )
        else:
            logger.info(
                f"Shadow ARB enabled. Replay mix ramps from 0.00 at episode {config.ARB_START_EPISODE} "
                f"to {config.ARB_MAX_MIX:.2f} at episode {config.ARB_FULL_EPISODE}."
            )
    best_score = -np.inf
    param_noise_scale = float(getattr(config, "PARAM_NOISE_INIT_STD", 0.0))
    validations_without_improvement = 0
    min_save_episode = int(getattr(config, "MIN_SAVE_EPISODE", 0))

    for episode in range(config.EPISODES):
        episode_number = episode + 1
        buffer.set_training_progress(episode_number)
        episode_df = _sample_episode_frame(train_df, gan=gan, gan_device=gan_device)
        env = PortfolioEnv(episode_df, config)
        obs, _ = env.reset()
        done = False
        prev_weights = np.zeros(len(config.ASSETS) + 1)
        prev_weights[0] = 1.0
        episode_reward = 0.0
        critic_losses = []
        rl_losses = []
        bcm_losses = []
        policy_entropies = []
        actor_grad_norms = []
        critic_grad_norms = []
        ipm_online_losses = []
        turnovers = []
        costs = []
        cash_weights = []
        concentrations = []
        drawdowns = []
        active_shares = []
        param_noise_states = []
        param_noise_prev_weights = []

        noise_scale = max(config.MIN_NOISE, config.INIT_NOISE * (1.0 - episode / config.EPISODES))
        action_noise_scale = 0.0 if getattr(config, "USE_PARAMETER_NOISE", False) else noise_scale
        oracle_weight = max(0.0, 1.0 - episode / config.ORACLE_ANNEAL_EPISODES)
        exploration_actor = (
            agent.create_perturbed_actor(param_noise_scale)
            if getattr(config, "USE_PARAMETER_NOISE", False)
            else agent.actor
        )

        while not done:
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(agent.device)
            prev_w_tensor = torch.FloatTensor(prev_weights).unsqueeze(0).to(agent.device)

            with torch.no_grad():
                if config.USE_IPM:
                    ipm_pred = ipm(obs_tensor)
                else:
                    ipm_pred = torch.zeros(1, ipm_feature_dim(config), device=agent.device)
                action = exploration_actor(obs_tensor, prev_w_tensor, ipm_pred).cpu().numpy()[0]

            if action_noise_scale > 0:
                noise = np.random.normal(0, action_noise_scale, size=action.shape)
                action = np.clip(action + noise, 0, 1)
                action /= action.sum() + 1e-8

            next_obs, reward, done, _, info = env.step(action)
            if getattr(config, "USE_PARAMETER_NOISE", False) and len(param_noise_states) < 256:
                param_noise_states.append(obs)
                param_noise_prev_weights.append(prev_weights)

            if config.USE_IPM and ipm_optimizer is not None:
                ipm_loss = IPM.online_update(
                    ipm,
                    ipm_optimizer,
                    obs,
                    env.get_ipm_target(info["period_start_step"]),
                    agent.device,
                )
                ipm_online_losses.append(ipm_loss)

            next_prev_weights = info["weights"]
            executed_action = info["executed_action"]
            turnovers.append(info["turnover"])
            costs.append(info["cost_rate"])
            cash_weights.append(info["weights"][0])
            concentrations.append(info["concentration"])
            drawdowns.append(info.get("drawdown", 0.0))
            active_shares.append(_active_share(executed_action, len(config.ASSETS)))

            if config.USE_BCM and info["rebalanced"]:
                benchmark_action = _benchmark_action(len(config.ASSETS), config.TRAINING_BENCHMARK_POLICY)
                opt_action = oracle.get_optimal_weights(
                    prev_weights,
                    info["price_relative_vector"],
                    max_weight=config.MAX_WEIGHT,
                    max_cash_weight=config.MAX_CASH_WEIGHT,
                )
                oracle_or_policy_action = oracle_weight * opt_action + (1.0 - oracle_weight) * executed_action
                greedy_action = oracle_weight * benchmark_action + (1.0 - oracle_weight) * oracle_or_policy_action
            else:
                greedy_action = executed_action
            greedy_action = env.transform_action_for_execution(greedy_action)

            buffer.add(obs, prev_weights, executed_action, reward, next_obs, next_prev_weights, greedy_action, done=done)

            if len(buffer) > config.BATCH_SIZE:
                batch = buffer.sample()
                critic_loss, rl_loss, bcm_loss = agent.update(batch, ipm)
                update_stats = getattr(agent, "last_update_stats", {})
                if update_stats.get("priority_indices") is not None:
                    buffer.update_priorities(update_stats["priority_indices"], update_stats["td_errors"])
                critic_losses.append(critic_loss)
                rl_losses.append(rl_loss)
                bcm_losses.append(bcm_loss)
                if update_stats:
                    policy_entropies.append(update_stats.get("normalized_policy_entropy", 0.0))
                    actor_grad_norms.append(update_stats.get("actor_grad_norm", 0.0))
                    critic_grad_norms.append(update_stats.get("critic_grad_norm", 0.0))

            obs = next_obs
            prev_weights = next_prev_weights
            episode_reward += reward

        average_turnover = float(np.mean(turnovers)) if turnovers else 0.0
        average_active_share = float(np.mean(active_shares)) if active_shares else 0.0
        buffer.observe_training_health(episode_reward, env.portfolio_value, average_turnover)
        arb_diagnostics = buffer.observe_policy(agent, ipm, episode_number)
        policy_drift = arb_diagnostics.get("policy_drift")
        policy_drift_text = "nan" if policy_drift is None else f"{policy_drift:.4f}"
        arb_stability_scale = arb_diagnostics.get("stability_multiplier", 0.0)
        arb_health = "ok" if arb_diagnostics.get("health_ok", True) else arb_diagnostics.get("health_reason", "bad")
        mean_policy_entropy = float(np.mean(policy_entropies)) if policy_entropies else 0.0
        mean_actor_grad_norm = float(np.mean(actor_grad_norms)) if actor_grad_norms else 0.0
        mean_critic_grad_norm = float(np.mean(critic_grad_norms)) if critic_grad_norms else 0.0
        mean_ipm_online_loss = float(np.mean(ipm_online_losses)) if ipm_online_losses else 0.0
        average_cash = float(np.mean(cash_weights)) if cash_weights else 0.0
        average_drawdown = float(np.mean(drawdowns)) if drawdowns else 0.0
        max_episode_drawdown = float(np.min(drawdowns)) if drawdowns else 0.0
        param_noise_distance = 0.0
        applied_param_noise_scale = param_noise_scale
        if getattr(config, "USE_PARAMETER_NOISE", False):
            param_noise_distance = agent.parameter_noise_distance(
                exploration_actor,
                np.asarray(param_noise_states, dtype=np.float32),
                np.asarray(param_noise_prev_weights, dtype=np.float32),
                ipm,
            )
            if param_noise_distance < config.PARAM_NOISE_TARGET_ACTION_STD:
                param_noise_scale *= config.PARAM_NOISE_ADAPT_RATE
            else:
                param_noise_scale /= config.PARAM_NOISE_ADAPT_RATE
            param_noise_scale = float(
                np.clip(param_noise_scale, config.PARAM_NOISE_MIN_STD, config.PARAM_NOISE_MAX_STD)
            )

        logger.info(
            f"Ep {episode_number:03d} | "
            f"Rew: {episode_reward:7.2f} | "
            f"Val: {env.portfolio_value:10.2f} | "
            f"Noise: {action_noise_scale:.3f} | "
            f"ParamNoise: {applied_param_noise_scale:.4f} | "
            f"Oracle: {oracle_weight:.2f} | "
            f"ARB: {buffer.current_arb_mix():.2f} | "
            f"ARBScale: {arb_stability_scale:.2f} | "
            f"Drift: {policy_drift_text} | "
            f"Health: {arb_health} | "
            f"Ent: {mean_policy_entropy:.3f} | "
            f"ActorGrad: {mean_actor_grad_norm:.3f} | "
            f"CriticGrad: {mean_critic_grad_norm:.3f} | "
            f"IPM: {mean_ipm_online_loss:.6f} | "
            f"Cash: {average_cash:.3f} | "
            f"ActShare: {average_active_share:.3f} | "
            f"DD: {max_episode_drawdown:.3f} | "
            f"Turn: {average_turnover:.3f}"
        )
        tracker.log_metrics(
            "train_episode",
            episode_number,
            {
                "episode_reward": episode_reward,
                "final_value": env.portfolio_value,
                "noise_scale": action_noise_scale,
                "parameter_noise_scale": applied_param_noise_scale,
                "parameter_noise_distance": param_noise_distance,
                "oracle_weight": oracle_weight,
                "arb_mix": buffer.current_arb_mix(),
                "arb_activation_episode": arb_diagnostics.get("activation_episode"),
                "arb_stable_count": arb_diagnostics.get("stable_count"),
                "arb_stability_multiplier": arb_stability_scale,
                "arb_health_ok": arb_diagnostics.get("health_ok", True),
                "arb_health_reason": arb_diagnostics.get("health_reason"),
                "arb_health_score": arb_diagnostics.get("health_score"),
                "policy_drift": policy_drift,
                "critic_loss": float(np.mean(critic_losses)) if critic_losses else 0.0,
                "actor_rl_loss": float(np.mean(rl_losses)) if rl_losses else 0.0,
                "bcm_loss": float(np.mean(bcm_losses)) if bcm_losses else 0.0,
                "policy_entropy": mean_policy_entropy,
                "actor_grad_norm": mean_actor_grad_norm,
                "critic_grad_norm": mean_critic_grad_norm,
                "ipm_online_loss": mean_ipm_online_loss,
                "average_turnover": average_turnover,
                "average_active_share": average_active_share,
                "total_cost_rate": float(np.sum(costs)) if costs else 0.0,
                "average_cash": average_cash,
                "average_concentration": float(np.mean(concentrations)) if concentrations else 0.0,
                "average_drawdown": average_drawdown,
                "max_drawdown": max_episode_drawdown,
            },
        )

        validation_metrics = None

        if episode_number % config.VALIDATION_FREQ == 0:
            selected_candidate, validation_candidates = _validation_policy_candidates(
                agent,
                ipm,
                valid_df,
                validation_hurdle_metrics,
            )
            validation_metrics = selected_candidate["metrics"]
            validation_trace = selected_candidate["trace"]
            score = selected_candidate["score"]
            candidate_text = " | ".join(
                f"{candidate['policy_source']}: {candidate['score']:.4f}"
                for candidate in validation_candidates
            )
            for name, hurdle_metrics in validation_hurdle_metrics.items():
                validation_metrics[f"{name} Sharpe Gap"] = (
                    validation_metrics["Sharpe Ratio"] - hurdle_metrics["Sharpe Ratio"]
                )
                validation_metrics[f"{name} Return Gap"] = (
                    validation_metrics["Total Return"] - hurdle_metrics["Total Return"]
                )
            buffer.observe_validation_health(score)
            weakest_hurdle = min(
                validation_hurdle_metrics,
                key=lambda name: _single_hurdle_selection_score(validation_metrics, validation_hurdle_metrics[name]),
            )
            logger.info(
                f"Validation {config.MODEL_SELECTION_METRIC}: {score:.4f} | "
                f"Policy: {validation_metrics['Policy Source']} | "
                f"Candidates: {candidate_text} | "
                f"Weakest Hurdle: {weakest_hurdle} | "
                f"Sharpe Gap: {validation_metrics[f'{weakest_hurdle} Sharpe Gap']:.4f} | "
                f"Return Gap: {validation_metrics[f'{weakest_hurdle} Return Gap']:.4f} | "
                f"Final Value: {validation_metrics['Final Value']:.2f}"
            )
            tracker.log_metrics("validation", episode_number, validation_metrics)
            tracker.log_table("validation_trace", episode_number, validation_trace)

            min_delta = float(getattr(config, "EARLY_STOPPING_MIN_DELTA", 0.0))
            can_save_checkpoint = episode_number >= min_save_episode
            if can_save_checkpoint and score > best_score + min_delta:
                best_score = score
                validations_without_improvement = 0
                save_training_checkpoint(
                    run_checkpoint_dir / "best.pt",
                    selected_candidate["actor"],
                    ipm,
                    config,
                    episode_number,
                    validation_metrics,
                    policy_source=selected_candidate["policy_source"],
                )
                logger.info(f"Best Model Saved: Validation {config.MODEL_SELECTION_METRIC}: {best_score:.4f}")
            else:
                if not can_save_checkpoint:
                    logger.info(
                        f"Validation checkpoint skipped before MIN_SAVE_EPISODE={min_save_episode}. "
                        f"Candidate {config.MODEL_SELECTION_METRIC}: {score:.4f}"
                    )
                else:
                    validations_without_improvement += 1
                    logger.info(
                        f"No validation improvement for {validations_without_improvement}/"
                        f"{config.EARLY_STOPPING_PATIENCE} checks. Best {config.MODEL_SELECTION_METRIC}: "
                        f"{best_score:.4f}"
                    )

            if can_save_checkpoint and validations_without_improvement >= config.EARLY_STOPPING_PATIENCE:
                logger.info(
                    f"Early stopping triggered at episode {episode_number}. "
                    f"Best {config.MODEL_SELECTION_METRIC}: {best_score:.4f}"
                )
                break

        if episode_number % config.CHECKPOINT_FREQ == 0:
            if validation_metrics is None:
                validation_metrics = evaluate_agent(agent, ipm, valid_df)

            save_training_checkpoint(
                run_checkpoint_dir / f"episode_{episode_number:03d}.pt",
                agent.actor,
                ipm,
                config,
                episode_number,
                validation_metrics,
            )


if __name__ == "__main__":
    train()
