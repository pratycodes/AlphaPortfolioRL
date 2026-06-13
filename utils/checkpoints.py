import hashlib
import json
from pathlib import Path

import torch

from data.universe import feature_dim


COMPATIBILITY_KEYS = (
    "assets",
    "window_size",
    "action_dim",
    "feature_dim",
    "use_market_feature",
    "require_data_cache",
    "initial_capital",
    "market_feature_ticker",
    "benchmark_ticker",
    "benchmark_name",
    "use_ipm",
    "use_bcm",
    "use_dam",
    "dam_loss_version",
    "use_online_ipm",
    "ipm_online_lr",
    "ipm_version",
    "use_parameter_noise",
    "param_noise_init_std",
    "param_noise_min_std",
    "param_noise_max_std",
    "param_noise_target_action_std",
    "param_noise_adapt_rate",
    "param_noise_version",
    "use_arb",
    "use_sparse_network",
    "use_prioritized_replay",
    "min_save_episode",
    "early_stopping_patience",
    "early_stopping_min_delta",
    "train_start_date",
    "train_end_date",
    "valid_start_date",
    "valid_end_date",
    "test_start_date",
    "test_end_date",
    "trading_cost_bps",
    "cost_model_version",
    "slippage_bps",
    "spread_bps",
    "market_impact_bps",
    "fixed_commission",
    "rebalance_freq",
    "reward_mode",
    "training_benchmark_policy",
    "return_reward_scale",
    "episode_length",
    "dropout",
    "gamma",
    "tau",
    "lr_actor",
    "lr_critic",
    "oracle_anneal_episodes",
    "bcm_lambda",
    "policy_entropy_coef",
    "max_grad_norm",
    "action_projection_version",
    "baseline_anchor_weight",
    "anchor_version",
    "bcm_target_version",
    "bcm_loss_version",
    "use_target_policy_eval",
    "stable_policy_selection_version",
    "gan_mmd_lambda",
    "turnover_penalty",
    "concentration_penalty",
    "drawdown_penalty",
    "cash_penalty",
    "use_active_overlay",
    "active_overlay_base_policy",
    "active_overlay_base_weights",
    "active_overlay_base_weight",
    "active_overlay_tilt_weight",
    "active_overlay_tracking_penalty",
    "max_weight",
    "max_cash_weight",
    "model_selection_metric",
    "model_selection_hurdle",
    "model_selection_hurdles",
    "use_adaptive_arb_activation",
    "arb_min_episode",
    "arb_stability_patience",
    "arb_policy_drift_threshold",
    "arb_instability_decay",
    "arb_stability_recovery",
    "arb_min_portfolio_value_ratio",
    "arb_max_activation_turnover",
    "arb_min_validation_score",
    "arb_probe_size",
    "arb_ramp_episodes",
    "arb_cache_refresh_interval",
    "arb_start_episode",
    "arb_full_episode",
    "arb_max_mix",
    "arb_temperature",
    "arb_min_probability",
    "arb_recency_tau",
    "arb_reward_weight",
    "arb_uncertainty_weight",
    "arb_on_policy_weight",
    "arb_recency_weight",
    "sparse_density",
    "sparse_width_multiplier",
    "sparse_topology",
    "priority_alpha",
    "priority_beta_start",
    "priority_beta_frames",
    "priority_epsilon",
)


def _to_builtin(value):
    if isinstance(value, dict):
        return {key: _to_builtin(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_builtin(item) for item in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    if hasattr(value, "tolist"):
        return _to_builtin(value.tolist())
    return value


def checkpoint_metadata(config):
    use_active_overlay = bool(getattr(config, "USE_ACTIVE_OVERLAY", False))
    return {
        "assets": list(config.ASSETS),
        "window_size": config.WINDOW_SIZE,
        "action_dim": len(config.ASSETS) + 1,
        "feature_dim": feature_dim(config),
        "use_market_feature": getattr(config, "USE_MARKET_FEATURE", None),
        "require_data_cache": getattr(config, "REQUIRE_DATA_CACHE", None),
        "initial_capital": getattr(config, "INITIAL_CAPITAL", None),
        "market_feature_ticker": getattr(config, "MARKET_FEATURE_TICKER", None),
        "benchmark_ticker": config.BENCHMARK_TICKER,
        "benchmark_name": config.BENCHMARK_NAME,
        "seed": getattr(config, "SEED", None),
        "use_ipm": getattr(config, "USE_IPM", None),
        "use_bcm": getattr(config, "USE_BCM", None),
        "use_dam": getattr(config, "USE_DAM", None),
        "dam_loss_version": getattr(config, "DAM_LOSS_VERSION", None),
        "use_online_ipm": getattr(config, "USE_ONLINE_IPM", None),
        "ipm_online_lr": getattr(config, "IPM_ONLINE_LR", None),
        "ipm_version": getattr(config, "IPM_VERSION", None),
        "use_parameter_noise": getattr(config, "USE_PARAMETER_NOISE", None),
        "param_noise_init_std": getattr(config, "PARAM_NOISE_INIT_STD", None),
        "param_noise_min_std": getattr(config, "PARAM_NOISE_MIN_STD", None),
        "param_noise_max_std": getattr(config, "PARAM_NOISE_MAX_STD", None),
        "param_noise_target_action_std": getattr(config, "PARAM_NOISE_TARGET_ACTION_STD", None),
        "param_noise_adapt_rate": getattr(config, "PARAM_NOISE_ADAPT_RATE", None),
        "param_noise_version": getattr(config, "PARAM_NOISE_VERSION", None),
        "use_arb": getattr(config, "USE_ARB", None),
        "use_sparse_network": getattr(config, "USE_SPARSE_NETWORK", None),
        "use_prioritized_replay": getattr(config, "USE_PRIORITIZED_REPLAY", None),
        "min_save_episode": getattr(config, "MIN_SAVE_EPISODE", None),
        "early_stopping_patience": getattr(config, "EARLY_STOPPING_PATIENCE", None),
        "early_stopping_min_delta": getattr(config, "EARLY_STOPPING_MIN_DELTA", None),
        "train_start_date": getattr(config, "TRAIN_START_DATE", None),
        "train_end_date": getattr(config, "TRAIN_END_DATE", None),
        "valid_start_date": getattr(config, "VALID_START_DATE", None),
        "valid_end_date": getattr(config, "VALID_END_DATE", None),
        "test_start_date": getattr(config, "TEST_START_DATE", None),
        "test_end_date": getattr(config, "TEST_END_DATE", None),
        "trading_cost_bps": getattr(config, "TRADING_COST_BPS", None),
        "cost_model_version": getattr(config, "COST_MODEL_VERSION", None),
        "slippage_bps": getattr(config, "SLIPPAGE_BPS", None),
        "spread_bps": getattr(config, "SPREAD_BPS", None),
        "market_impact_bps": getattr(config, "MARKET_IMPACT_BPS", None),
        "fixed_commission": getattr(config, "FIXED_COMMISSION", None),
        "rebalance_freq": getattr(config, "REBALANCE_FREQ", None),
        "reward_mode": getattr(config, "REWARD_MODE", None),
        "training_benchmark_policy": getattr(config, "TRAINING_BENCHMARK_POLICY", None),
        "return_reward_scale": getattr(config, "RETURN_REWARD_SCALE", None),
        "episode_length": getattr(config, "EPISODE_LENGTH", None),
        "dropout": getattr(config, "DROPOUT", None),
        "gamma": getattr(config, "GAMMA", None),
        "tau": getattr(config, "TAU", None),
        "lr_actor": getattr(config, "LR_ACTOR", None),
        "lr_critic": getattr(config, "LR_CRITIC", None),
        "oracle_anneal_episodes": getattr(config, "ORACLE_ANNEAL_EPISODES", None),
        "bcm_lambda": getattr(config, "BCM_LAMBDA", None),
        "policy_entropy_coef": getattr(config, "POLICY_ENTROPY_COEF", None),
        "max_grad_norm": getattr(config, "MAX_GRAD_NORM", None),
        "action_projection_version": getattr(config, "ACTION_PROJECTION_VERSION", None),
        "baseline_anchor_weight": getattr(config, "BASELINE_ANCHOR_WEIGHT", None),
        "anchor_version": getattr(config, "ANCHOR_VERSION", None),
        "bcm_target_version": getattr(config, "BCM_TARGET_VERSION", None),
        "bcm_loss_version": getattr(config, "BCM_LOSS_VERSION", None),
        "use_target_policy_eval": getattr(config, "USE_TARGET_POLICY_EVAL", None),
        "stable_policy_selection_version": getattr(config, "STABLE_POLICY_SELECTION_VERSION", None),
        "gan_mmd_lambda": getattr(config, "GAN_MMD_LAMBDA", None),
        "turnover_penalty": getattr(config, "TURNOVER_PENALTY", None),
        "concentration_penalty": getattr(config, "CONCENTRATION_PENALTY", None),
        "drawdown_penalty": getattr(config, "DRAWDOWN_PENALTY", None),
        "cash_penalty": getattr(config, "CASH_PENALTY", None),
        "use_active_overlay": use_active_overlay,
        "active_overlay_base_policy": (
            getattr(config, "ACTIVE_OVERLAY_BASE_POLICY", "Equal Weight") if use_active_overlay else "Equal Weight"
        ),
        "active_overlay_base_weights": (
            getattr(config, "ACTIVE_OVERLAY_BASE_WEIGHTS", None) if use_active_overlay else None
        ),
        "active_overlay_base_weight": (
            getattr(config, "ACTIVE_OVERLAY_BASE_WEIGHT", 0.80) if use_active_overlay else 0.80
        ),
        "active_overlay_tilt_weight": (
            getattr(config, "ACTIVE_OVERLAY_TILT_WEIGHT", 0.20) if use_active_overlay else 0.20
        ),
        "active_overlay_tracking_penalty": (
            getattr(config, "ACTIVE_OVERLAY_TRACKING_PENALTY", 0.0) if use_active_overlay else 0.0
        ),
        "max_weight": getattr(config, "MAX_WEIGHT", None),
        "max_cash_weight": getattr(config, "MAX_CASH_WEIGHT", None),
        "model_selection_metric": getattr(config, "MODEL_SELECTION_METRIC", None),
        "model_selection_hurdle": getattr(config, "MODEL_SELECTION_HURDLE", None),
        "model_selection_hurdles": list(getattr(config, "MODEL_SELECTION_HURDLES", [])),
        "use_adaptive_arb_activation": getattr(config, "USE_ADAPTIVE_ARB_ACTIVATION", None),
        "arb_min_episode": getattr(config, "ARB_MIN_EPISODE", None),
        "arb_stability_patience": getattr(config, "ARB_STABILITY_PATIENCE", None),
        "arb_policy_drift_threshold": getattr(config, "ARB_POLICY_DRIFT_THRESHOLD", None),
        "arb_instability_decay": getattr(config, "ARB_INSTABILITY_DECAY", None),
        "arb_stability_recovery": getattr(config, "ARB_STABILITY_RECOVERY", None),
        "arb_min_portfolio_value_ratio": getattr(config, "ARB_MIN_PORTFOLIO_VALUE_RATIO", None),
        "arb_max_activation_turnover": getattr(config, "ARB_MAX_ACTIVATION_TURNOVER", None),
        "arb_min_validation_score": getattr(config, "ARB_MIN_VALIDATION_SCORE", None),
        "arb_probe_size": getattr(config, "ARB_PROBE_SIZE", None),
        "arb_ramp_episodes": getattr(config, "ARB_RAMP_EPISODES", None),
        "arb_cache_refresh_interval": getattr(config, "ARB_CACHE_REFRESH_INTERVAL", None),
        "arb_start_episode": getattr(config, "ARB_START_EPISODE", None),
        "arb_full_episode": getattr(config, "ARB_FULL_EPISODE", None),
        "arb_max_mix": getattr(config, "ARB_MAX_MIX", None),
        "arb_temperature": getattr(config, "ARB_TEMPERATURE", None),
        "arb_min_probability": getattr(config, "ARB_MIN_PROBABILITY", None),
        "arb_recency_tau": getattr(config, "ARB_RECENCY_TAU", None),
        "arb_reward_weight": getattr(config, "ARB_REWARD_WEIGHT", None),
        "arb_uncertainty_weight": getattr(config, "ARB_UNCERTAINTY_WEIGHT", None),
        "arb_on_policy_weight": getattr(config, "ARB_ON_POLICY_WEIGHT", None),
        "arb_recency_weight": getattr(config, "ARB_RECENCY_WEIGHT", None),
        "sparse_density": getattr(config, "SPARSE_DENSITY", None),
        "sparse_width_multiplier": getattr(config, "SPARSE_WIDTH_MULTIPLIER", None),
        "sparse_topology": getattr(config, "SPARSE_TOPOLOGY", None),
        "priority_alpha": getattr(config, "PRIORITY_ALPHA", None),
        "priority_beta_start": getattr(config, "PRIORITY_BETA_START", None),
        "priority_beta_frames": getattr(config, "PRIORITY_BETA_FRAMES", None),
        "priority_epsilon": getattr(config, "PRIORITY_EPSILON", None),
    }


def universe_id(config):
    payload = checkpoint_metadata(config)
    encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha1(encoded).hexdigest()[:10]


def checkpoint_dir(config):
    return Path(config.MODEL_DIR) / universe_id(config)


def best_checkpoint_path(config):
    return checkpoint_dir(config) / "best.pt"


def validate_checkpoint_metadata(metadata, config):
    expected = checkpoint_metadata(config)
    metadata = dict(metadata)
    if "drawdown_penalty" not in metadata:
        metadata["drawdown_penalty"] = 0.0
    if "use_active_overlay" not in metadata:
        metadata["use_active_overlay"] = False
    if "active_overlay_base_policy" not in metadata:
        metadata["active_overlay_base_policy"] = "Equal Weight"
    if "active_overlay_base_weights" not in metadata:
        metadata["active_overlay_base_weights"] = None
    if "active_overlay_base_weight" not in metadata:
        metadata["active_overlay_base_weight"] = 0.80
    if "active_overlay_tilt_weight" not in metadata:
        metadata["active_overlay_tilt_weight"] = 0.20
    if "active_overlay_tracking_penalty" not in metadata:
        metadata["active_overlay_tracking_penalty"] = 0.0
    mismatches = {
        key: {"checkpoint": metadata.get(key), "expected": value}
        for key, value in expected.items()
        if key in COMPATIBILITY_KEYS
        if metadata.get(key) != value
    }
    if mismatches:
        raise RuntimeError(f"Checkpoint metadata does not match current config: {mismatches}")


def save_training_checkpoint(path, actor, ipm, config, episode, validation_metrics, policy_source=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "metadata": checkpoint_metadata(config),
            "episode": episode,
            "policy_source": policy_source,
            "validation_metrics": _to_builtin(validation_metrics),
            "actor_state_dict": actor.state_dict(),
            "ipm_state_dict": ipm.state_dict(),
        },
        path,
    )


def load_training_checkpoint(path, agent, ipm, device, config):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {path}")

    # Project checkpoints are generated locally and include Python metadata
    # alongside tensors. PyTorch 2.6 defaults to weights_only=True, which
    # rejects numpy scalar metrics saved by older runs.
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)
    if not isinstance(checkpoint, dict) or "metadata" not in checkpoint:
        raise RuntimeError(f"Checkpoint at {path} is an old raw state_dict without metadata. Retrain under the current config.")

    validate_checkpoint_metadata(checkpoint["metadata"], config)
    agent.actor.load_state_dict(checkpoint["actor_state_dict"])
    if hasattr(agent, "actor_target"):
        agent.actor_target.load_state_dict(agent.actor.state_dict())
    ipm.load_state_dict(checkpoint["ipm_state_dict"])
    return checkpoint
