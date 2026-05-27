import hashlib
import json
from pathlib import Path

import torch


COMPATIBILITY_KEYS = ("assets", "window_size", "action_dim", "feature_dim")


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
    return {
        "assets": list(config.ASSETS),
        "window_size": config.WINDOW_SIZE,
        "action_dim": len(config.ASSETS) + 1,
        "feature_dim": len(config.ASSETS) * 3,
        "benchmark_ticker": config.BENCHMARK_TICKER,
        "benchmark_name": config.BENCHMARK_NAME,
        "seed": getattr(config, "SEED", None),
        "use_ipm": getattr(config, "USE_IPM", None),
        "use_bcm": getattr(config, "USE_BCM", None),
        "use_dam": getattr(config, "USE_DAM", None),
        "use_arb": getattr(config, "USE_ARB", None),
        "use_sparse_network": getattr(config, "USE_SPARSE_NETWORK", None),
        "train_start_date": getattr(config, "TRAIN_START_DATE", None),
        "train_end_date": getattr(config, "TRAIN_END_DATE", None),
        "valid_start_date": getattr(config, "VALID_START_DATE", None),
        "valid_end_date": getattr(config, "VALID_END_DATE", None),
        "test_start_date": getattr(config, "TEST_START_DATE", None),
        "test_end_date": getattr(config, "TEST_END_DATE", None),
        "trading_cost_bps": getattr(config, "TRADING_COST_BPS", None),
        "slippage_bps": getattr(config, "SLIPPAGE_BPS", None),
        "spread_bps": getattr(config, "SPREAD_BPS", None),
        "market_impact_bps": getattr(config, "MARKET_IMPACT_BPS", None),
        "fixed_commission": getattr(config, "FIXED_COMMISSION", None),
        "rebalance_freq": getattr(config, "REBALANCE_FREQ", None),
        "reward_mode": getattr(config, "REWARD_MODE", None),
        "return_reward_scale": getattr(config, "RETURN_REWARD_SCALE", None),
        "turnover_penalty": getattr(config, "TURNOVER_PENALTY", None),
        "concentration_penalty": getattr(config, "CONCENTRATION_PENALTY", None),
        "max_weight": getattr(config, "MAX_WEIGHT", None),
        "max_cash_weight": getattr(config, "MAX_CASH_WEIGHT", None),
        "model_selection_metric": getattr(config, "MODEL_SELECTION_METRIC", None),
        "model_selection_hurdle": getattr(config, "MODEL_SELECTION_HURDLE", None),
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
    mismatches = {
        key: {"checkpoint": metadata.get(key), "expected": value}
        for key, value in expected.items()
        if key in COMPATIBILITY_KEYS
        if metadata.get(key) != value
    }
    if mismatches:
        raise RuntimeError(f"Checkpoint metadata does not match current config: {mismatches}")


def save_training_checkpoint(path, actor, ipm, config, episode, validation_metrics):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "metadata": checkpoint_metadata(config),
            "episode": episode,
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
    ipm.load_state_dict(checkpoint["ipm_state_dict"])
    return checkpoint
