from types import SimpleNamespace

import numpy as np
import pandas as pd

from env.portfolio_env import PortfolioEnv
from evaluation.metrics import FinancialMetrics
from optimization.oracle import PortfolioOracle


def _date_at(df, step):
    return pd.to_datetime([df.index[step]]).tz_localize(None).normalize()[0]


def _risky_action(weights):
    weights = np.asarray(weights, dtype=float)
    total = weights.sum()
    if total <= 1e-12:
        return None
    return np.insert(weights / total, 0, 0.0)


def _equal_weight(env):
    return np.array([0.0] + [1.0 / env.n_assets] * env.n_assets)


def _crp(env):
    return np.full(env.n_assets + 1, 1.0 / (env.n_assets + 1))


def _buy_and_hold(env):
    if env.current_weights[0] > 0.999:
        return _equal_weight(env)
    return env.current_weights.copy()


def _inverse_volatility(env):
    history = env._get_price_relative_history(env.current_step) - 1.0
    vol = np.std(history, axis=0, ddof=1)
    vol = np.where(vol <= 1e-8, np.nan, vol)
    inv_vol = 1.0 / vol
    inv_vol = np.nan_to_num(inv_vol, nan=0.0, posinf=0.0, neginf=0.0)
    action = _risky_action(inv_vol)
    return action if action is not None else _equal_weight(env)


def _momentum(env):
    history = env._get_price_relative_history(env.current_step)
    scores = np.prod(history, axis=0) - 1.0
    scores = np.clip(scores, 0.0, None)
    action = _risky_action(scores)
    if action is not None:
        return action
    cash = np.zeros(env.n_assets + 1)
    cash[0] = 1.0
    return cash


def _min_variance(env):
    history = env._get_price_relative_history(env.current_step) - 1.0
    cov = np.cov(history, rowvar=False)
    cov = np.atleast_2d(cov) + np.eye(env.n_assets) * 1e-6
    ones = np.ones(env.n_assets)
    try:
        inv_cov = np.linalg.pinv(cov)
        weights = inv_cov @ ones
        weights = np.clip(weights, 0.0, None)
    except np.linalg.LinAlgError:
        return _equal_weight(env)

    action = _risky_action(weights)
    return action if action is not None else _equal_weight(env)


def _mean_variance(env, config):
    history = env._get_price_relative_history(env.current_step)
    oracle = PortfolioOracle(env.n_assets + 1, config.TRADING_COST_BPS)
    return oracle.get_historical_weights(
        env.current_weights,
        history,
        max_weight=config.MAX_WEIGHT,
        max_cash_weight=getattr(config, "MAX_CASH_WEIGHT", None),
        risk_aversion=config.ORACLE_RISK_AVERSION,
    )


def _random_long_only(env, rng):
    weights = rng.dirichlet(np.ones(env.n_assets))
    return np.insert(weights, 0, 0.0)


def baseline_policies(config):
    rng = np.random.default_rng(getattr(config, "SEED", 42))
    return {
        "CRP": lambda env: _crp(env),
        "Equal Weight": lambda env: _equal_weight(env),
        "Buy & Hold EW": lambda env: _buy_and_hold(env),
        "Inverse Vol": lambda env: _inverse_volatility(env),
        "Min Variance": lambda env: _min_variance(env),
        "Mean-Variance": lambda env: _mean_variance(env, config),
        "Momentum": lambda env: _momentum(env),
        "Random Long-Only": lambda env: _random_long_only(env, rng),
    }


def _baseline_config(config):
    payload = config.model_dump() if hasattr(config, "model_dump") else dict(config.__dict__)
    payload["USE_ACTIVE_OVERLAY"] = False
    payload["ACTIVE_OVERLAY_TRACKING_PENALTY"] = 0.0
    return SimpleNamespace(**payload)


def run_policy(df, config, policy_fn):
    baseline_config = _baseline_config(config)
    env = PortfolioEnv(df, baseline_config)
    obs, _ = env.reset()
    done = False

    values = [env.portfolio_value]
    weights = [env.current_weights.copy()]
    dates = [_date_at(df, env.current_step)]
    turnovers = []
    costs = []

    while not done:
        action = policy_fn(env)
        obs, _, done, _, info = env.step(action)
        values.append(info["portfolio_value"])
        weights.append(info["weights"].copy())
        dates.append(_date_at(df, info["period_end_step"]))
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

    return {
        "values": values,
        "weights": weights,
        "metrics": metrics,
    }


def run_baselines(df, config, names=None):
    baseline_config = _baseline_config(config)
    policies = baseline_policies(baseline_config)
    selected_names = names or policies.keys()
    return {name: run_policy(df, baseline_config, policies[name]) for name in selected_names}
