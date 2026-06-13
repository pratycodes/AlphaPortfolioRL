import gymnasium as gym
import numpy as np
from gymnasium import spaces
from data.features import ohlc_feature_matrix
from data.universe import feature_assets, feature_dim
from utils.costs import bps_to_rate

class PortfolioEnv(gym.Env):
    """
    Features:
    - Transaction Costs (Trading & Slippage)
    - Weight Constraints (Max Weight)
    - Risk-Adjusted Reward (Rolling Sharpe Ratio)
    """
    def __init__(self, df, config):
        super(PortfolioEnv, self).__init__()
        self.df = df
        self.assets = list(config.ASSETS)
        self.feature_assets = feature_assets(config)
        self.market_ticker = getattr(config, "MARKET_FEATURE_TICKER", None)
        self.use_market_feature = bool(getattr(config, "USE_MARKET_FEATURE", False))
        self.n_assets = len(self.assets)
        self.feature_dim = feature_dim(config)
        self.window = config.WINDOW_SIZE
        self.action_space = spaces.Box(
            low=0, high=1, shape=(self.n_assets + 1,), dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(self.window, self.feature_dim),
            dtype=np.float32
        )
        
        # Configuration
        self.cost_rate = bps_to_rate(config.TRADING_COST_BPS)
        self.slippage_rate = bps_to_rate(getattr(config, "SLIPPAGE_BPS", 0.0))
        self.spread_rate = bps_to_rate(getattr(config, "SPREAD_BPS", 0.0))
        self.market_impact_rate = bps_to_rate(getattr(config, "MARKET_IMPACT_BPS", 0.0))
        self.fixed_commission = getattr(config, "FIXED_COMMISSION", 0.0)
        self.max_weight = config.MAX_WEIGHT
        self.max_cash_weight = getattr(config, "MAX_CASH_WEIGHT", None)
        self.rebalance_freq = max(1, int(getattr(config, "REBALANCE_FREQ", 1)))
        
        # Risk Parameters 
        self.reward_mode = getattr(config, "REWARD_MODE", "rolling_sharpe")
        self.training_benchmark_policy = getattr(config, "TRAINING_BENCHMARK_POLICY", "Equal Weight")
        self.return_reward_scale = getattr(config, "RETURN_REWARD_SCALE", 100.0)
        self.turnover_penalty = getattr(config, "TURNOVER_PENALTY", 0.0)
        self.concentration_penalty = getattr(config, "CONCENTRATION_PENALTY", 0.0)
        self.drawdown_penalty = getattr(config, "DRAWDOWN_PENALTY", 0.0)
        self.cash_penalty = getattr(config, "CASH_PENALTY", 0.0)
        self.use_active_overlay = bool(getattr(config, "USE_ACTIVE_OVERLAY", False))
        self.active_overlay_base_policy = getattr(config, "ACTIVE_OVERLAY_BASE_POLICY", "Equal Weight")
        self.active_overlay_base_weights = getattr(config, "ACTIVE_OVERLAY_BASE_WEIGHTS", None)
        self.active_overlay_base_weight = float(getattr(config, "ACTIVE_OVERLAY_BASE_WEIGHT", 0.80))
        self.active_overlay_tilt_weight = float(getattr(config, "ACTIVE_OVERLAY_TILT_WEIGHT", 0.20))
        self.active_overlay_tracking_penalty = float(getattr(config, "ACTIVE_OVERLAY_TRACKING_PENALTY", 0.0))
        self.sharpe_window = getattr(config, "SHARPE_WINDOW", 30)
        self.initial_capital = float(getattr(config, "INITIAL_CAPITAL", 10000.0))
        self.episode_length = getattr(config, "EPISODE_LENGTH", None)
        self.return_memory = []
        self.episode_end_step = None
        self._prepare_market_arrays()
        
        self.reset()

    def _prepare_market_arrays(self):
        self._close_prices = (
            self.df.xs("Close", level=1, axis=1)
            .loc[:, self.assets]
            .to_numpy(dtype=float)
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            risky_relatives = self._close_prices[1:] / self._close_prices[:-1]
            risky_relatives = np.nan_to_num(risky_relatives, nan=1.0, posinf=1.0, neginf=1.0)
        self._risky_price_relatives = risky_relatives
        self._price_relative_vectors = np.concatenate(
            [np.ones((len(risky_relatives), 1), dtype=float), risky_relatives],
            axis=1,
        )

        self._ipm_feature_values = ohlc_feature_matrix(self.df, assets=self.feature_assets)
        with np.errstate(divide="ignore", invalid="ignore"):
            self._ipm_targets = (self._ipm_feature_values[1:] - self._ipm_feature_values[:-1]) / (
                self._ipm_feature_values[:-1] + 1e-8
            )
            self._ipm_targets = np.nan_to_num(self._ipm_targets, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

        self._matrix_assets = list(self.feature_assets)
        all_tickers = self.df.columns.get_level_values(0)
        if self.use_market_feature and self.market_ticker and self.market_ticker in all_tickers:
            self._matrix_assets.append(self.market_ticker)

        risky_and_market = ohlc_feature_matrix(self.df, assets=self._matrix_assets)
        cash = np.ones((len(self.df), 3), dtype=float)
        self._observation_values = np.concatenate([cash, risky_and_market], axis=1)

        matrix_close = (
            self.df.xs("Close", level=1, axis=1)
            .loc[:, self._matrix_assets]
            .to_numpy(dtype=float)
        )
        close_with_cash = np.concatenate(
            [np.ones((len(self.df), 1), dtype=float), matrix_close],
            axis=1,
        )
        self._observation_denominators = np.repeat(close_with_cash, 3, axis=1)
        self._observation_cache = {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}
        min_step = self.window - 1
        max_step = len(self.df) - 2
        if max_step < min_step:
            raise ValueError("Not enough rows for the configured WINDOW_SIZE")

        random_start = bool(options.get("random_start", False))
        episode_length = options.get("episode_length", self.episode_length)
        if random_start and episode_length:
            latest_start = max(min_step, max_step - int(episode_length))
            self.current_step = int(self.np_random.integers(min_step, latest_start + 1))
            self.episode_end_step = min(self.current_step + int(episode_length), max_step)
        else:
            self.current_step = min_step
            self.episode_end_step = max_step

        self.current_weights = np.zeros(self.n_assets + 1)
        self.current_weights[0] = 1.0 
        self.benchmark_weights = self._initial_benchmark_weights()
        self.portfolio_value = self.initial_capital
        self.peak_portfolio_value = self.initial_capital
        self.current_drawdown = 0.0
        self.return_memory = []
        
        obs = self._get_observation(self.current_step)
        info = {
            "portfolio_value": self.portfolio_value,
            "weights": self.current_weights
        }
        return obs, info

    def step(self, action):
        period_start_step = self.current_step

        # 1. Enforce long-only simplex, rebalance cadence, and target caps.
        rebalanced = self._should_rebalance(period_start_step)
        requested_weights = self._normalize_action(action)
        overlay_base_weights = self._active_overlay_base_weights()
        overlay_tracking_error = 0.0
        if rebalanced:
            weights = self._apply_active_overlay(requested_weights, overlay_base_weights)
            overlay_tracking_error = self._tracking_error(weights, overlay_base_weights)
        else:
            weights = self.current_weights.copy()

        # 2. Transaction costs. Yu et al. charge risky-asset turnover only.
        turnover = np.sum(np.abs(weights[1:] - self.current_weights[1:]))
        cost_components = self._get_cost_components(turnover)
        cost = sum(cost_components.values())
        
        # 3. Price relatives 
        y = self._price_relative_vectors[period_start_step].copy()

        # 4. Portfolio update 
        raw_return = np.dot(weights, y)
        net_return = raw_return * (1.0 - cost)
        concentration = np.sum(weights[1:] ** 2)
        benchmark_period_return = self._benchmark_period_return(y)
        benchmark_log_ret = np.log(max(benchmark_period_return, 1e-8))
        if net_return <= 1e-8:
            done = True
            reward = -10.0 
            next_obs = self._get_observation(self.current_step)
            
            self.current_weights = np.zeros_like(self.current_weights)
            self.current_weights[0] = 1.0
            self.current_drawdown = -1.0
            
            return next_obs, reward, done, False, {
                "portfolio_value": 0.0,
                "weights": self.current_weights,
                "raw_return": -1.0,
                "net_return": -1.0,
                "benchmark_return": benchmark_period_return - 1.0,
                "benchmark_log_return": benchmark_log_ret,
                "benchmark_policy": self.training_benchmark_policy,
                "turnover": turnover,
                "cost_rate": cost,
                "cost_components": cost_components,
                "concentration": concentration,
                "drawdown": self.current_drawdown,
                "peak_portfolio_value": self.peak_portfolio_value,
                "price_relative_vector": y,
                "period_start_step": period_start_step,
                "period_end_step": period_start_step + 1,
                "rebalanced": rebalanced,
                "requested_weights": requested_weights,
                "executed_action": weights,
                "active_overlay_base_weights": overlay_base_weights,
                "active_overlay_tracking_error": overlay_tracking_error,
            }

        self.portfolio_value *= net_return
        self.peak_portfolio_value = max(self.peak_portfolio_value, self.portfolio_value)
        self.current_drawdown = (self.portfolio_value - self.peak_portfolio_value) / (
            self.peak_portfolio_value + 1e-12
        )

        self.current_weights = (weights * y) / (raw_return + 1e-8)
        self.current_weights /= self.current_weights.sum()

        # 5. Reward: net return minus explicit trading/concentration penalties.
        log_ret = np.log(net_return + 1e-8)
        reward_signal = log_ret
        if self.reward_mode == "benchmark_relative":
            reward_signal = log_ret - benchmark_log_ret
        benchmark_cash_weight = float(self.benchmark_weights[0])
        reward = self._get_reward(
            reward_signal,
            turnover,
            concentration,
            weights[0],
            benchmark_cash_weight,
            overlay_tracking_error,
        )

        self.benchmark_weights = self._next_benchmark_weights(y)

        #  6. Step 
        self.current_step += 1
        done = self.current_step >= self.episode_end_step

        next_obs = self._get_observation(self.current_step)

        info = {
            "portfolio_value": self.portfolio_value,
            "weights": self.current_weights,
            "raw_return": raw_return - 1.0,
            "net_return": net_return - 1.0,
            "benchmark_return": benchmark_period_return - 1.0,
            "benchmark_log_return": benchmark_log_ret,
            "benchmark_policy": self.training_benchmark_policy,
            "turnover": turnover,
            "cost_rate": cost,
            "cost_components": cost_components,
            "concentration": concentration,
            "drawdown": self.current_drawdown,
            "peak_portfolio_value": self.peak_portfolio_value,
            "price_relative_vector": y,
            "period_start_step": period_start_step,
            "period_end_step": period_start_step + 1,
            "rebalanced": rebalanced,
            "requested_weights": requested_weights,
            "executed_action": weights,
            "active_overlay_base_weights": overlay_base_weights,
            "active_overlay_tracking_error": overlay_tracking_error,
        }

        return next_obs, reward, done, False, info

    def _should_rebalance(self, step):
        return (step - (self.window - 1)) % self.rebalance_freq == 0

    def _get_cost_components(self, turnover):
        fixed_commission_rate = 0.0
        if turnover > 1e-12 and self.portfolio_value > 0:
            fixed_commission_rate = self.fixed_commission / self.portfolio_value

        return {
            "trading": turnover * self.cost_rate,
            "slippage": turnover * self.slippage_rate,
            "spread": turnover * self.spread_rate,
            "market_impact": (turnover ** 2) * self.market_impact_rate,
            "fixed_commission": fixed_commission_rate,
        }

    def _normalize_action(self, action):
        weights = np.nan_to_num(np.asarray(action, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
        weights = np.clip(weights, 0.0, None)

        if weights.shape != (self.n_assets + 1,):
            raise ValueError(f"Expected action shape {(self.n_assets + 1,)}, got {weights.shape}")

        weight_sum = weights.sum()
        if weight_sum <= 1e-12:
            return self._fallback_weights()

        weights = weights / weight_sum

        if self.max_weight is not None:
            clipped_assets = np.minimum(weights[1:], self.max_weight)
            clipped_mass = weights[1:].sum() - clipped_assets.sum()
            weights[1:] = clipped_assets
            weights[0] += clipped_mass

        if self.max_cash_weight is not None and weights[0] > self.max_cash_weight:
            excess_cash = weights[0] - self.max_cash_weight
            weights[0] = self.max_cash_weight
            weights[1:] = self._redistribute_to_assets(weights[1:], excess_cash)

        if not np.isclose(weights.sum(), 1.0, atol=1e-8):
            weights = weights / (weights.sum() + 1e-12)
        return weights

    def transform_action_for_execution(self, action):
        requested_weights = self._normalize_action(action)
        overlay_base_weights = self._active_overlay_base_weights()
        executed_weights = self._apply_active_overlay(requested_weights, overlay_base_weights)
        return executed_weights

    def _apply_active_overlay(self, requested_weights, base_weights=None):
        requested_weights = self._normalize_action(requested_weights)
        if not self.use_active_overlay:
            return requested_weights

        base_weights = self._active_overlay_base_weights() if base_weights is None else self._normalize_action(base_weights)
        base_weight = np.clip(self.active_overlay_base_weight, 0.0, 1.0)
        tilt_weight = np.clip(self.active_overlay_tilt_weight, 0.0, 1.0)
        if base_weight + tilt_weight <= 1e-12:
            base_weight, tilt_weight = 1.0, 0.0

        combined = (base_weight * base_weights) + (tilt_weight * requested_weights)
        return self._normalize_action(combined)

    def _active_overlay_base_weights(self):
        if not self.use_active_overlay:
            return np.zeros(self.n_assets + 1)
        normalized_name = str(self.active_overlay_base_policy).strip().lower()
        if normalized_name in {"benchmark proxy", "benchmark_proxy", "index proxy", "index_proxy"}:
            return self._custom_overlay_base_weights()
        return self._normalize_action(self._benchmark_target_weights(self.active_overlay_base_policy))

    def _tracking_error(self, weights, base_weights):
        if not self.use_active_overlay:
            return 0.0
        return 0.5 * float(np.sum(np.abs(np.asarray(weights, dtype=float) - np.asarray(base_weights, dtype=float))))

    def _custom_overlay_base_weights(self):
        if self.active_overlay_base_weights is None:
            raise ValueError("ACTIVE_OVERLAY_BASE_WEIGHTS is required for Benchmark Proxy overlay")

        weights = np.asarray(self.active_overlay_base_weights, dtype=float)
        if weights.shape == (self.n_assets,):
            weights = np.insert(weights, 0, 0.0)
        if weights.shape != (self.n_assets + 1,):
            raise ValueError(
                "ACTIVE_OVERLAY_BASE_WEIGHTS must have either "
                f"{self.n_assets} risky weights or {self.n_assets + 1} cash+risky weights"
            )
        return self._normalize_action(weights)

    def _fallback_weights(self):
        weights = np.zeros(self.n_assets + 1)
        if self.max_cash_weight is None:
            weights[0] = 1.0
            return weights

        weights[0] = min(self.max_cash_weight, 1.0)
        weights[1:] = self._redistribute_to_assets(weights[1:], 1.0 - weights[0])
        return weights

    def _redistribute_to_assets(self, asset_weights, mass):
        asset_weights = np.asarray(asset_weights, dtype=float).copy()
        remaining = float(mass)

        for _ in range(self.n_assets + 1):
            if remaining <= 1e-12:
                break

            if self.max_weight is None:
                preference = asset_weights.copy()
                if preference.sum() <= 1e-12:
                    preference = np.ones_like(asset_weights)
                asset_weights += remaining * preference / preference.sum()
                remaining = 0.0
                break

            room = np.maximum(self.max_weight - asset_weights, 0.0)
            eligible = room > 1e-12
            if not eligible.any():
                raise ValueError("Weight caps are infeasible; increase MAX_WEIGHT or MAX_CASH_WEIGHT")

            preference = asset_weights[eligible].copy()
            if preference.sum() <= 1e-12:
                preference = np.ones_like(preference)

            proposed = remaining * preference / preference.sum()
            increment = np.minimum(proposed, room[eligible])
            asset_weights[eligible] += increment
            remaining -= increment.sum()

        if remaining > 1e-8:
            raise ValueError("Weight caps are infeasible; increase MAX_WEIGHT or MAX_CASH_WEIGHT")
        return asset_weights

    def _initial_benchmark_weights(self):
        return self._benchmark_target_weights(self.training_benchmark_policy)

    def _benchmark_target_weights(self, name):
        normalized_name = str(name).strip().lower()
        if normalized_name in {"equal weight", "equal_weight", "ew"}:
            weights = np.zeros(self.n_assets + 1)
            weights[1:] = 1.0 / self.n_assets
            return weights
        if normalized_name == "crp":
            return np.full(self.n_assets + 1, 1.0 / (self.n_assets + 1))
        if normalized_name in {"buy & hold ew", "buy and hold ew", "buy_hold_ew", "bh ew"}:
            weights = np.zeros(self.n_assets + 1)
            weights[1:] = 1.0 / self.n_assets
            return weights
        raise ValueError(f"Unsupported TRAINING_BENCHMARK_POLICY: {name}")

    def _benchmark_period_return(self, price_relative_vector):
        return float(np.dot(self.benchmark_weights, price_relative_vector))

    def _next_benchmark_weights(self, price_relative_vector):
        normalized_name = str(self.training_benchmark_policy).strip().lower()
        if normalized_name in {"equal weight", "equal_weight", "ew", "crp"}:
            return self._benchmark_target_weights(self.training_benchmark_policy)

        raw_return = max(float(np.dot(self.benchmark_weights, price_relative_vector)), 1e-12)
        next_weights = (self.benchmark_weights * price_relative_vector) / raw_return
        return next_weights / (next_weights.sum() + 1e-12)

    def _get_reward(
        self,
        return_signal,
        turnover,
        concentration,
        cash_weight=0.0,
        benchmark_cash_weight=0.0,
        active_overlay_tracking_error=0.0,
    ):
        self.return_memory.append(return_signal)
        penalty = (self.turnover_penalty * turnover) + (self.concentration_penalty * concentration)
        penalty += self.drawdown_penalty * abs(float(self.current_drawdown))
        penalty += self.cash_penalty * max(float(cash_weight) - float(benchmark_cash_weight), 0.0)
        penalty += self.active_overlay_tracking_penalty * float(active_overlay_tracking_error)

        if self.reward_mode in {"log_return", "benchmark_relative"}:
            return float((return_signal * self.return_reward_scale) - penalty)

        window = np.asarray(self.return_memory[-self.sharpe_window:], dtype=float)
        if len(window) < 2:
            reward = (return_signal * self.return_reward_scale) - penalty
            return float(np.clip(reward, -10.0, 10.0))

        rolling_std = np.std(window, ddof=1)
        if rolling_std <= 1e-12:
            return float(np.clip(-penalty, -10.0, 10.0))

        rolling_sharpe = np.mean(window) / rolling_std * np.sqrt(252)
        return float(np.clip(rolling_sharpe - penalty, -10.0, 10.0))

    def _get_prices(self, step):
        return self._close_prices[step].copy()

    def _get_price_relative_history(self, step):
        start = step - self.window + 1
        return self._risky_price_relatives[start:step].copy()

    def get_ipm_target(self, step):
        return self._ipm_targets[step].copy()

    def _get_observation(self, step):
        if step not in self._observation_cache:
            start = step - self.window + 1
            obs_values = self._observation_values[start : step + 1]
            denominators = self._observation_denominators[step]
            self._observation_cache[step] = (obs_values / (denominators + 1e-8)).astype(np.float32)
        return self._observation_cache[step].copy()
