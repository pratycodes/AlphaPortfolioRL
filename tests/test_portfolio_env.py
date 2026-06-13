import unittest
from types import SimpleNamespace

import numpy as np
import pandas as pd

from data.features import ohlc_feature_matrix
from env.portfolio_env import PortfolioEnv
from utils.costs import bps_to_rate


def make_config(assets=("AAA", "BBB"), max_weight=0.6):
    return SimpleNamespace(
        ASSETS=list(assets),
        WINDOW_SIZE=3,
        TRADING_COST_BPS=0.002,
        SLIPPAGE_BPS=0.005,
        SPREAD_BPS=0.0005,
        MARKET_IMPACT_BPS=0.0001,
        FIXED_COMMISSION=1.0,
        MAX_WEIGHT=max_weight,
        MAX_CASH_WEIGHT=None,
        REBALANCE_FREQ=1,
        REWARD_MODE="rolling_sharpe",
        RETURN_REWARD_SCALE=100.0,
        TURNOVER_PENALTY=0.0,
        CONCENTRATION_PENALTY=0.0,
        DRAWDOWN_PENALTY=0.0,
        CASH_PENALTY=0.0,
        USE_ACTIVE_OVERLAY=False,
        ACTIVE_OVERLAY_BASE_POLICY="Equal Weight",
        ACTIVE_OVERLAY_BASE_WEIGHTS=None,
        ACTIVE_OVERLAY_BASE_WEIGHT=0.80,
        ACTIVE_OVERLAY_TILT_WEIGHT=0.20,
        ACTIVE_OVERLAY_TRACKING_PENALTY=0.0,
        SHARPE_WINDOW=30,
        USE_MARKET_FEATURE=False,
        MARKET_FEATURE_TICKER="^GSPC",
    )


def make_price_frame(prices, assets=("AAA", "BBB")):
    index = pd.date_range("2025-01-01", periods=len(prices), freq="B")
    columns = pd.MultiIndex.from_product([assets, ["Open", "High", "Low", "Close", "Volume"]])
    df = pd.DataFrame(100.0, index=index, columns=columns)

    for asset_index, asset in enumerate(assets):
        close = np.asarray(prices, dtype=float) + asset_index
        df[(asset, "Open")] = close
        df[(asset, "High")] = close + 1.0
        df[(asset, "Low")] = close - 1.0
        df[(asset, "Close")] = close
        df[(asset, "Volume")] = 1_000.0

    return df


class PortfolioEnvTest(unittest.TestCase):
    def test_max_weight_cap_survives_normalization(self):
        config = make_config()
        df = make_price_frame(np.full(8, 100.0), config.ASSETS)
        env = PortfolioEnv(df, config)

        action = np.array([0.0, 0.8, 0.2])
        _, _, _, _, info = env.step(action)

        np.testing.assert_allclose(info["weights"], np.array([0.2, 0.6, 0.2]), atol=1e-8)
        self.assertLessEqual(info["weights"][1:].max(), config.MAX_WEIGHT + 1e-8)

    def test_observation_uses_same_asset_major_order_as_ipm(self):
        config = make_config()
        df = make_price_frame(np.arange(100.0, 108.0), config.ASSETS)
        env = PortfolioEnv(df, config)

        obs = env._get_observation(config.WINDOW_SIZE - 1)
        risky_expected = ohlc_feature_matrix(df.iloc[: config.WINDOW_SIZE], assets=config.ASSETS)
        expected = np.concatenate([np.ones((config.WINDOW_SIZE, 3)), risky_expected], axis=1)
        close = df.iloc[: config.WINDOW_SIZE].xs("Close", level=1, axis=1).loc[:, config.ASSETS]
        denominator = np.repeat(np.insert(close.iloc[-1].to_numpy(dtype=float), 0, 1.0), 3)
        expected = expected / (denominator + 1e-8)

        np.testing.assert_allclose(obs, expected.astype(np.float32))

    def test_cached_market_arrays_match_pandas_formulas_and_are_copy_safe(self):
        config = make_config()
        df = make_price_frame(np.arange(100.0, 108.0), config.ASSETS)
        env = PortfolioEnv(df, config)
        step = config.WINDOW_SIZE - 1

        prices = env._get_prices(step)
        prices[0] = -1.0
        np.testing.assert_allclose(
            env._get_prices(step),
            df.xs("Close", level=1, axis=1).iloc[step].loc[config.ASSETS].to_numpy(dtype=float),
        )

        ipm_target = env.get_ipm_target(step)
        ipm_target[0] = -999.0
        features = ohlc_feature_matrix(df.iloc[step : step + 2], assets=config.ASSETS)
        expected_target = (features[1] - features[0]) / (features[0] + 1e-8)
        np.testing.assert_allclose(env.get_ipm_target(step), expected_target.astype(np.float32))

        obs = env._get_observation(step)
        obs[0, 0] = -123.0
        self.assertNotEqual(env._get_observation(step)[0, 0], -123.0)

    def test_market_feature_is_observed_but_not_traded(self):
        config = make_config()
        config.USE_MARKET_FEATURE = True
        config.MARKET_FEATURE_TICKER = "SPY"
        df = make_price_frame(np.arange(100.0, 108.0), tuple(config.ASSETS) + ("SPY",))
        env = PortfolioEnv(df, config)

        obs = env._get_observation(config.WINDOW_SIZE)
        self.assertEqual(obs.shape, (config.WINDOW_SIZE, (len(config.ASSETS) + 2) * 3))
        self.assertEqual(env.action_space.shape, (len(config.ASSETS) + 1,))

        _, _, _, _, info = env.step(np.array([0.0, 0.5, 0.5]))
        self.assertEqual(len(info["weights"]), len(config.ASSETS) + 1)

    def test_step_exposes_same_period_price_relatives_for_benchmarks(self):
        config = make_config()
        df = make_price_frame(np.arange(100.0, 108.0), config.ASSETS)
        env = PortfolioEnv(df, config)

        action = np.array([0.0, 1.0, 0.0])
        _, _, _, _, info = env.step(action)

        start = config.WINDOW_SIZE - 1
        expected_relative = df[(config.ASSETS[0], "Close")].iloc[start + 1] / df[(config.ASSETS[0], "Close")].iloc[start]
        self.assertEqual(info["period_start_step"], start)
        self.assertEqual(info["period_end_step"], start + 1)
        self.assertAlmostEqual(info["price_relative_vector"][1], expected_relative)

    def test_transaction_cost_components_are_reported(self):
        config = make_config()
        df = make_price_frame(np.full(8, 100.0), config.ASSETS)
        env = PortfolioEnv(df, config)

        action = np.array([0.0, 0.5, 0.5])
        _, _, _, _, info = env.step(action)

        self.assertIn("cost_components", info)
        self.assertIn("spread", info["cost_components"])
        self.assertIn("market_impact", info["cost_components"])
        self.assertIn("fixed_commission", info["cost_components"])
        self.assertAlmostEqual(info["cost_rate"], sum(info["cost_components"].values()))
        self.assertAlmostEqual(info["cost_components"]["trading"], info["turnover"] * bps_to_rate(config.TRADING_COST_BPS))
        self.assertAlmostEqual(info["cost_components"]["slippage"], info["turnover"] * bps_to_rate(config.SLIPPAGE_BPS))

    def test_cash_cap_redistributes_excess_cash_to_assets(self):
        config = make_config()
        config.MAX_CASH_WEIGHT = 0.05
        df = make_price_frame(np.full(8, 100.0), config.ASSETS)
        env = PortfolioEnv(df, config)

        weights = env._normalize_action(np.array([1.0, 0.0, 0.0]))

        self.assertAlmostEqual(weights[0], 0.05)
        self.assertAlmostEqual(weights.sum(), 1.0)
        self.assertLessEqual(weights[1:].max(), config.MAX_WEIGHT + 1e-8)

    def test_active_overlay_blends_actor_request_with_equal_weight_base(self):
        config = make_config(max_weight=1.0)
        config.USE_ACTIVE_OVERLAY = True
        config.ACTIVE_OVERLAY_BASE_WEIGHT = 0.80
        config.ACTIVE_OVERLAY_TILT_WEIGHT = 0.20
        df = make_price_frame(np.full(8, 100.0), config.ASSETS)
        env = PortfolioEnv(df, config)

        _, _, _, _, info = env.step(np.array([0.0, 1.0, 0.0]))

        np.testing.assert_allclose(info["requested_weights"], np.array([0.0, 1.0, 0.0]), atol=1e-8)
        np.testing.assert_allclose(info["executed_action"], np.array([0.0, 0.6, 0.4]), atol=1e-8)
        np.testing.assert_allclose(info["weights"], np.array([0.0, 0.6, 0.4]), atol=1e-8)
        self.assertAlmostEqual(info["active_overlay_tracking_error"], 0.1)

    def test_active_overlay_can_use_benchmark_proxy_base_weights(self):
        config = make_config(max_weight=1.0)
        config.USE_ACTIVE_OVERLAY = True
        config.ACTIVE_OVERLAY_BASE_POLICY = "Benchmark Proxy"
        config.ACTIVE_OVERLAY_BASE_WEIGHTS = [0.75, 0.25]
        config.ACTIVE_OVERLAY_BASE_WEIGHT = 0.90
        config.ACTIVE_OVERLAY_TILT_WEIGHT = 0.10
        df = make_price_frame(np.full(8, 100.0), config.ASSETS)
        env = PortfolioEnv(df, config)

        _, _, _, _, info = env.step(np.array([0.0, 0.0, 1.0]))

        np.testing.assert_allclose(info["active_overlay_base_weights"], np.array([0.0, 0.75, 0.25]), atol=1e-8)
        np.testing.assert_allclose(info["executed_action"], np.array([0.0, 0.675, 0.325]), atol=1e-8)

    def test_disabled_active_overlay_preserves_requested_action(self):
        config = make_config(max_weight=1.0)
        df = make_price_frame(np.full(8, 100.0), config.ASSETS)
        env = PortfolioEnv(df, config)

        executed = env.transform_action_for_execution(np.array([0.0, 1.0, 0.0]))

        np.testing.assert_allclose(executed, np.array([0.0, 1.0, 0.0]), atol=1e-8)

    def test_active_overlay_tracking_penalty_reduces_reward(self):
        config = make_config(max_weight=1.0)
        config.USE_ACTIVE_OVERLAY = True
        config.REWARD_MODE = "log_return"
        config.TRADING_COST_BPS = 0.0
        config.SLIPPAGE_BPS = 0.0
        config.SPREAD_BPS = 0.0
        config.MARKET_IMPACT_BPS = 0.0
        config.FIXED_COMMISSION = 0.0
        config.ACTIVE_OVERLAY_BASE_WEIGHT = 0.80
        config.ACTIVE_OVERLAY_TILT_WEIGHT = 0.20
        config.ACTIVE_OVERLAY_TRACKING_PENALTY = 2.0
        df = make_price_frame(np.full(8, 100.0), config.ASSETS)
        env = PortfolioEnv(df, config)

        _, reward, _, _, info = env.step(np.array([0.0, 1.0, 0.0]))

        self.assertAlmostEqual(info["active_overlay_tracking_error"], 0.1)
        self.assertAlmostEqual(reward, -0.2, places=5)

    def test_rebalance_frequency_holds_weights_between_rebalances(self):
        config = make_config()
        config.REBALANCE_FREQ = 5
        config.FIXED_COMMISSION = 0.0
        df = make_price_frame(np.full(12, 100.0), config.ASSETS)
        env = PortfolioEnv(df, config)

        _, _, _, _, first_info = env.step(np.array([0.0, 0.5, 0.5]))
        _, _, _, _, second_info = env.step(np.array([1.0, 0.0, 0.0]))

        self.assertTrue(first_info["rebalanced"])
        self.assertFalse(second_info["rebalanced"])
        np.testing.assert_allclose(second_info["executed_action"], first_info["weights"], atol=1e-8)
        self.assertAlmostEqual(second_info["turnover"], 0.0)

    def test_reward_penalizes_turnover_and_concentration(self):
        config = make_config()
        config.REWARD_MODE = "log_return"
        config.TURNOVER_PENALTY = 1.0
        config.CONCENTRATION_PENALTY = 1.0
        config.TRADING_COST_BPS = 0.0
        config.SLIPPAGE_BPS = 0.0
        config.SPREAD_BPS = 0.0
        config.MARKET_IMPACT_BPS = 0.0
        config.FIXED_COMMISSION = 0.0
        df = make_price_frame(np.full(8, 100.0), config.ASSETS)
        env = PortfolioEnv(df, config)

        _, reward, _, _, info = env.step(np.array([0.0, 1.0, 0.0]))

        expected_penalty = info["turnover"] + info["concentration"]
        self.assertAlmostEqual(reward, -expected_penalty, places=5)

    def test_benchmark_relative_reward_uses_excess_return(self):
        config = make_config()
        config.REWARD_MODE = "benchmark_relative"
        config.TRADING_COST_BPS = 0.0
        config.SLIPPAGE_BPS = 0.0
        config.SPREAD_BPS = 0.0
        config.MARKET_IMPACT_BPS = 0.0
        config.FIXED_COMMISSION = 0.0
        config.TURNOVER_PENALTY = 0.0
        config.CONCENTRATION_PENALTY = 0.0
        df = make_price_frame(np.full(8, 100.0), config.ASSETS)
        step = config.WINDOW_SIZE
        df.loc[df.index[step + 1], (config.ASSETS[0], "Close")] = 110.0
        df.loc[df.index[step + 1], (config.ASSETS[0], "High")] = 111.0
        df.loc[df.index[step + 1], (config.ASSETS[0], "Low")] = 109.0
        env = PortfolioEnv(df, config)

        _, reward, _, _, info = env.step(np.array([0.0, 1.0, 0.0]))

        expected = (
            np.log(1.0 + info["net_return"] + 1e-8)
            - np.log(1.0 + info["benchmark_return"] + 1e-8)
        ) * config.RETURN_REWARD_SCALE
        self.assertAlmostEqual(reward, expected, places=5)

    def test_cash_penalty_discourages_cash_above_training_benchmark(self):
        config = make_config()
        config.REWARD_MODE = "log_return"
        config.TRADING_COST_BPS = 0.0
        config.SLIPPAGE_BPS = 0.0
        config.SPREAD_BPS = 0.0
        config.MARKET_IMPACT_BPS = 0.0
        config.FIXED_COMMISSION = 0.0
        config.TURNOVER_PENALTY = 0.0
        config.CONCENTRATION_PENALTY = 0.0
        config.CASH_PENALTY = 1.0
        df = make_price_frame(np.full(8, 100.0), config.ASSETS)
        env = PortfolioEnv(df, config)

        _, reward, _, _, _ = env.step(np.array([1.0, 0.0, 0.0]))

        self.assertAlmostEqual(reward, -1.0, places=5)

    def test_drawdown_is_reported_after_portfolio_value_falls_from_peak(self):
        config = make_config(max_weight=1.0)
        config.REWARD_MODE = "log_return"
        config.TRADING_COST_BPS = 0.0
        config.SLIPPAGE_BPS = 0.0
        config.SPREAD_BPS = 0.0
        config.MARKET_IMPACT_BPS = 0.0
        config.FIXED_COMMISSION = 0.0
        df = make_price_frame(np.array([100.0, 100.0, 100.0, 110.0, 90.0, 90.0, 90.0, 90.0]), config.ASSETS)
        env = PortfolioEnv(df, config)

        env.step(np.array([0.0, 1.0, 0.0]))
        _, _, _, _, info = env.step(np.array([0.0, 1.0, 0.0]))

        self.assertLess(info["drawdown"], 0.0)
        self.assertAlmostEqual(info["peak_portfolio_value"], 11000.0)
        self.assertAlmostEqual(info["drawdown"], (9000.0 - 11000.0) / 11000.0)

    def test_drawdown_penalty_reduces_reward_but_zero_penalty_preserves_default(self):
        prices = np.array([100.0, 100.0, 100.0, 110.0, 90.0, 90.0, 90.0, 90.0])
        base_config = make_config(max_weight=1.0)
        base_config.REWARD_MODE = "log_return"
        base_config.TRADING_COST_BPS = 0.0
        base_config.SLIPPAGE_BPS = 0.0
        base_config.SPREAD_BPS = 0.0
        base_config.MARKET_IMPACT_BPS = 0.0
        base_config.FIXED_COMMISSION = 0.0
        df = make_price_frame(prices, base_config.ASSETS)

        unpenalized_env = PortfolioEnv(df, base_config)
        unpenalized_env.step(np.array([0.0, 1.0, 0.0]))
        _, unpenalized_reward, _, _, unpenalized_info = unpenalized_env.step(np.array([0.0, 1.0, 0.0]))

        penalized_config = make_config(max_weight=1.0)
        penalized_config.REWARD_MODE = "log_return"
        penalized_config.TRADING_COST_BPS = 0.0
        penalized_config.SLIPPAGE_BPS = 0.0
        penalized_config.SPREAD_BPS = 0.0
        penalized_config.MARKET_IMPACT_BPS = 0.0
        penalized_config.FIXED_COMMISSION = 0.0
        penalized_config.DRAWDOWN_PENALTY = 2.0
        penalized_env = PortfolioEnv(df, penalized_config)
        penalized_env.step(np.array([0.0, 1.0, 0.0]))
        _, penalized_reward, _, _, penalized_info = penalized_env.step(np.array([0.0, 1.0, 0.0]))

        self.assertAlmostEqual(unpenalized_info["drawdown"], penalized_info["drawdown"])
        self.assertAlmostEqual(
            penalized_reward,
            unpenalized_reward - (2.0 * abs(penalized_info["drawdown"])),
            places=5,
        )


if __name__ == "__main__":
    unittest.main()
