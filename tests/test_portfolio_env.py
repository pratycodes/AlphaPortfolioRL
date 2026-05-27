import unittest
from types import SimpleNamespace

import numpy as np
import pandas as pd

from data.features import ohlc_feature_matrix
from env.portfolio_env import PortfolioEnv


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
        SHARPE_WINDOW=30,
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

        obs = env._get_observation(config.WINDOW_SIZE)
        expected = ohlc_feature_matrix(df.iloc[: config.WINDOW_SIZE])
        expected = expected / (expected[0] + 1e-8)

        np.testing.assert_allclose(obs, expected.astype(np.float32))

    def test_step_exposes_same_period_price_relatives_for_benchmarks(self):
        config = make_config()
        df = make_price_frame(np.arange(100.0, 108.0), config.ASSETS)
        env = PortfolioEnv(df, config)

        action = np.array([0.0, 1.0, 0.0])
        _, _, _, _, info = env.step(action)

        start = config.WINDOW_SIZE
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

    def test_cash_cap_redistributes_excess_cash_to_assets(self):
        config = make_config()
        config.MAX_CASH_WEIGHT = 0.05
        df = make_price_frame(np.full(8, 100.0), config.ASSETS)
        env = PortfolioEnv(df, config)

        weights = env._normalize_action(np.array([1.0, 0.0, 0.0]))

        self.assertAlmostEqual(weights[0], 0.05)
        self.assertAlmostEqual(weights.sum(), 1.0)
        self.assertLessEqual(weights[1:].max(), config.MAX_WEIGHT + 1e-8)

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


if __name__ == "__main__":
    unittest.main()
