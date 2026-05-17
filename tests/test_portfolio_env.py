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


if __name__ == "__main__":
    unittest.main()
