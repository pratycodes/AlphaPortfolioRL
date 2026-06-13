import unittest

import numpy as np

from evaluation.baselines import baseline_policies, run_baselines, run_policy
from tests.test_portfolio_env import make_config, make_price_frame


class BaselineTest(unittest.TestCase):
    def test_expected_baselines_are_registered(self):
        names = set(baseline_policies(make_config()).keys())
        self.assertIn("Equal Weight", names)
        self.assertIn("Buy & Hold EW", names)
        self.assertIn("Inverse Vol", names)
        self.assertIn("Min Variance", names)
        self.assertIn("Mean-Variance", names)
        self.assertIn("Momentum", names)
        self.assertIn("Random Long-Only", names)

    def test_baselines_return_metrics_and_values(self):
        config = make_config()
        df = make_price_frame(np.arange(100.0, 120.0), config.ASSETS)
        results = run_baselines(df, config, names=["Equal Weight", "Momentum", "Random Long-Only"])

        self.assertEqual(set(results), {"Equal Weight", "Momentum", "Random Long-Only"})
        for result in results.values():
            self.assertIn("values", result)
            self.assertIn("metrics", result)
            self.assertIn("Final Value", result["metrics"])
            self.assertGreater(len(result["values"]), 1)

    def test_baseline_policy_is_not_projected_through_active_overlay(self):
        config = make_config(max_weight=1.0)
        config.USE_ACTIVE_OVERLAY = True
        config.ACTIVE_OVERLAY_BASE_WEIGHT = 0.80
        config.ACTIVE_OVERLAY_TILT_WEIGHT = 0.20
        df = make_price_frame(np.full(8, 100.0), config.ASSETS)

        result = run_policy(df, config, lambda env: np.array([0.0, 1.0, 0.0]))

        first_rebalance_weights = result["weights"].iloc[1].to_numpy()
        np.testing.assert_allclose(first_rebalance_weights, np.array([0.0, 1.0, 0.0]), atol=1e-8)


if __name__ == "__main__":
    unittest.main()
