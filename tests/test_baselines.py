import unittest

import numpy as np

from evaluation.baselines import baseline_policies, run_baselines
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

    def test_baselines_return_metrics_and_values(self):
        config = make_config()
        df = make_price_frame(np.arange(100.0, 120.0), config.ASSETS)
        results = run_baselines(df, config, names=["Equal Weight", "Momentum"])

        self.assertEqual(set(results), {"Equal Weight", "Momentum"})
        for result in results.values():
            self.assertIn("values", result)
            self.assertIn("metrics", result)
            self.assertIn("Final Value", result["metrics"])
            self.assertGreater(len(result["values"]), 1)


if __name__ == "__main__":
    unittest.main()
