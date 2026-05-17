import unittest

import numpy as np

from optimization.oracle import PortfolioOracle


class PortfolioOracleTest(unittest.TestCase):
    def test_historical_oracle_respects_simplex_and_max_weight(self):
        oracle = PortfolioOracle(num_assets=3, transaction_cost_bps=0.001)
        current_weights = np.array([1.0, 0.0, 0.0])
        history = np.array(
            [
                [1.01, 0.99],
                [1.02, 0.98],
                [1.01, 1.00],
            ]
        )

        weights = oracle.get_historical_weights(current_weights, history, max_weight=0.6)

        self.assertAlmostEqual(weights.sum(), 1.0)
        self.assertTrue(np.all(weights >= 0.0))
        self.assertLessEqual(weights[1:].max(), 0.6 + 1e-6)


if __name__ == "__main__":
    unittest.main()
