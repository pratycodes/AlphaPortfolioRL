import unittest

import numpy as np

from optimization.oracle import PortfolioOracle
from utils.costs import bps_to_rate


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

    def test_historical_oracle_respects_cash_cap(self):
        oracle = PortfolioOracle(num_assets=3, transaction_cost_bps=0.001)
        current_weights = np.array([1.0, 0.0, 0.0])
        history = np.array(
            [
                [0.99, 0.98],
                [0.98, 0.97],
                [0.99, 0.98],
            ]
        )

        weights = oracle.get_historical_weights(
            current_weights,
            history,
            max_weight=0.6,
            max_cash_weight=0.05,
        )

        self.assertAlmostEqual(weights.sum(), 1.0)
        self.assertLessEqual(weights[0], 0.05 + 1e-5)
        self.assertLessEqual(weights[1:].max(), 0.6 + 1e-5)

    def test_one_step_oracle_respects_portfolio_caps(self):
        oracle = PortfolioOracle(num_assets=4, transaction_cost_bps=0.001)
        current_weights = np.array([1.0, 0.0, 0.0, 0.0])
        price_relative_vector = np.array([1.0, 1.10, 0.90, 1.05])

        weights = oracle.get_optimal_weights(
            current_weights,
            price_relative_vector,
            max_weight=0.4,
            max_cash_weight=0.05,
        )

        self.assertAlmostEqual(weights.sum(), 1.0)
        self.assertLessEqual(weights[0], 0.05 + 1e-5)
        self.assertLessEqual(weights[1:].max(), 0.4 + 1e-5)

    def test_transaction_cost_argument_is_basis_points(self):
        oracle = PortfolioOracle(num_assets=3, transaction_cost_bps=2.5)

        self.assertAlmostEqual(oracle.cost, bps_to_rate(2.5))


if __name__ == "__main__":
    unittest.main()
