import unittest

from evaluation.metrics import FinancialMetrics


class FinancialMetricsTest(unittest.TestCase):
    def test_max_drawdown_includes_starting_equity(self):
        metrics = FinancialMetrics.get_metrics([-0.10])
        self.assertAlmostEqual(metrics["Max Drawdown"], -0.10)

    def test_empty_returns_are_well_defined(self):
        metrics = FinancialMetrics.get_metrics([])
        self.assertEqual(metrics["Total Return"], 0.0)
        self.assertEqual(metrics["Max Drawdown"], 0.0)


if __name__ == "__main__":
    unittest.main()
