import unittest
from types import SimpleNamespace

from main import selection_score


class ModelSelectionTest(unittest.TestCase):
    def test_benchmark_relative_score_rewards_hurdle_outperformance(self):
        cfg = SimpleNamespace(
            MODEL_SELECTION_METRIC="Benchmark Relative Score",
            SELECTION_RETURN_WEIGHT=0.5,
            SELECTION_DRAWDOWN_WEIGHT=0.25,
            SELECTION_TURNOVER_WEIGHT=0.1,
        )
        agent_metrics = {
            "Sharpe Ratio": 1.2,
            "Total Return": 0.20,
            "Max Drawdown": -0.10,
            "Average Turnover": 0.05,
        }
        hurdle_metrics = {
            "Sharpe Ratio": 0.8,
            "Total Return": 0.10,
            "Max Drawdown": -0.20,
        }

        score = selection_score(agent_metrics, hurdle_metrics, cfg)

        self.assertAlmostEqual(score, 0.47)

    def test_benchmark_relative_score_requires_hurdle_metrics(self):
        cfg = SimpleNamespace(MODEL_SELECTION_METRIC="Benchmark Relative Score")

        with self.assertRaises(ValueError):
            selection_score({}, None, cfg)


if __name__ == "__main__":
    unittest.main()
