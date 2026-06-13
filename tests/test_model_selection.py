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
            "Average Turnover": 0.05,
        }

        score = selection_score(agent_metrics, hurdle_metrics, cfg)

        self.assertAlmostEqual(score, 0.475)

    def test_benchmark_relative_score_penalizes_turnover_gap_not_raw_turnover(self):
        cfg = SimpleNamespace(
            MODEL_SELECTION_METRIC="Benchmark Relative Score",
            SELECTION_RETURN_WEIGHT=0.0,
            SELECTION_DRAWDOWN_WEIGHT=0.0,
            SELECTION_TURNOVER_WEIGHT=0.1,
        )
        agent_metrics = {
            "Sharpe Ratio": 1.0,
            "Total Return": 0.10,
            "Max Drawdown": -0.10,
            "Average Turnover": 0.10,
        }
        hurdle_metrics = {
            "Sharpe Ratio": 1.0,
            "Total Return": 0.10,
            "Max Drawdown": -0.10,
            "Average Turnover": 0.08,
        }

        score = selection_score(agent_metrics, hurdle_metrics, cfg)

        self.assertAlmostEqual(score, -0.002)

    def test_benchmark_relative_score_requires_hurdle_metrics(self):
        cfg = SimpleNamespace(MODEL_SELECTION_METRIC="Benchmark Relative Score")

        with self.assertRaises(ValueError):
            selection_score({}, None, cfg)

    def test_multi_hurdle_score_uses_weakest_hurdle(self):
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
            "CRP": {
                "Sharpe Ratio": 0.8,
                "Total Return": 0.10,
                "Max Drawdown": -0.20,
                "Average Turnover": 0.05,
            },
            "Equal Weight": {
                "Sharpe Ratio": 1.3,
                "Total Return": 0.25,
                "Max Drawdown": -0.09,
                "Average Turnover": 0.04,
            },
        }

        score = selection_score(agent_metrics, hurdle_metrics, cfg)

        self.assertAlmostEqual(score, -0.1285)

    def test_higher_drawdown_weight_penalizes_worse_drawdown_more_strongly(self):
        agent_metrics = {
            "Sharpe Ratio": 1.0,
            "Total Return": 0.10,
            "Max Drawdown": -0.15,
            "Average Turnover": 0.05,
        }
        hurdle_metrics = {
            "Sharpe Ratio": 1.0,
            "Total Return": 0.10,
            "Max Drawdown": -0.10,
            "Average Turnover": 0.05,
        }
        mild_cfg = SimpleNamespace(
            MODEL_SELECTION_METRIC="Benchmark Relative Score",
            SELECTION_RETURN_WEIGHT=0.0,
            SELECTION_DRAWDOWN_WEIGHT=0.25,
            SELECTION_TURNOVER_WEIGHT=0.0,
        )
        strict_cfg = SimpleNamespace(
            MODEL_SELECTION_METRIC="Benchmark Relative Score",
            SELECTION_RETURN_WEIGHT=0.0,
            SELECTION_DRAWDOWN_WEIGHT=1.0,
            SELECTION_TURNOVER_WEIGHT=0.0,
        )

        mild_score = selection_score(agent_metrics, hurdle_metrics, mild_cfg)
        strict_score = selection_score(agent_metrics, hurdle_metrics, strict_cfg)

        self.assertAlmostEqual(mild_score, -0.0125)
        self.assertAlmostEqual(strict_score, -0.05)
        self.assertLess(strict_score, mild_score)


if __name__ == "__main__":
    unittest.main()
