import unittest

import pandas as pd

from evaluation.walkforward_champion_selector import summarize_candidates


class WalkForwardChampionSelectorTest(unittest.TestCase):
    def test_summarize_candidates_prefers_candidate_that_passes_all_windows(self):
        rows = [
            {
                "candidate": "unstable",
                "window": "a",
                "return_gap_vs_nifty": 0.10,
                "sharpe_gap_vs_nifty": 0.20,
                "return_gap_vs_equal_weight": 0.10,
                "sharpe_gap_vs_equal_weight": 0.20,
                "rl_return": 0.20,
                "rl_sharpe": 1.0,
                "rl_max_drawdown": -0.10,
                "rl_average_turnover": 0.01,
                "rl_average_cash": 0.01,
            },
            {
                "candidate": "unstable",
                "window": "b",
                "return_gap_vs_nifty": -0.01,
                "sharpe_gap_vs_nifty": 0.10,
                "return_gap_vs_equal_weight": 0.10,
                "sharpe_gap_vs_equal_weight": 0.20,
                "rl_return": 0.10,
                "rl_sharpe": 0.8,
                "rl_max_drawdown": -0.11,
                "rl_average_turnover": 0.02,
                "rl_average_cash": 0.01,
            },
            {
                "candidate": "robust",
                "window": "a",
                "return_gap_vs_nifty": 0.02,
                "sharpe_gap_vs_nifty": 0.03,
                "return_gap_vs_equal_weight": 0.02,
                "sharpe_gap_vs_equal_weight": 0.03,
                "rl_return": 0.12,
                "rl_sharpe": 0.7,
                "rl_max_drawdown": -0.08,
                "rl_average_turnover": 0.01,
                "rl_average_cash": 0.01,
            },
            {
                "candidate": "robust",
                "window": "b",
                "return_gap_vs_nifty": 0.01,
                "sharpe_gap_vs_nifty": 0.02,
                "return_gap_vs_equal_weight": 0.01,
                "sharpe_gap_vs_equal_weight": 0.02,
                "rl_return": 0.11,
                "rl_sharpe": 0.6,
                "rl_max_drawdown": -0.09,
                "rl_average_turnover": 0.02,
                "rl_average_cash": 0.01,
            },
        ]

        summary = summarize_candidates(pd.DataFrame(rows))

        self.assertEqual("robust", summary.iloc[0]["candidate"])
        self.assertEqual("robust_champion", summary.iloc[0]["selection_status"])


if __name__ == "__main__":
    unittest.main()
