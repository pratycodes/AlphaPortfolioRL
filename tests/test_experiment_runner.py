import unittest
from unittest.mock import patch
from types import SimpleNamespace

from config.settings import Settings
from experiments.runner import (
    ABLATION_OVERRIDES,
    FOCUSED_ABLATIONS,
    FOCUSED_SEEDS,
    FOCUSED_SMOKE_OVERRIDES,
    FOCUSED_SMOKE_SEEDS,
    build_focused_runs,
    build_runs,
    walk_forward_splits,
)


class ExperimentRunnerTest(unittest.TestCase):
    def test_ablation_matrix_contains_required_variants(self):
        self.assertIn("ddpg_only", ABLATION_OVERRIDES)
        self.assertEqual(ABLATION_OVERRIDES["ddpg_only"]["USE_IPM"], "false")
        self.assertEqual(ABLATION_OVERRIDES["ddpg_only"]["USE_ARB"], "false")
        self.assertEqual(ABLATION_OVERRIDES["ddpg_ipm_bcm_arb"]["USE_ARB"], "true")
        self.assertEqual(ABLATION_OVERRIDES["ddpg_ipm_bcm_arb"]["USE_SPARSE_NETWORK"], "false")
        self.assertEqual(ABLATION_OVERRIDES["ddpg_ipm_bcm_sparse"]["USE_ARB"], "false")
        self.assertEqual(ABLATION_OVERRIDES["ddpg_ipm_bcm_sparse"]["USE_SPARSE_NETWORK"], "true")
        self.assertEqual(ABLATION_OVERRIDES["ddpg_ipm_bcm_dam_arb_sparse"]["USE_DAM"], "true")
        self.assertEqual(ABLATION_OVERRIDES["ddpg_ipm_bcm_dam_arb_sparse"]["USE_ARB"], "true")

    def test_walk_forward_splits_are_chronological(self):
        cfg = SimpleNamespace(
            WALK_FORWARD_START_YEAR=2010,
            WALK_FORWARD_END_YEAR=2013,
            WALK_FORWARD_TRAIN_YEARS=2,
            WALK_FORWARD_VALID_YEARS=1,
            WALK_FORWARD_TEST_YEARS=1,
        )
        splits = walk_forward_splits(cfg)

        self.assertEqual(len(splits), 1)
        self.assertEqual(splits[0]["train_start_date"], "2010-01-01")
        self.assertEqual(splits[0]["test_end_date"], "2013-12-31")

    def test_build_runs_crosses_seed_ablation_and_split(self):
        cfg = SimpleNamespace(
            WALK_FORWARD_START_YEAR=2010,
            WALK_FORWARD_END_YEAR=2013,
            WALK_FORWARD_TRAIN_YEARS=2,
            WALK_FORWARD_VALID_YEARS=1,
            WALK_FORWARD_TEST_YEARS=1,
            ABLATIONS=["ddpg_only", "ddpg_ipm"],
            EXPERIMENT_SEEDS=[1, 2],
        )
        runs = build_runs(cfg)

        self.assertEqual(len(runs), 4)
        self.assertEqual({run.seed for run in runs}, {1, 2})
        self.assertEqual({run.ablation for run in runs}, {"ddpg_only", "ddpg_ipm"})
        self.assertEqual(runs[0].env()["ABLATION"], runs[0].ablation)

    def test_focused_matrix_uses_current_split_and_tuning_variants(self):
        cfg = SimpleNamespace(
            TRAIN_START_DATE="2010-01-01",
            TRAIN_END_DATE="2023-12-31",
            VALID_START_DATE="2024-01-01",
            VALID_END_DATE="2024-12-31",
            TEST_START_DATE="2025-01-01",
            TEST_END_DATE="2026-05-27",
        )
        runs = build_focused_runs(cfg)

        self.assertEqual(len(runs), len(FOCUSED_ABLATIONS) * len(FOCUSED_SEEDS))
        self.assertEqual({run.seed for run in runs}, set(FOCUSED_SEEDS))
        self.assertEqual({run.ablation for run in runs}, set(FOCUSED_ABLATIONS))
        self.assertEqual(runs[0].train_start_date, "2010-01-01")
        self.assertEqual(runs[0].test_end_date, "2026-05-27")
        self.assertEqual(ABLATION_OVERRIDES["paper_costs"]["TRADING_COST_BPS"], "5.0")
        self.assertEqual(ABLATION_OVERRIDES["free_cash"]["MAX_CASH_WEIGHT"], "none")

    def test_focused_smoke_matrix_uses_short_split_and_one_seed(self):
        runs = build_focused_runs(smoke=True)

        self.assertEqual(len(runs), len(FOCUSED_ABLATIONS) * len(FOCUSED_SMOKE_SEEDS))
        self.assertEqual({run.seed for run in runs}, set(FOCUSED_SMOKE_SEEDS))
        self.assertEqual(runs[0].train_start_date, "2018-01-01")
        self.assertEqual(runs[0].test_end_date, "2023-12-31")
        env = runs[0].env()
        self.assertEqual(env["EPISODES"], FOCUSED_SMOKE_OVERRIDES["EPISODES"])
        self.assertEqual(env["MIN_SAVE_EPISODE"], FOCUSED_SMOKE_OVERRIDES["MIN_SAVE_EPISODE"])
        self.assertEqual(env["USE_DAM"], "false")

    def test_optional_cash_cap_env_override_accepts_none(self):
        with patch.dict("os.environ", {"MAX_CASH_WEIGHT": "none"}):
            cfg = Settings()

        self.assertIsNone(cfg.MAX_CASH_WEIGHT)


if __name__ == "__main__":
    unittest.main()
