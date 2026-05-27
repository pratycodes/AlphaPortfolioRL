import unittest
from types import SimpleNamespace

from experiments.runner import ABLATION_OVERRIDES, build_runs, walk_forward_splits


class ExperimentRunnerTest(unittest.TestCase):
    def test_ablation_matrix_contains_required_variants(self):
        self.assertIn("ddpg_only", ABLATION_OVERRIDES)
        self.assertEqual(ABLATION_OVERRIDES["ddpg_only"]["USE_IPM"], "false")
        self.assertEqual(ABLATION_OVERRIDES["ddpg_only"]["USE_ARB"], "false")
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


if __name__ == "__main__":
    unittest.main()
