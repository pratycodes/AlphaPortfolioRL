import json
import os
import tempfile
import unittest
from pathlib import Path

from experiments.aggregate import aggregate_runs


class ExperimentAggregateTest(unittest.TestCase):
    def test_aggregate_selects_best_validation_score(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir) / "run_test"
            run_dir.mkdir()
            with (run_dir / "config.json").open("w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "SEED": 42,
                        "ABLATION": "ddpg_ipm_bcm",
                        "LR_ACTOR": 1e-5,
                        "TRADING_COST_BPS": 20.0,
                        "MAX_CASH_WEIGHT": 0.25,
                    },
                    handle,
                )
            records = [
                {"stage": "validation", "step": 5, "metrics": {"Selection Score": -1.0, "Sharpe Ratio": 0.1}},
                {"stage": "validation", "step": 10, "metrics": {"Selection Score": 0.2, "Sharpe Ratio": 0.5}},
                {"stage": "train_episode", "step": 10, "metrics": {"critic_loss": 0.01}},
            ]
            with (run_dir / "metrics.jsonl").open("w", encoding="utf-8") as handle:
                for record in records:
                    handle.write(json.dumps(record) + "\n")

            summary = aggregate_runs(tmpdir)

        self.assertEqual(len(summary), 1)
        self.assertEqual(summary.loc[0, "best_val_step"], 10)
        self.assertEqual(summary.loc[0, "best_val_selection_score"], 0.2)
        self.assertEqual(summary.loc[0, "final_train_critic_loss"], 0.01)
        self.assertEqual(summary.loc[0, "lr_actor"], 1e-5)
        self.assertEqual(summary.loc[0, "trading_cost_bps"], 20.0)
        self.assertEqual(summary.loc[0, "max_cash_weight"], 0.25)

    def test_aggregate_can_filter_by_manifest(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            selected = Path(tmpdir) / "run_selected"
            selected.mkdir()
            ignored = Path(tmpdir) / "run_ignored"
            ignored.mkdir()

            selected_config = {
                "SEED": 42,
                "ABLATION": "current_baseline",
                "TRAIN_START_DATE": "2018-01-01",
                "TRAIN_END_DATE": "2021-12-31",
                "VALID_START_DATE": "2022-01-01",
                "VALID_END_DATE": "2022-12-31",
                "TEST_START_DATE": "2023-01-01",
                "TEST_END_DATE": "2023-12-31",
            }
            ignored_config = dict(selected_config)
            ignored_config["ABLATION"] = "no_bcm"

            for run_dir, config in ((selected, selected_config), (ignored, ignored_config)):
                with (run_dir / "config.json").open("w", encoding="utf-8") as handle:
                    json.dump(config, handle)
                with (run_dir / "metrics.jsonl").open("w", encoding="utf-8") as handle:
                    handle.write(json.dumps({"stage": "validation", "step": 5, "metrics": {"Selection Score": 1.0}}) + "\n")

            manifest = Path(tmpdir) / "manifest.json"
            with manifest.open("w", encoding="utf-8") as handle:
                json.dump(
                    [
                        {
                            "seed": 42,
                            "ablation": "current_baseline",
                            "train_start_date": "2018-01-01",
                            "train_end_date": "2021-12-31",
                            "valid_start_date": "2022-01-01",
                            "valid_end_date": "2022-12-31",
                            "test_start_date": "2023-01-01",
                            "test_end_date": "2023-12-31",
                        }
                    ],
                    handle,
                )

            summary = aggregate_runs(tmpdir, manifest=manifest)

        self.assertEqual(len(summary), 1)
        self.assertEqual(summary.loc[0, "ablation"], "current_baseline")

    def test_manifest_filter_keeps_latest_duplicate_run_key(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = {
                "SEED": 42,
                "ABLATION": "current_baseline",
                "TRAIN_START_DATE": "2018-01-01",
                "TRAIN_END_DATE": "2021-12-31",
                "VALID_START_DATE": "2022-01-01",
                "VALID_END_DATE": "2022-12-31",
                "TEST_START_DATE": "2023-01-01",
                "TEST_END_DATE": "2023-12-31",
            }
            runs = [
                ("run_old", 1_000_000_000, 0.1),
                ("run_new", 1_000_000_100, 0.9),
            ]
            for name, mtime, score in runs:
                run_dir = Path(tmpdir) / name
                run_dir.mkdir()
                with (run_dir / "config.json").open("w", encoding="utf-8") as handle:
                    json.dump(config, handle)
                with (run_dir / "metrics.jsonl").open("w", encoding="utf-8") as handle:
                    handle.write(
                        json.dumps(
                            {"stage": "validation", "step": 5, "metrics": {"Selection Score": score}}
                        )
                        + "\n"
                    )
                os.utime(run_dir, (mtime, mtime))

            manifest = Path(tmpdir) / "manifest.json"
            with manifest.open("w", encoding="utf-8") as handle:
                json.dump(
                    [
                        {
                            "seed": 42,
                            "ablation": "current_baseline",
                            "train_start_date": "2018-01-01",
                            "train_end_date": "2021-12-31",
                            "valid_start_date": "2022-01-01",
                            "valid_end_date": "2022-12-31",
                            "test_start_date": "2023-01-01",
                            "test_end_date": "2023-12-31",
                        }
                    ],
                    handle,
                )

            summary = aggregate_runs(tmpdir, manifest=manifest)

        self.assertEqual(len(summary), 1)
        self.assertEqual(summary.loc[0, "run_id"], "run_new")
        self.assertEqual(summary.loc[0, "best_val_selection_score"], 0.9)


if __name__ == "__main__":
    unittest.main()
