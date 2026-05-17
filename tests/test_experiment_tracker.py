import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from utils.experiment_tracker import ExperimentTracker


class ExperimentTrackerTest(unittest.TestCase):
    def test_writes_manifest_and_metrics(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = SimpleNamespace(
                EXPERIMENT_DIR=tmpdir,
                SEED=42,
                model_dump=lambda: {"SEED": 42, "EXPERIMENT_DIR": tmpdir},
            )
            tracker = ExperimentTracker(config, run_name="test_run")
            tracker.write_manifest(config)
            tracker.log_metrics("validation", 5, {"Sharpe Ratio": 1.2})

            run_dir = Path(tmpdir) / "test_run"
            self.assertTrue((run_dir / "manifest.json").exists())
            self.assertTrue((run_dir / "config.json").exists())
            self.assertTrue((run_dir / "metrics.jsonl").exists())

            with (run_dir / "metrics.jsonl").open(encoding="utf-8") as handle:
                record = json.loads(handle.readline())
            self.assertEqual(record["stage"], "validation")
            self.assertEqual(record["step"], 5)


if __name__ == "__main__":
    unittest.main()
