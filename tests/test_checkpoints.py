import unittest
from types import SimpleNamespace

import numpy as np

from utils.checkpoints import _to_builtin, checkpoint_metadata, universe_id, validate_checkpoint_metadata


def make_config(assets):
    return SimpleNamespace(
        ASSETS=list(assets),
        WINDOW_SIZE=10,
        BENCHMARK_TICKER="^GSPC",
        BENCHMARK_NAME="S&P 500",
    )


class CheckpointMetadataTest(unittest.TestCase):
    def test_universe_id_changes_with_asset_universe(self):
        self.assertNotEqual(universe_id(make_config(["AAPL"])), universe_id(make_config(["AAPL", "MSFT"])))

    def test_metadata_validation_rejects_mismatched_assets(self):
        metadata = checkpoint_metadata(make_config(["AAPL", "MSFT"]))
        with self.assertRaises(RuntimeError):
            validate_checkpoint_metadata(metadata, make_config(["AAPL"]))

    def test_metric_scalars_are_serialized_as_python_types(self):
        metrics = {
            "Sharpe Ratio": np.float64(1.25),
            "Drawdowns": np.array([0.1, 0.2]),
        }

        converted = _to_builtin(metrics)

        self.assertIsInstance(converted["Sharpe Ratio"], float)
        self.assertEqual(converted["Drawdowns"], [0.1, 0.2])


if __name__ == "__main__":
    unittest.main()
