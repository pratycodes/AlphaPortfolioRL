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
        USE_SPARSE_NETWORK=True,
        SPARSE_DENSITY=0.35,
    )


class CheckpointMetadataTest(unittest.TestCase):
    def test_universe_id_changes_with_asset_universe(self):
        self.assertNotEqual(universe_id(make_config(["AAPL"])), universe_id(make_config(["AAPL", "MSFT"])))

    def test_metadata_validation_rejects_mismatched_assets(self):
        metadata = checkpoint_metadata(make_config(["AAPL", "MSFT"]))
        with self.assertRaises(RuntimeError):
            validate_checkpoint_metadata(metadata, make_config(["AAPL"]))

    def test_metadata_validation_rejects_architecture_mismatch(self):
        metadata = checkpoint_metadata(make_config(["AAPL", "MSFT"]))
        config = make_config(["AAPL", "MSFT"])
        config.USE_SPARSE_NETWORK = False

        with self.assertRaises(RuntimeError):
            validate_checkpoint_metadata(metadata, config)

    def test_metadata_validation_rejects_min_save_episode_mismatch(self):
        config = make_config(["AAPL", "MSFT"])
        config.MIN_SAVE_EPISODE = 40
        metadata = checkpoint_metadata(config)
        config.MIN_SAVE_EPISODE = 5

        with self.assertRaises(RuntimeError):
            validate_checkpoint_metadata(metadata, config)

    def test_missing_drawdown_penalty_metadata_defaults_to_zero(self):
        config = make_config(["AAPL", "MSFT"])
        config.DRAWDOWN_PENALTY = 0.0
        metadata = checkpoint_metadata(config)
        metadata.pop("drawdown_penalty")

        validate_checkpoint_metadata(metadata, config)

    def test_missing_drawdown_penalty_metadata_rejects_nonzero_config(self):
        config = make_config(["AAPL", "MSFT"])
        config.DRAWDOWN_PENALTY = 1.0
        metadata = checkpoint_metadata(config)
        metadata.pop("drawdown_penalty")

        with self.assertRaises(RuntimeError):
            validate_checkpoint_metadata(metadata, config)

    def test_missing_active_overlay_metadata_defaults_to_disabled(self):
        config = make_config(["AAPL", "MSFT"])
        config.USE_ACTIVE_OVERLAY = False
        metadata = checkpoint_metadata(config)
        for key in [
            "use_active_overlay",
            "active_overlay_base_policy",
            "active_overlay_base_weight",
            "active_overlay_tilt_weight",
            "active_overlay_tracking_penalty",
        ]:
            metadata.pop(key)

        validate_checkpoint_metadata(metadata, config)

    def test_active_overlay_metadata_rejects_mismatched_mode(self):
        config = make_config(["AAPL", "MSFT"])
        config.USE_ACTIVE_OVERLAY = True
        metadata = checkpoint_metadata(config)
        config.USE_ACTIVE_OVERLAY = False

        with self.assertRaises(RuntimeError):
            validate_checkpoint_metadata(metadata, config)

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
