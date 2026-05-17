import unittest
from types import SimpleNamespace

from utils.checkpoints import checkpoint_metadata, universe_id, validate_checkpoint_metadata


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


if __name__ == "__main__":
    unittest.main()
