import unittest
from types import SimpleNamespace

from data.splits import validate_research_dates


class ResearchSplitsTest(unittest.TestCase):
    def test_validates_chronological_non_overlapping_splits(self):
        config = SimpleNamespace(
            TRAIN_START_DATE="2010-01-01",
            TRAIN_END_DATE="2023-12-31",
            VALID_START_DATE="2024-01-01",
            VALID_END_DATE="2024-12-31",
            TEST_START_DATE="2025-01-01",
            TEST_END_DATE="2025-12-31",
        )
        validate_research_dates(config)

    def test_rejects_overlapping_splits(self):
        config = SimpleNamespace(
            TRAIN_START_DATE="2010-01-01",
            TRAIN_END_DATE="2024-06-01",
            VALID_START_DATE="2024-01-01",
            VALID_END_DATE="2024-12-31",
            TEST_START_DATE="2025-01-01",
            TEST_END_DATE="2025-12-31",
        )
        with self.assertRaises(ValueError):
            validate_research_dates(config)


if __name__ == "__main__":
    unittest.main()
