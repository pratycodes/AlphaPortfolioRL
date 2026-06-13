import tempfile
import unittest

import numpy as np
import pandas as pd

import data.fetcher as fetcher


class FetcherCacheTest(unittest.TestCase):
    def test_fetch_data_writes_and_reuses_yfinance_cache(self):
        original_download = fetcher.yf.download
        calls = {"count": 0}

        def fake_download(**kwargs):
            calls["count"] += 1
            assets = kwargs["tickers"]
            dates = pd.date_range("2025-01-01", periods=5, freq="B")
            columns = pd.MultiIndex.from_product([fetcher.FEATURES, assets])
            return pd.DataFrame(np.full((len(dates), len(columns)), 100.0), index=dates, columns=columns)

        with tempfile.TemporaryDirectory() as tmpdir:
            try:
                fetcher.yf.download = fake_download
                first = fetcher.fetch_data(
                    ["AAA", "BBB"],
                    "2025-01-01",
                    "2025-01-10",
                    cache_dir=tmpdir,
                    require_cache=False,
                )

                def fail_download(**kwargs):
                    raise AssertionError("download should not be called when cache exists")

                fetcher.yf.download = fail_download
                second = fetcher.fetch_data(["AAA", "BBB"], "2025-01-01", "2025-01-10", cache_dir=tmpdir)
            finally:
                fetcher.yf.download = original_download

        self.assertEqual(calls["count"], 1)
        pd.testing.assert_frame_equal(first, second)

    def test_fetch_data_slices_covering_cache_when_exact_cache_is_missing(self):
        dates = pd.date_range("2024-01-01", "2024-01-31", freq="B")
        columns = pd.MultiIndex.from_product([["AAA", "BBB"], fetcher.FEATURES])
        full = pd.DataFrame(
            np.arange(len(dates) * len(columns)).reshape(len(dates), len(columns)),
            index=dates,
            columns=columns,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            covering_path = fetcher._cache_path(
                "ohlcv",
                ["AAA", "BBB"],
                "2024-01-01",
                "2024-01-31",
                tmpdir,
            )
            fetcher._write_cache(
                full,
                covering_path,
                ["AAA", "BBB"],
                "2024-01-01",
                "2024-01-31",
                "ohlcv",
            )

            original_download = fetcher.yf.download

            def fail_download(**kwargs):
                raise AssertionError("download should not be called when a covering cache exists")

            try:
                fetcher.yf.download = fail_download
                sliced = fetcher.fetch_data(
                    ["BBB", "AAA"],
                    "2024-01-10",
                    "2024-01-19",
                    cache_dir=tmpdir,
                    require_cache=True,
                )
            finally:
                fetcher.yf.download = original_download

            exact_path = fetcher._cache_path(
                "ohlcv",
                ["BBB", "AAA"],
                "2024-01-10",
                "2024-01-19",
                tmpdir,
            )

            expected = full.loc["2024-01-10":"2024-01-19"]
            pd.testing.assert_frame_equal(sliced, expected)
            self.assertTrue(exact_path.exists())

    def test_fetch_close_series_can_slice_close_from_covering_ohlcv_cache(self):
        dates = pd.date_range("2024-01-01", "2024-01-31", freq="B")
        columns = pd.MultiIndex.from_product([["AAA", "^IDX"], fetcher.FEATURES])
        full = pd.DataFrame(
            np.arange(len(dates) * len(columns)).reshape(len(dates), len(columns)),
            index=dates,
            columns=columns,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            covering_path = fetcher._cache_path(
                "ohlcv",
                ["AAA", "^IDX"],
                "2024-01-01",
                "2024-01-31",
                tmpdir,
            )
            fetcher._write_cache(
                full,
                covering_path,
                ["AAA", "^IDX"],
                "2024-01-01",
                "2024-01-31",
                "ohlcv",
            )

            original_download = fetcher.yf.download

            def fail_download(*args, **kwargs):
                raise AssertionError("download should not be called when OHLCV cache contains close data")

            try:
                fetcher.yf.download = fail_download
                close = fetcher.fetch_close_series(
                    "^IDX",
                    "2024-01-10",
                    "2024-01-19",
                    cache_dir=tmpdir,
                    require_cache=True,
                )
            finally:
                fetcher.yf.download = original_download

            exact_path = fetcher._cache_path(
                "close",
                ["^IDX"],
                "2024-01-10",
                "2024-01-19",
                tmpdir,
            )
            expected = full.loc["2024-01-10":"2024-01-19", ("^IDX", "Close")]
            expected.name = "^IDX"
            pd.testing.assert_series_equal(close, expected)
            self.assertTrue(exact_path.exists())


if __name__ == "__main__":
    unittest.main()
