import argparse
import json
from pathlib import Path

from config.settings import config
from data.fetcher import fetch_close_series, fetch_data
from data.universe import data_tickers
from utils.logger import setup_logger


logger = setup_logger(name="data_bootstrap")


def _date_ranges(cfg):
    return [
        ("train", cfg.TRAIN_START_DATE, cfg.TRAIN_END_DATE),
        ("validation", cfg.VALID_START_DATE, cfg.VALID_END_DATE),
        ("test", cfg.TEST_START_DATE, cfg.TEST_END_DATE),
    ]


def bootstrap(refresh=False, cache_dir=None):
    cache_dir = cache_dir or config.DATA_CACHE_DIR
    tickers = data_tickers(config)
    manifest = {
        "source": "Yahoo Finance",
        "cache_dir": cache_dir,
        "assets": list(config.ASSETS),
        "market_feature_ticker": config.MARKET_FEATURE_TICKER,
        "benchmark_ticker": config.BENCHMARK_TICKER,
        "ranges": [],
    }

    for name, start, end in _date_ranges(config):
        logger.info(f"Bootstrapping {name} OHLCV cache ({start} -> {end})")
        df = fetch_data(
            tickers,
            start,
            end,
            cache_dir=cache_dir,
            use_cache=True,
            refresh_cache=refresh,
            require_cache=False,
        )
        manifest["ranges"].append(
            {
                "name": name,
                "start_date": start,
                "end_date": end,
                "rows": int(len(df)),
                "columns": [str(column) for column in df.columns],
            }
        )

    logger.info(f"Bootstrapping benchmark close cache ({config.BENCHMARK_TICKER})")
    benchmark = fetch_close_series(
        config.BENCHMARK_TICKER,
        config.TEST_START_DATE,
        config.TEST_END_DATE,
        cache_dir=cache_dir,
        use_cache=True,
        refresh_cache=refresh,
        require_cache=False,
    )
    manifest["benchmark_rows"] = int(len(benchmark))

    path = Path(cache_dir) / "paper_data_manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
    logger.info(f"Wrote paper data manifest to {path}")


def main():
    parser = argparse.ArgumentParser(description="Download and cache all data needed for paper-faithful runs.")
    parser.add_argument("--refresh", action="store_true", help="Refresh cached Yahoo Finance data.")
    parser.add_argument("--cache-dir", default=None, help="Override cache directory.")
    args = parser.parse_args()
    bootstrap(refresh=args.refresh, cache_dir=args.cache_dir)


if __name__ == "__main__":
    main()
