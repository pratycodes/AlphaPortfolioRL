#!/usr/bin/env python
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config.settings import config
from data.fetcher import fetch_data
from data.universe import data_tickers
from utils.benchmark_proxy import fit_benchmark_proxy_from_close


def main():
    parser = argparse.ArgumentParser(description="Fit long-only proxy weights to the configured benchmark.")
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--ridge", type=float, default=1e-4)
    parser.add_argument("--steps", type=int, default=2500)
    parser.add_argument("--include-cash", action="store_true")
    args = parser.parse_args()

    start_date = args.start_date or config.TRAIN_START_DATE
    end_date = args.end_date or config.TRAIN_END_DATE
    tickers = data_tickers(config)
    if config.BENCHMARK_TICKER not in tickers:
        tickers.append(config.BENCHMARK_TICKER)
    df = fetch_data(tickers, start_date, end_date)
    close = df.xs("Close", level=1, axis=1)
    weights = fit_benchmark_proxy_from_close(
        close,
        config.ASSETS,
        config.BENCHMARK_TICKER,
        ridge=args.ridge,
        steps=args.steps,
    )

    values = [float(value) for value in weights.to_numpy()]
    if args.include_cash:
        values = [0.0] + values
    print(json.dumps(values))


if __name__ == "__main__":
    main()
