import logging
import hashlib
import json
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import yfinance as yf

from config.settings import config

logger = logging.getLogger(__name__)

FEATURES = ["Open", "High", "Low", "Close", "Volume"]


def fetch_data(
    assets: List[str],
    start_date: str,
    end_date: str,
    strict: bool = True,
    cache_dir: str | None = None,
    use_cache: bool | None = None,
    refresh_cache: bool | None = None,
    require_cache: bool | None = None,
) -> pd.DataFrame:
    cache_dir = cache_dir or config.DATA_CACHE_DIR
    use_cache = config.USE_DATA_CACHE if use_cache is None else use_cache
    refresh_cache = config.REFRESH_DATA_CACHE if refresh_cache is None else refresh_cache
    require_cache = config.REQUIRE_DATA_CACHE if require_cache is None else require_cache
    cache_path = _cache_path("ohlcv", assets, start_date, end_date, cache_dir)

    logger.info(
        f"Fetching data for {len(assets)} assets "
        f"from {start_date} to {end_date}"
    )

    if use_cache and not refresh_cache and cache_path.exists():
        logger.info(f"Loading Yahoo Finance data from cache: {cache_path}")
        return pd.read_pickle(cache_path)

    if use_cache and not refresh_cache:
        covering_cache = _read_covering_cache("ohlcv", assets, start_date, end_date, cache_dir)
        if covering_cache is not None:
            logger.info(f"Loading Yahoo Finance data from covering cache for {start_date} -> {end_date}")
            _write_cache(covering_cache, cache_path, assets, start_date, end_date, "ohlcv")
            return covering_cache

    if require_cache and not refresh_cache:
        raise FileNotFoundError(
            f"Required data cache is missing: {cache_path}. "
            "Run `python -m data.bootstrap_paper_data` or pass refresh_cache=True."
        )

    try:
        raw_df = yf.download(
            tickers=assets,
            start=start_date,
            end=end_date,
            group_by="column",
            auto_adjust=True,
            progress=False,
            threads=False,
        )
    except Exception as e:
        logger.exception("Data download failed")
        raise RuntimeError("Data download failed") from e

    if raw_df.empty:
        raise ValueError("Downloaded data is empty")

    raw_df.columns = raw_df.columns.swaplevel(0, 1)
    raw_df.sort_index(axis=1, level=0, inplace=True)

    fetched_assets = raw_df.columns.get_level_values(0).unique()
    missing = set(assets) - set(fetched_assets)

    if missing:
        msg = f"Missing assets from Yahoo Finance: {missing}"
        if strict:
            raise ValueError(msg)
        logger.warning(msg)

    raw_df = raw_df.loc[:, pd.IndexSlice[:, FEATURES]]
    raw_df = raw_df.asfreq("B")

    df = raw_df.dropna(how="any")

    if df.empty:
        raise ValueError("No usable data after NaN filtering")

    logger.info(
        f"Final dataset shape: {df.shape} "
        f"(rows, assets × features)"
    )
    logger.debug(f"Data preview:\n{df.head()}")

    if use_cache:
        _write_cache(df, cache_path, assets, start_date, end_date, "ohlcv")

    return df


def fetch_close_series(
    ticker: str,
    start_date: str,
    end_date: str,
    cache_dir: str | None = None,
    use_cache: bool | None = None,
    refresh_cache: bool | None = None,
    require_cache: bool | None = None,
) -> pd.Series:
    cache_dir = cache_dir or config.DATA_CACHE_DIR
    use_cache = config.USE_DATA_CACHE if use_cache is None else use_cache
    refresh_cache = config.REFRESH_DATA_CACHE if refresh_cache is None else refresh_cache
    require_cache = config.REQUIRE_DATA_CACHE if require_cache is None else require_cache
    cache_path = _cache_path("close", [ticker], start_date, end_date, cache_dir)

    if use_cache and not refresh_cache and cache_path.exists():
        logger.info(f"Loading Yahoo Finance close data from cache: {cache_path}")
        return pd.read_pickle(cache_path)

    if use_cache and not refresh_cache:
        covering_cache = _read_covering_cache("close", [ticker], start_date, end_date, cache_dir)
        if covering_cache is not None:
            logger.info(f"Loading Yahoo Finance close data from covering cache for {start_date} -> {end_date}")
            _write_cache(covering_cache, cache_path, [ticker], start_date, end_date, "close")
            return covering_cache

        ohlcv_close = _read_close_from_covering_ohlcv_cache(ticker, start_date, end_date, cache_dir)
        if ohlcv_close is not None:
            logger.info(f"Loading Yahoo Finance close data from OHLCV covering cache for {start_date} -> {end_date}")
            _write_cache(ohlcv_close, cache_path, [ticker], start_date, end_date, "close")
            return ohlcv_close

    if require_cache and not refresh_cache:
        raise FileNotFoundError(
            f"Required benchmark cache is missing: {cache_path}. "
            "Run `python -m data.bootstrap_paper_data` or pass refresh_cache=True."
        )

    raw = yf.download(ticker, start=start_date, end=end_date, progress=False, auto_adjust=True)
    if raw.empty:
        raise ValueError(f"Downloaded benchmark data is empty for {ticker}")

    close = raw["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    close = close.dropna()
    close.name = ticker

    if use_cache:
        _write_cache(close, cache_path, [ticker], start_date, end_date, "close")

    return close


def _cache_path(kind, assets, start_date, end_date, cache_dir):
    payload = {
        "kind": kind,
        "assets": sorted(list(assets)),
        "start_date": start_date,
        "end_date": end_date,
    }
    digest = hashlib.sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:12]
    label = "_".join(_safe_label(asset) for asset in payload["assets"])
    label = label[:80] if label else "data"
    return Path(cache_dir) / f"{kind}_{label}_{start_date}_{end_date}_{digest}.pkl"


def _write_cache(data, path, assets, start_date, end_date, kind):
    path.parent.mkdir(parents=True, exist_ok=True)
    data.to_pickle(path)
    metadata = {
        "kind": kind,
        "assets": list(assets),
        "start_date": start_date,
        "end_date": end_date,
        "rows": int(len(data)),
        "columns": [str(column) for column in getattr(data, "columns", [])],
    }
    with path.with_suffix(".json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)


def _read_covering_cache(kind, assets, start_date, end_date, cache_dir):
    cache_dir = Path(cache_dir)
    if not cache_dir.exists():
        return None

    requested_assets = sorted(list(assets))
    requested_start = pd.Timestamp(start_date)
    requested_end = pd.Timestamp(end_date)
    candidates = []

    for metadata_path in cache_dir.glob(f"{kind}_*.json"):
        try:
            with metadata_path.open("r", encoding="utf-8") as handle:
                metadata = json.load(handle)
            cache_start = pd.Timestamp(metadata["start_date"])
            cache_end = pd.Timestamp(metadata["end_date"])
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            continue

        if metadata.get("kind") != kind:
            continue
        if sorted(metadata.get("assets", [])) != requested_assets:
            continue
        if cache_start > requested_start or cache_end < requested_end:
            continue

        data_path = metadata_path.with_suffix(".pkl")
        if data_path.exists():
            span = cache_end - cache_start
            candidates.append((span, data_path))

    if not candidates:
        return None

    _, data_path = min(candidates, key=lambda candidate: candidate[0])
    data = pd.read_pickle(data_path)
    if not isinstance(data.index, pd.DatetimeIndex):
        return None

    sliced = data.loc[(data.index >= requested_start) & (data.index <= requested_end)].copy()
    if sliced.empty:
        return None
    return sliced


def _read_close_from_covering_ohlcv_cache(ticker, start_date, end_date, cache_dir):
    requested_start = pd.Timestamp(start_date)
    requested_end = pd.Timestamp(end_date)
    candidates = []

    for metadata_path in Path(cache_dir).glob("ohlcv_*.json"):
        try:
            with metadata_path.open("r", encoding="utf-8") as handle:
                metadata = json.load(handle)
            cache_start = pd.Timestamp(metadata["start_date"])
            cache_end = pd.Timestamp(metadata["end_date"])
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            continue

        if metadata.get("kind") != "ohlcv":
            continue
        if ticker not in metadata.get("assets", []):
            continue
        if cache_start > requested_start or cache_end < requested_end:
            continue

        data_path = metadata_path.with_suffix(".pkl")
        if data_path.exists():
            candidates.append((cache_end - cache_start, data_path))

    if not candidates:
        return None

    _, data_path = min(candidates, key=lambda candidate: candidate[0])
    data = pd.read_pickle(data_path)
    if not isinstance(data.index, pd.DatetimeIndex):
        return None
    if not isinstance(data.columns, pd.MultiIndex):
        return None
    if (ticker, "Close") not in data.columns:
        return None

    close = data.loc[(data.index >= requested_start) & (data.index <= requested_end), (ticker, "Close")].copy()
    if close.empty:
        return None
    close.name = ticker
    return close


def _safe_label(value):
    return "".join(char if char.isalnum() else "-" for char in str(value)).strip("-")


def preprocess_for_rl(
    df: pd.DataFrame,
    price_feature: str = "Close"
) -> pd.DataFrame:
    """
    Computes price relatives for RL environments.

    u_t = p_t / p_{t-1}

    Output:
        Columns: MultiIndex (Ticker, PriceRelative)
        Index:   t
    """

    prices = df.xs(price_feature, axis=1, level=1)

    if prices.isna().any().any():
        raise ValueError("NaNs present in price series")

    price_relatives = prices / prices.shift(1)
    price_relatives = price_relatives.dropna()

    price_relatives.columns = pd.MultiIndex.from_product(
        [price_relatives.columns, ["PriceRelative"]],
        names=["Ticker", "Feature"]
    )

    return price_relatives
