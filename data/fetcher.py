import yfinance as yf
import pandas as pd
import numpy as np
import logging
from typing import List

logger = logging.getLogger(__name__)


def fetch_data(
    assets: List[str],
    start_date: str,
    end_date: str,
    strict: bool = True
) -> pd.DataFrame:

    logger.info(
        f"Fetching data for {len(assets)} assets "
        f"from {start_date} to {end_date}"
    )

    try:
        raw_df = yf.download(
            tickers=assets,
            start=start_date,
            end=end_date,
            group_by="column",
            auto_adjust=True,
            progress=False,
            threads=True,
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

    features = ["Open", "High", "Low", "Close", "Volume"]
    raw_df = raw_df.loc[:, pd.IndexSlice[:, features]]
    raw_df = raw_df.asfreq("B")

    df = raw_df.dropna(how="any")

    if df.empty:
        raise ValueError("No usable data after NaN filtering")

    logger.info(
        f"Final dataset shape: {df.shape} "
        f"(rows, assets × features)"
    )
    logger.debug(f"Data preview:\n{df.head()}")

    return df


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
