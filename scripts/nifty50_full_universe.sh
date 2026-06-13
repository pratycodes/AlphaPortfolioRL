#!/usr/bin/env bash

# Long-history 50-stock NIFTY-style universe for 2010-start experiments.
# This intentionally avoids newer current NIFTY names with short public histories
# because the data loader drops rows with any missing ticker data.
export NIFTY50_FULL_ASSETS="${NIFTY50_FULL_ASSETS:-[\"RELIANCE.NS\",\"HDFCBANK.NS\",\"ICICIBANK.NS\",\"INFY.NS\",\"TCS.NS\",\"BHARTIARTL.NS\",\"LT.NS\",\"ITC.NS\",\"SBIN.NS\",\"HINDUNILVR.NS\",\"KOTAKBANK.NS\",\"AXISBANK.NS\",\"BAJFINANCE.NS\",\"BAJAJFINSV.NS\",\"M&M.NS\",\"MARUTI.NS\",\"SUNPHARMA.NS\",\"TATASTEEL.NS\",\"POWERGRID.NS\",\"NTPC.NS\",\"ONGC.NS\",\"COALINDIA.NS\",\"ULTRACEMCO.NS\",\"HCLTECH.NS\",\"WIPRO.NS\",\"TECHM.NS\",\"TITAN.NS\",\"NESTLEIND.NS\",\"ASIANPAINT.NS\",\"GRASIM.NS\",\"JSWSTEEL.NS\",\"HINDALCO.NS\",\"ADANIPORTS.NS\",\"ADANIENT.NS\",\"CIPLA.NS\",\"DRREDDY.NS\",\"EICHERMOT.NS\",\"HEROMOTOCO.NS\",\"BAJAJ-AUTO.NS\",\"TATAMOTORS.NS\",\"BPCL.NS\",\"BRITANNIA.NS\",\"DIVISLAB.NS\",\"UPL.NS\",\"VEDL.NS\",\"TATAPOWER.NS\",\"HINDPETRO.NS\",\"IOC.NS\",\"GAIL.NS\",\"DLF.NS\"]}"

export ASSETS="${ASSETS:-${NIFTY50_FULL_ASSETS}}"
export MARKET_FEATURE_TICKER="${MARKET_FEATURE_TICKER:-^NSEI}"
export BENCHMARK_TICKER="${BENCHMARK_TICKER:-^NSEI}"
export BENCHMARK_NAME="${BENCHMARK_NAME:-NIFTY 50}"
export DATA_CACHE_DIR="${DATA_CACHE_DIR:-data_cache/nifty50_full}"
