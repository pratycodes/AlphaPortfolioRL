def feature_assets(config):
    return list(config.ASSETS)


def data_tickers(config):
    assets = list(config.ASSETS)
    if getattr(config, "USE_MARKET_FEATURE", False):
        market_ticker = getattr(config, "MARKET_FEATURE_TICKER", None)
        if market_ticker and market_ticker not in assets:
            assets.append(market_ticker)
    return assets


def feature_dim(config):
    asset_count = len(config.ASSETS) + 1  # cash + risky assets
    if getattr(config, "USE_MARKET_FEATURE", False):
        asset_count += 1
    return asset_count * 3


def ipm_feature_dim(config):
    return len(config.ASSETS) * 3


def market_feature_dim(config):
    return 1 if getattr(config, "USE_MARKET_FEATURE", False) else 0
