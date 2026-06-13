import numpy as np
import pandas as pd


def project_to_simplex(values):
    values = np.asarray(values, dtype=float)
    if values.ndim != 1:
        raise ValueError("Simplex projection expects a 1D vector")

    sorted_values = np.sort(values)[::-1]
    cssv = np.cumsum(sorted_values) - 1.0
    indices = np.arange(1, len(values) + 1)
    support = sorted_values - cssv / indices > 0
    if not np.any(support):
        return np.full_like(values, 1.0 / len(values))

    rho = indices[support][-1]
    theta = cssv[support][-1] / rho
    projected = np.maximum(values - theta, 0.0)
    return projected / (projected.sum() + 1e-12)


def fit_benchmark_proxy_weights(asset_returns, benchmark_returns, ridge=1e-4, steps=2500):
    asset_returns = np.asarray(asset_returns, dtype=float)
    benchmark_returns = np.asarray(benchmark_returns, dtype=float)
    if asset_returns.ndim != 2:
        raise ValueError("asset_returns must be a 2D array")
    if benchmark_returns.ndim != 1:
        raise ValueError("benchmark_returns must be a 1D array")
    if len(asset_returns) != len(benchmark_returns):
        raise ValueError("asset_returns and benchmark_returns must have matching rows")
    if len(asset_returns) == 0:
        raise ValueError("Cannot fit benchmark proxy on an empty return matrix")

    n_assets = asset_returns.shape[1]
    weights = np.full(n_assets, 1.0 / n_assets)
    scale = np.linalg.norm(asset_returns, ord=2) ** 2 / max(len(asset_returns), 1)
    learning_rate = 1.0 / (scale + float(ridge) + 1e-8)

    for _ in range(int(steps)):
        residual = asset_returns @ weights - benchmark_returns
        gradient = (asset_returns.T @ residual) / len(asset_returns)
        gradient += float(ridge) * weights
        weights = project_to_simplex(weights - learning_rate * gradient)

    return weights


def fit_benchmark_proxy_from_close(close, assets, benchmark_ticker, ridge=1e-4, steps=2500):
    close = close.loc[:, list(assets) + [benchmark_ticker]].dropna(how="any")
    returns = close.pct_change().dropna(how="any")
    if returns.empty:
        raise ValueError("Cannot fit benchmark proxy without overlapping returns")

    asset_returns = returns.loc[:, list(assets)]
    benchmark_returns = returns.loc[:, benchmark_ticker]
    weights = fit_benchmark_proxy_weights(
        asset_returns.to_numpy(dtype=float),
        benchmark_returns.to_numpy(dtype=float),
        ridge=ridge,
        steps=steps,
    )
    fitted = pd.Series(weights, index=list(assets), name="benchmark_proxy_weight")
    return fitted
