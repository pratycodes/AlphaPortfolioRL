import numpy as np


class FinancialMetrics:
    @staticmethod
    def get_metrics(returns: np.array, risk_free_rate: float = 0.0):
        returns = np.asarray(returns, dtype=float)

        if returns.size == 0:
            return {
                "Total Return": 0.0,
                "Annual Volatility": 0.0,
                "Sharpe Ratio": 0.0,
                "Sortino Ratio": 0.0,
                "VaR 95": 0.0,
                "CVaR 95": 0.0,
                "Max Drawdown": 0.0,
            }

        total_return = np.prod(1 + returns) - 1
        volatility = np.std(returns) * np.sqrt(252)

        mean_return = np.mean(returns)
        if np.std(returns) == 0:
            sharpe = 0
        else:
            sharpe = (mean_return - (risk_free_rate / 252)) / np.std(returns) * np.sqrt(252)

        downside_returns = returns[returns < 0]
        if len(downside_returns) == 0 or np.std(downside_returns) == 0:
            sortino = 0
        else:
            downside_std = np.std(downside_returns)
            sortino = (mean_return - (risk_free_rate / 252)) / downside_std * np.sqrt(252)

        equity = np.concatenate(([1.0], np.cumprod(1 + returns)))
        peak = np.maximum.accumulate(equity)
        drawdown = (equity - peak) / peak
        max_drawdown = np.min(drawdown)
        var_95 = -np.percentile(returns, 5)
        tail = returns[returns <= np.percentile(returns, 5)]
        cvar_95 = -np.mean(tail) if tail.size else 0.0

        return {
            "Total Return": total_return,
            "Annual Volatility": volatility,
            "Sharpe Ratio": sharpe,
            "Sortino Ratio": sortino,
            "VaR 95": var_95,
            "CVaR 95": cvar_95,
            "Max Drawdown": max_drawdown,
        }
