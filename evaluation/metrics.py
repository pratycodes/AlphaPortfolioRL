import numpy as np
import pandas as pd

class FinancialMetrics:
    @staticmethod
    def get_metrics(returns: np.array, risk_free_rate: float = 0.0):
        returns = np.array(returns)
        
        # 1. Total Return
        total_return = np.prod(1 + returns) - 1
        
        # 2. Volatility (Annualized)
        volatility = np.std(returns) * np.sqrt(252)
        
        # 3. Sharpe Ratio
        mean_return = np.mean(returns)
        if np.std(returns) == 0:
            sharpe = 0
        else:
            sharpe = (mean_return - (risk_free_rate/252)) / np.std(returns) * np.sqrt(252)
            
        # 4. Sortino Ratio
        downside_returns = returns[returns < 0]
        if len(downside_returns) == 0 or np.std(downside_returns) == 0:
            sortino = 0
        else:
            downside_std = np.std(downside_returns)
            sortino = (mean_return - (risk_free_rate/252)) / downside_std * np.sqrt(252)
            
        # 5. Maximum Drawdown (MDD)
        cum_returns = np.cumprod(1 + returns)
        peak = np.maximum.accumulate(cum_returns)
        drawdown = (cum_returns - peak) / peak
        max_drawdown = np.min(drawdown)
        
        return {
            "Total Return": total_return,
            "Annual Volatility": volatility,
            "Sharpe Ratio": sharpe,
            "Sortino Ratio": sortino,
            "Max Drawdown": max_drawdown
        }