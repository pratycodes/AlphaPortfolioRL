import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

class PortfolioEnv(gym.Env):
    """
    Features:
    - Transaction Costs (Trading & Slippage)
    - Weight Constraints (Max Weight)
    - Risk-Adjusted Reward (Rolling Sharpe Ratio)
    """
    def __init__(self, df, config):
        super(PortfolioEnv, self).__init__()
        self.df = df
        self.assets = config.ASSETS
        self.n_assets = len(self.assets)
        self.window = config.WINDOW_SIZE
        self.action_space = spaces.Box(
            low=0, high=1, shape=(self.n_assets + 1,), dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(self.window, self.n_assets * 3), 
            dtype=np.float32
        )
        
        # Configuration
        self.cost_bps = config.TRADING_COST_BPS
        self.max_weight = config.MAX_WEIGHT
        
        # Risk Parameters 
        self.risk_aversion = 0.05 
        self.return_memory = [] 
        
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = self.window
        self.current_weights = np.zeros(self.n_assets + 1)
        self.current_weights[0] = 1.0 
        self.portfolio_value = 10000.0
        self.return_memory = []
        
        obs = self._get_observation(self.current_step)
        info = {
            "portfolio_value": self.portfolio_value,
            "weights": self.current_weights
        }
        return obs, info

    def step(self, action):
        # 1. Enforce max-weight cap 
        weights = np.clip(action, 0.0, 1.0)

        if self.max_weight is not None:
            weights = np.minimum(weights, self.max_weight)
        weights = weights / (weights.sum() + 1e-8)

        # 2. Transaction costs 
        turnover = np.sum(np.abs(weights - self.current_weights))
        cost = turnover * self.cost_bps
        
        # 3. Price relatives 
        prices_t = self._get_prices(self.current_step)
        prices_tp1 = self._get_prices(self.current_step + 1)

        with np.errstate(divide="ignore", invalid="ignore"):
            y = prices_tp1 / prices_t
            y = np.nan_to_num(y, nan=1.0, posinf=1.0, neginf=1.0)

        y = np.insert(y, 0, 1.0)  # cash relative is 1.0

        # 4. Portfolio update 
        raw_return = np.dot(weights, y)
        net_return = raw_return * (1.0 - cost)
        if net_return <= 1e-8:
            done = True
            reward = -10.0 
            next_obs = self._get_observation(self.current_step)
            
            self.current_weights = np.zeros_like(self.current_weights)
            self.current_weights[0] = 1.0
            
            return next_obs, reward, done, False, {
                "portfolio_value": 0.0,
                "weights": self.current_weights,
                "raw_return": -1.0
            }

        self.portfolio_value *= net_return

        self.current_weights = (weights * y) / (raw_return + 1e-8)
        self.current_weights /= self.current_weights.sum()

        # 5. Reward: Rolling Sharpe Ratio
        log_ret = np.log(net_return + 1e-8)
        self.return_memory.append(log_ret)
        
        if len(self.return_memory) > 30:
            rolling_std = np.std(self.return_memory[-30:])
        else:
            rolling_std = 0.01 
            
        reward = (log_ret - (self.risk_aversion * rolling_std)) * 100.0
        reward = np.clip(reward, -10.0, 10.0)

        #  6. Step 
        self.current_step += 1
        done = self.current_step >= len(self.df) - 1

        next_obs = self._get_observation(self.current_step)

        info = {
            "portfolio_value": self.portfolio_value,
            "weights": self.current_weights,
            "raw_return": raw_return - 1.0,
            "turnover": turnover,
            "concentration": np.sum(weights ** 2), 
        }

        return next_obs, reward, done, False, info

    def _get_prices(self, step):
        return self.df.iloc[step].xs('Close', level=1).to_numpy()

    def _get_observation(self, step):
        window_data = self.df.iloc[step-self.window:step]
        high = window_data.xs('High', level=1, axis=1)
        low = window_data.xs('Low', level=1, axis=1)
        close = window_data.xs('Close', level=1, axis=1)
        
        obs_df = pd.concat([high, low, close], axis=1)
        obs_values = obs_df.values
        start_prices = obs_values[0, :]
        obs_normalized = obs_values / (start_prices + 1e-8)
        
        return obs_normalized.astype(np.float32)