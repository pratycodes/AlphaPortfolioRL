import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os 
from config.settings import config
from env.portfolio_env import PortfolioEnv
from agent.ddpg_agent import DDPGAgent
from models.ipm import IPM
from data.fetcher import fetch_data
from evaluation.metrics import FinancialMetrics
from utils.logger import setup_logger
import yfinance as yf

logger = setup_logger(name="tearsheet")

def fetch_sp500(start_date, end_date):
    logger.info("Fetching S&P 500 Benchmark Data")
    sp500 = yf.download("^GSPC", start=start_date, end=end_date, progress=False, auto_adjust=True)
    sp500['Return'] = sp500['Close'].pct_change().fillna(0)
    sp500['Value'] = 10000 * (1 + sp500['Return']).cumprod()
    return sp500

def run_tearsheet(checkpoint_path: str, ipm_path: str):
    df = fetch_data(config.ASSETS, config.TEST_START_DATE, config.TEST_END_DATE)
    sp500_df = fetch_sp500(config.TEST_START_DATE, config.TEST_END_DATE)
    
    env = PortfolioEnv(df, config)
    agent = DDPGAgent(len(config.ASSETS), config.WINDOW_SIZE)
    ipm = IPM(len(config.ASSETS), config.WINDOW_SIZE).to(agent.device)
    
    
    logger.info(f"Loading Actor weights: {checkpoint_path}")
    agent.actor.load_state_dict(torch.load(checkpoint_path, map_location=agent.device))
    
    if os.path.exists(ipm_path):
        logger.info(f"Loading IPM weights: {ipm_path}")
        ipm.load_state_dict(torch.load(ipm_path, map_location=agent.device))
    else:
        logger.error(f"IPM checkpoint not found at {ipm_path}!")
        return 

    agent.actor.eval()
    ipm.eval()

    obs, _ = env.reset()
    done = False
    prev_weights = np.zeros(len(config.ASSETS) + 1)
    prev_weights[0] = 1.0
    
    rl_values = [env.portfolio_value]
    rl_weights = [prev_weights]
    
    # --- Benchmark 1: Equal Weight Portfolio ---
    n_assets = len(config.ASSETS)
    bench_weights = np.array([0.0] + [1.0/n_assets]*n_assets)
    bench_value = 10000.0
    bench_values = [bench_value]
    
    while not done:
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(agent.device)
        prev_w_tensor = torch.FloatTensor(prev_weights).unsqueeze(0).to(agent.device)
        
        with torch.no_grad():
            ipm_pred = ipm(obs_tensor)
            action = agent.actor(obs_tensor, prev_w_tensor, ipm_pred).cpu().numpy()[0]
            
        next_obs, reward, done, _, info = env.step(action)
        rl_values.append(info['portfolio_value'])
        rl_weights.append(info['weights'])
        
        idx = env.current_step - 1 
        curr_p = env._get_prices(idx-1)
        next_p = env._get_prices(idx)
        with np.errstate(divide='ignore', invalid='ignore'):
            y = next_p / curr_p
            y = np.nan_to_num(y, nan=1.0)
        y = np.insert(y, 0, 1.0) 
        
        bench_return = np.dot(bench_weights, y)
        bench_value *= bench_return
        bench_values.append(bench_value)
        
        obs = next_obs
        prev_weights = info['weights']
    
    # Metrics Calculation
    rl_returns = np.diff(rl_values) / rl_values[:-1]
    bench_returns = np.diff(bench_values) / bench_values[:-1]
    
    min_len = min(len(rl_returns), len(sp500_df['Return']))
    sp500_vals = sp500_df['Value'].values[:min_len+1] 
    sp500_returns = sp500_df['Return'].values[:min_len]
    
    rl_metrics = FinancialMetrics.get_metrics(rl_returns[:min_len])
    bench_metrics = FinancialMetrics.get_metrics(bench_returns[:min_len])
    sp500_metrics = FinancialMetrics.get_metrics(sp500_returns)
    

    logger.info(f"{'Metric':<20} | {'RL Agent':<15} | {'Eq-Weight':<15} | {'S&P 500':<15}")
    logger.info("-" * 75)
    for k in rl_metrics.keys():
        logger.info(f"{k:<20} | {rl_metrics[k]:<15.4f} | {bench_metrics[k]:<15.4f} | {sp500_metrics[k]:<15.4f}")
        

    fig, axes = plt.subplots(2, 1, figsize=(12, 12))
    
    ax1 = axes[0]
    ax1.plot(rl_values, label='RL Agent (Model)', color='blue', linewidth=2)
    ax1.plot(bench_values, label='Equal Weight Benchmark', color='gray', linestyle='--')
    
    ax1.plot(sp500_vals, label='S&P 500 Index', color='red', linestyle='-.', alpha=0.8)
    
    ax1.set_title('Cumulative Return: Agent vs Benchmarks (2017-2018)')
    ax1.set_ylabel('Portfolio Value ($)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2 = axes[1]
    w_array = np.array(rl_weights).T 
    labels = ["Cash"] + config.ASSETS
    ax2.stackplot(range(len(rl_values)), w_array, labels=labels, alpha=0.8)
    ax2.set_title('Agent Asset Allocation Over Time')
    ax2.set_ylabel('Weight')
    ax2.set_xlabel('Trading Steps')
    ax2.legend(loc='upper left', bbox_to_anchor=(1, 1))
    
    plt.tight_layout()
    plt.savefig('assets/dashboard_sp500.png')
    logger.info("Saved comparison analysis to tearsheet_sp500.png")

if __name__ == "__main__":
    checkpoint = "models/best_actor.pth"  
    ipm = "models/best_ipm.pth"        
    run_tearsheet(checkpoint, ipm)
