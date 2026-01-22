import numpy as np
import torch
import os

from config.settings import config
from env.portfolio_env import PortfolioEnv
from agent.ddpg_agent import DDPGAgent
from agent.replay_buffer import ReplayBuffer
from models.ipm import IPM
from optimization.oracle import PortfolioOracle
from data.fetcher import fetch_data
from data.train_dam import train_gan
from data.dam_gan import RGAN_Generator
from models.ipm import IPM
from utils.logger import setup_logger

logger = setup_logger()

def train():
    # Setup
    np.random.seed(config.SEED)
    torch.manual_seed(config.SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.SEED)
    if not os.path.exists("models"):
        os.makedirs("models")

    if config.USE_DAM:
        if not os.path.exists("models/dam_generator.pth"):
            train_gan()

    logger.info(f"Fetching Training Data ({config.TRAIN_START_DATE} -> {config.TRAIN_END_DATE})")
    df = fetch_data(config.ASSETS, config.TRAIN_START_DATE, config.TRAIN_END_DATE)

    env = PortfolioEnv(df, config)
    agent = DDPGAgent(num_assets=len(config.ASSETS), window_size=config.WINDOW_SIZE)
    buffer = ReplayBuffer(capacity=config.BUFFER_SIZE, batch_size=config.BATCH_SIZE, device=agent.device)
    ipm = IPM(num_assets=len(config.ASSETS), window_size=config.WINDOW_SIZE).to(agent.device)
    oracle = PortfolioOracle(num_assets=len(config.ASSETS) + 1, transaction_cost_bps=config.TRADING_COST_BPS)

    # Load GAN
    gan = None
    if config.USE_DAM:
        gan = RGAN_Generator(config.GAN_NOISE_DIM, config.GAN_HIDDEN_DIM, config.GAN_SEQ_LEN, len(config.ASSETS))
        try:
            gan.load_state_dict(torch.load("models/dam_generator.pth", map_location=agent.device))
            gan.eval()
            logger.info("DAM Generator Loaded.")
        except:
            gan = None

    # Pre-train IPM
    if config.USE_IPM:
        IPM.pretrain_ipm(ipm, df, agent.device, epochs=config.IPM_PRETRAIN_EPOCHS)

    logger.info("Starting RL Training.")
    
    best_value = -np.inf
    ema_value = 10000.0  
    MIN_SAVE_EPISODE = 20  
    
    for episode in range(config.EPISODES):
        obs, _ = env.reset()
        done = False
        prev_weights = np.zeros(len(config.ASSETS) + 1); prev_weights[0] = 1.0
        episode_reward = 0.0
        
        # Annealing
        noise_scale = max(config.MIN_NOISE, config.INIT_NOISE * (1.0 - episode / config.EPISODES))
        oracle_weight = max(0.0, 1.0 - episode / config.ORACLE_ANNEAL_EPISODES)

        if gan and episode % 5 == 0: 
            with torch.no_grad():
                noise = torch.randn(64, config.GAN_SEQ_LEN, config.GAN_NOISE_DIM).to(agent.device)
                fake_returns = gan(noise) 
                fake_state = fake_returns.repeat(1, 1, 3) 
                for i in range(len(fake_state)):
                    s = fake_state[i].cpu().numpy()
                    dummy_w = np.zeros(len(config.ASSETS) + 1); dummy_w[0] = 1.0
                    buffer.add(s, dummy_w, dummy_w, 0.0, s, dummy_w, dummy_w)

        while not done:
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(agent.device)
            prev_w_tensor = torch.FloatTensor(prev_weights).unsqueeze(0).to(agent.device)

            with torch.no_grad():
                ipm_pred = ipm(obs_tensor)
                action = agent.actor(obs_tensor, prev_w_tensor, ipm_pred).cpu().numpy()[0]

            noise = np.random.normal(0, noise_scale, size=action.shape)
            action = np.clip(action + noise, 0, 1)
            action /= (action.sum() + 1e-8)

            price_t = env._get_prices(env.current_step)
            next_obs, reward, done, _, info = env.step(action)
            next_prev_weights = info["weights"]

            if not done:
                price_tp1 = env._get_prices(env.current_step)
                with np.errstate(divide="ignore", invalid="ignore"):
                    y = np.nan_to_num(price_tp1 / price_t, nan=1.0)
                y_vec = np.insert(y, 0, 1.0)
                opt_action = oracle.get_optimal_weights(prev_weights, y_vec)
                greedy_action = (oracle_weight * opt_action + (1.0 - oracle_weight) * action)
            else:
                greedy_action = action

            buffer.add(obs, prev_weights, action, reward, next_obs, next_prev_weights, greedy_action)
            
            if len(buffer) > config.BATCH_SIZE:
                batch = buffer.sample()
                agent.update(batch, ipm)

            obs = next_obs
            prev_weights = next_prev_weights
            episode_reward += reward

        # EMA filter for Model Saving
        ema_value = 0.9 * ema_value + 0.1 * env.portfolio_value
        
        # Diagnostic Logging
        logger.info(
            f"Ep {episode+1:03d} | "
            f"Rew: {episode_reward:7.2f} | "
            f"Val: {env.portfolio_value:10.2f} | "
            f"EMA: {ema_value:10.2f} | " 
            f"Noise: {noise_scale:.3f} | "
            f"Oracle: {oracle_weight:.2f}"
        )

        if (episode + 1) % config.CHECKPOINT_FREQ == 0:
            torch.save(agent.actor.state_dict(), f"models/actor_ep_{episode+1}.pth")
            torch.save(ipm.state_dict(), f"models/ipm_ep_{episode+1}.pth")

    
        if (episode + 1) >= MIN_SAVE_EPISODE:
            if ema_value > best_value:
                best_value = ema_value
                
                torch.save(agent.actor.state_dict(), "models/best_actor.pth")
                torch.save(ipm.state_dict(), "models/best_ipm.pth")
                
                logger.info(f"Best Model Saved: EMA Value: {best_value:.2f}")

if __name__ == "__main__":
    train()