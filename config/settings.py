from typing import List
from pydantic_settings import BaseSettings
import torch

class Settings(BaseSettings):
    SEED: int = 42
    
    # Device
    if torch.cuda.is_available():
        DEVICE: str = "cuda"
    elif torch.backends.mps.is_available():
        DEVICE: str = "mps"
    else:
        DEVICE: str = "cpu"
        
    # Data
    ASSETS: List[str] = ["AAPL", "MSFT", "AMZN", "GOOGL", "META", "TSLA", "JPM"]
    TRAIN_START_DATE: str = "2010-01-01"
    TRAIN_END_DATE: str = "2024-12-31"
    TEST_START_DATE: str = "2025-01-01"
    TEST_END_DATE: str = "2025-12-31"
    WINDOW_SIZE: int = 10

    # Training
    EPISODES: int = 100
    BATCH_SIZE: int = 64
    BUFFER_SIZE: int = 100000
    GAMMA: float = 0.99
    TAU: float = 0.001
    LR_ACTOR: float = 1e-4
    LR_CRITIC: float = 1e-3

    # Exploration
    INIT_NOISE: float = 0.05
    MIN_NOISE: float = 0.01

    # Oracle
    ORACLE_ANNEAL_EPISODES: int = 100
    BCM_LAMBDA: float = 0.1

    # Modules
    USE_IPM: bool = True
    USE_BCM: bool = True
    USE_DAM: bool = True

    # Costs
    TRADING_COST_BPS: float = 0.0020
    SLIPPAGE_BPS: float = 0.0050

    # DAM (GAN)
    GAN_NOISE_DIM: int = 10
    GAN_HIDDEN_DIM: int = 32
    GAN_SEQ_LEN: int = 10
    GAN_LR: float = 0.001
    GAN_EPOCHS: int = 200
    
    #IPM
    IPM_PRETRAIN_EPOCHS: int = 50

    # Risk Regularization
    TURNOVER_PENALTY: float = 1e-3
    CONCENTRATION_PENALTY: float = 1e-4
    MAX_WEIGHT: float = 0.60

    # Checkpointing
    CHECKPOINT_FREQ: int = 5

    class Config:
        env_file = ".env"


config = Settings()
