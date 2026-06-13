from typing import List
from pydantic import field_validator
from pydantic_settings import BaseSettings
import torch

class Settings(BaseSettings):
    SEED: int = 42
    ABLATION: str = "default"
    
    # Device
    if torch.cuda.is_available():
        DEVICE: str = "cuda"
    elif torch.backends.mps.is_available():
        DEVICE: str = "mps"
    else:
        DEVICE: str = "cpu"
        
    # Data
    # Yu et al. (2019) stock universe: COST, CSCO, F, GS, AIG, CAT + cash.
    ASSETS: List[str] = ["COST", "CSCO", "F", "GS", "AIG", "CAT"]
    USE_MARKET_FEATURE: bool = True
    MARKET_FEATURE_TICKER: str = "^GSPC"

    TRAIN_START_DATE: str = "2010-01-01"
    TRAIN_END_DATE: str = "2023-12-31"
    VALID_START_DATE: str = "2024-01-01"
    VALID_END_DATE: str = "2024-12-31"
    TEST_START_DATE: str = "2025-01-01"
    TEST_END_DATE: str = "2026-05-27"
    BENCHMARK_TICKER: str = "^GSPC"
    BENCHMARK_NAME: str = "S&P 500"
    WINDOW_SIZE: int = 10
    DATA_CACHE_DIR: str = "data_cache/paper_faithful"
    USE_DATA_CACHE: bool = True
    REFRESH_DATA_CACHE: bool = False
    REQUIRE_DATA_CACHE: bool = True
    INITIAL_CAPITAL: float = 500000.0

    # Training
    EPISODES: int = 200
    EPISODE_LENGTH: int = 650
    BATCH_SIZE: int = 128
    BUFFER_SIZE: int = 1000
    GAMMA: float = 0.99
    TAU: float = 0.001
    LR_ACTOR: float = 1e-5
    LR_CRITIC: float = 1e-3
    DROPOUT: float = 0.5
    POLICY_ENTROPY_COEF: float = 0.0
    MAX_GRAD_NORM: float = 1.0
    ACTION_PROJECTION_VERSION: int = 1
    BASELINE_ANCHOR_WEIGHT: float = 0.0
    ANCHOR_VERSION: int = 2
    BCM_TARGET_VERSION: int = 2
    BCM_LOSS_VERSION: int = 2
    USE_TARGET_POLICY_EVAL: bool = True
    STABLE_POLICY_SELECTION_VERSION: int = 1

    # Exploration
    INIT_NOISE: float = 0.05
    MIN_NOISE: float = 0.01
    USE_PARAMETER_NOISE: bool = True
    PARAM_NOISE_INIT_STD: float = 0.01
    PARAM_NOISE_MIN_STD: float = 0.002
    PARAM_NOISE_MAX_STD: float = 0.05
    PARAM_NOISE_TARGET_ACTION_STD: float = 0.10
    PARAM_NOISE_ADAPT_RATE: float = 1.01
    PARAM_NOISE_VERSION: int = 1

    # Oracle
    ORACLE_ANNEAL_EPISODES: int = 200
    ORACLE_RISK_AVERSION: float = 1.0
    BCM_LAMBDA: float = 0.10

    # Modules
    USE_IPM: bool = True
    USE_BCM: bool = True
    USE_DAM: bool = False
    USE_ARB: bool = False
    USE_SPARSE_NETWORK: bool = False
    USE_PRIORITIZED_REPLAY: bool = True

    # Costs
    COST_MODEL_VERSION: int = 2
    TRADING_COST_BPS: float = 20.0
    SLIPPAGE_BPS: float = 50.0
    SPREAD_BPS: float = 0.0
    MARKET_IMPACT_BPS: float = 0.0
    FIXED_COMMISSION: float = 0.0

    # DAM (GAN)
    GAN_NOISE_DIM: int = 8
    GAN_HIDDEN_DIM: int = 32
    GAN_SEQ_LEN: int = 95
    GAN_LR: float = 0.001
    GAN_EPOCHS: int = 200
    GAN_MMD_LAMBDA: float = 0.10
    DAM_LOSS_VERSION: int = 1
    
    #IPM
    IPM_PRETRAIN_EPOCHS: int = 50
    USE_ONLINE_IPM: bool = True
    IPM_ONLINE_LR: float = 1e-4
    IPM_VERSION: int = 2

    # Risk Regularization
    REBALANCE_FREQ: int = 1
    REWARD_MODE: str = "benchmark_relative"
    TRAINING_BENCHMARK_POLICY: str = "Equal Weight"
    RETURN_REWARD_SCALE: float = 1000.0
    SHARPE_WINDOW: int = 30
    TURNOVER_PENALTY: float = 0.0
    CONCENTRATION_PENALTY: float = 0.0
    DRAWDOWN_PENALTY: float = 0.0
    MAX_WEIGHT: float | None = None
    MAX_CASH_WEIGHT: float | None = 0.25
    CASH_PENALTY: float = 0.05
    USE_ACTIVE_OVERLAY: bool = False
    ACTIVE_OVERLAY_BASE_POLICY: str = "Equal Weight"
    ACTIVE_OVERLAY_BASE_WEIGHTS: List[float] | None = None
    ACTIVE_OVERLAY_BASE_WEIGHT: float = 0.80
    ACTIVE_OVERLAY_TILT_WEIGHT: float = 0.20
    ACTIVE_OVERLAY_TRACKING_PENALTY: float = 0.05

    # Adaptive Replay Buffer
    USE_ADAPTIVE_ARB_ACTIVATION: bool = True
    ARB_MIN_EPISODE: int = 20
    ARB_STABILITY_PATIENCE: int = 3
    ARB_POLICY_DRIFT_THRESHOLD: float = 0.075
    ARB_INSTABILITY_DECAY: float = 0.50
    ARB_STABILITY_RECOVERY: float = 0.20
    ARB_MIN_PORTFOLIO_VALUE_RATIO: float = 0.90
    ARB_MAX_ACTIVATION_TURNOVER: float = 0.18
    ARB_MIN_VALIDATION_SCORE: float = 0.0
    ARB_PROBE_SIZE: int = 256
    ARB_RAMP_EPISODES: int = 30
    ARB_CACHE_REFRESH_INTERVAL: int = 256
    ARB_START_EPISODE: int = 30
    ARB_FULL_EPISODE: int = 80
    ARB_MAX_MIX: float = 0.80
    ARB_TEMPERATURE: float = 0.25
    ARB_MIN_PROBABILITY: float = 1e-4
    ARB_RECENCY_TAU: float = 5000.0
    ARB_REWARD_WEIGHT: float = 0.35
    ARB_UNCERTAINTY_WEIGHT: float = 0.25
    ARB_ON_POLICY_WEIGHT: float = 0.25
    ARB_RECENCY_WEIGHT: float = 0.15

    # Prioritized replay
    PRIORITY_ALPHA: float = 0.6
    PRIORITY_BETA_START: float = 0.4
    PRIORITY_BETA_FRAMES: int = 100000
    PRIORITY_EPSILON: float = 1e-6

    # SparseNetwork4DRL
    SPARSE_DENSITY: float = 0.50
    SPARSE_WIDTH_MULTIPLIER: float = 2.0
    SPARSE_TOPOLOGY: str = "erdos_renyi"

    # Checkpointing
    MODEL_DIR: str = "models"
    EXPERIMENT_DIR: str = "runs"
    CHECKPOINT_FREQ: int = 5
    VALIDATION_FREQ: int = 5
    MIN_SAVE_EPISODE: int = 40
    EARLY_STOPPING_PATIENCE: int = 4
    EARLY_STOPPING_MIN_DELTA: float = 0.001
    MODEL_SELECTION_METRIC: str = "Benchmark Relative Score"
    MODEL_SELECTION_HURDLE: str = "CRP"
    MODEL_SELECTION_HURDLES: List[str] = ["CRP", "Equal Weight", "Buy & Hold EW"]
    SELECTION_RETURN_WEIGHT: float = 0.50
    SELECTION_DRAWDOWN_WEIGHT: float = 0.25
    SELECTION_TURNOVER_WEIGHT: float = 0.02
    ENSEMBLE_TOP_K: int = 1
    ENSEMBLE_TEMPERATURE: float = 0.25
    ENSEMBLE_MIN_SELECTION_SCORE: float | None = None
    ENSEMBLE_WEIGHTING: str = "softmax"

    # Experiment matrix
    EXPERIMENT_SEEDS: List[int] = [42, 7, 123, 2024, 2025]
    ABLATIONS: List[str] = [
        "paper_baseline",
        "paper_ipm",
        "paper_dam",
        "paper_bcm",
        "paper_ipm_dam",
        "paper_ipm_bcm",
        "paper_dam_bcm",
        "paper_all",
    ]
    WALK_FORWARD_START_YEAR: int = 2010
    WALK_FORWARD_END_YEAR: int = 2026
    WALK_FORWARD_TRAIN_YEARS: int = 6
    WALK_FORWARD_VALID_YEARS: int = 1
    WALK_FORWARD_TEST_YEARS: int = 1

    @field_validator("MAX_WEIGHT", "MAX_CASH_WEIGHT", "ENSEMBLE_MIN_SELECTION_SCORE", mode="before")
    @classmethod
    def parse_optional_float(cls, value):
        if isinstance(value, str) and value.strip().lower() in {"", "none", "null"}:
            return None
        return value

    class Config:
        env_file = ".env"


config = Settings()
