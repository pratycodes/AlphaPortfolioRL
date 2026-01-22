# AlphaPortfolioRL

**AlphaPortfolioRL** is an institutional-grade algorithmic trading system that leverages Deep Reinforcement Learning (DRL) for dynamic portfolio optimization. The system utilizes a **Deep Deterministic Policy Gradient (DDPG)** agent, augmented with **Generative Adversarial Networks (GANs)** for data augmentation and a **Convex Optimization Oracle** for behavioral cloning.

This framework is designed to solve the continuous control problem of asset allocation while strictly adhering to real-world financial constraints, including transaction costs, slippage, and risk-adjusted performance objectives.

---

## Overview

Traditional portfolio optimization methods, such as Mean-Variance Optimization (MVO), often fail to capture the non-linear, stochastic nature of financial markets. AlphaPortfolioRL addresses this by modeling portfolio management as a Markov Decision Process (MDP).

The system trains an agent to autonomously rebalance a portfolio of equities to maximize the Sharpe Ratio. It employs a **Model-Based Reinforcement Learning** approach where the agent is supported by:

* **Infused Prediction Module (IPM):** Pre-processes market data to extract latent predictive features.
* **Data Augmentation Module (DAM):** Generates synthetic market regimes to prevent overfitting.
* **Behavior Cloning Module (BCM):** Stabilizes training by imitating an optimal greedy oracle.

---

## Key Features

* **Deep Deterministic Policy Gradient (DDPG):** Implements an Actor-Critic architecture suitable for continuous action spaces (portfolio weights).
* **Risk-Adjusted Reward Function:** Optimizes for a Rolling Sharpe Ratio rather than raw returns, explicitly penalizing excessive volatility.
* **Synthetic Data Injection:** Utilizes a Recurrent GAN (RGAN) to inject synthetic market trajectories into the replay buffer, enhancing the agent's generalization capabilities.
* **Realistic Market Simulation:**
* **Transaction Costs:** Models trading fees (20 bps) and slippage (50 bps).
* **Constraints:** Enforces diversification via maximum weight constraints.


* **Hybrid Training Mechanism:** Combines standard RL gradients with supervised behavioral cloning loss from a convex optimization oracle (CVXPY).

---

## System Architecture

The AlphaPortfolioRL pipeline consists of four distinct modules:

### 1. Infused Prediction Module (IPM)

A supervised regression network (LSTM-based) that predicts future price movements. It serves as a feature extractor, providing the RL agent with a "forward-looking" state representation rather than just historical lag features.

### 2. Data Augmentation Module (DAM)

A **Recurrent Generative Adversarial Network (RGAN)** trained on historical asset prices. It generates realistic, synthetic time-series data to augment the experience replay buffer, mitigating the risk of overfitting to specific historical dates.

### 3. Behavior Cloning Module (BCM)

An auxiliary optimization layer. During training, a **Convex Oracle** calculates the mathematically optimal rebalancing weight for the next time step (hindsight optimization). The Actor network includes a loss term to minimize the divergence between its action and the Oracle's optimal action, accelerating convergence.

### 4. DDPG Agent

The core decision-making unit.

* **Actor:** Maps the IPM state to portfolio weights via a Softmax output layer (ensuring ).
* **Critic:** Estimates the Q-value (expected future risk-adjusted reward) of the Actor's allocation.

---

## Installation

### Prerequisites

* Python 3.10 or higher
* CUDA-enabled GPU (Recommended for GAN training)

### Setup Steps

1. **Clone the Repository**
```bash
git clone https://github.com/yourusername/AlphaPortfolioRL.git
cd AlphaPortfolioRL

```


2. **Create a Virtual Environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

```


3. **Install Dependencies**
```bash
pip install -r requirements.txt

```



---

## Configuration

All system hyperparameters are centralized in `config/settings.py`. Key parameters include:

* **Asset Universe:**
```python
ASSETS = ["AAPL", "MSFT", "AMZN", "GOOGL", "META", "TSLA", "JPM"]

```


* **Training Period:**
```python
TRAIN_START_DATE = "2010-01-01"
TRAIN_END_DATE = "2024-12-31"

```


* **Risk Parameters:**
```python
TRADING_COST_BPS = 0.0020  # 20 basis points
MAX_WEIGHT = 0.60          # Max allocation per asset
RISK_AVERSION = 0.05       # Volatility penalty factor

```



---

## Usage

### 1. Training

To start the full training pipeline (DAM training, IPM pre-training, and RL loop), run:

```bash
python main.py

```

* **Output:** Checkpoints are saved in the `models/` directory (`best_actor.pth`, `best_ipm.pth`).
* **Logs:** Training metrics are streamed to `logs/`.

### 2. Evaluation 

To generate a professional performance report comparing the agent against the S&P 500 and an Equal-Weight Benchmark:

```bash
python -m evaluation.dashboard
```

* **Output:** Generates `dashboard_sp500.png` containing equity curves and asset allocation area plots.

---

## Results

In out-of-sample testing (2025), the agent demonstrated significant alpha generation capabilities:

* **Total Return:** +34.53% (vs S&P 500: +16.34%)
* **Sharpe Ratio:** 1.05
* **Sortino Ratio:** 1.71
* **Max Drawdown:** -30.86%

*Note: Past performance is not indicative of future results.*

---

## References

This project is an enhanced implementation and real-world extension of the research presented in:

> **Yu, P., Lee, J. S., Kulyatin, I., Shi, Z., & Dasgupta, S. (2019). Model-based Deep Reinforcement Learning for Dynamic Portfolio Optimization.** *arXiv preprint arXiv:1901.08740.*
> [https://doi.org/10.48550/arXiv.1901.08740](https://doi.org/10.48550/arXiv.1901.08740)

We credit the original authors for the architectural concepts of the **Infused Prediction Module (IPM)**, **Data Augmentation Module (DAM)**, and **Behavior Cloning Module (BCM)** used in this repository.

---

## Disclaimer

**IMPORTANT: READ BEFORE USE**

This software is provided for **educational and research purposes only**.

* **No Financial Advice:** Nothing in this repository constitutes financial, investment, legal, or tax advice.
* **Not for Live Trading:** This system is a research prototype. It has **not** been audited for live deployment and lacks critical safeguards required for real-money trading (e.g., latency handling, order execution logic, risk kill-switches).
* **Risk of Loss:** Algorithmic trading involves a substantial risk of loss. The authors and contributors assume **no liability** for any financial losses, damages, or legal consequences resulting from the use or misuse of this code.

**Use this software at your own risk.**