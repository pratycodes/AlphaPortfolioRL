import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from models.actor_critic import Actor, Critic
from config.settings import config

class DDPGAgent:
    def __init__(self, num_assets, window_size):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_assets = num_assets
        
        self.action_dim = num_assets + 1 
        
        # Networks
        self.actor = Actor(num_assets, window_size, self.action_dim).to(self.device)
        self.actor_target = Actor(num_assets, window_size, self.action_dim).to(self.device)
        self.critic = Critic(num_assets, window_size, self.action_dim).to(self.device)
        self.critic_target = Critic(num_assets, window_size, self.action_dim).to(self.device)
        
        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=config.LR_ACTOR)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=config.LR_CRITIC)
        
        # Sync targets
        self._hard_update(self.actor, self.actor_target)
        self._hard_update(self.critic, self.critic_target)
        
        self.bcm_lambda = config.BCM_LAMBDA

    def update(self, batch, ipm_model):
        state, prev_w, action, reward, next_state, next_prev_w, greedy_action = batch
        
        state = torch.FloatTensor(state).to(self.device)
        prev_w = torch.FloatTensor(prev_w).to(self.device)
        action = torch.FloatTensor(action).to(self.device)
        reward = torch.FloatTensor(reward).unsqueeze(1).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        next_prev_w = torch.FloatTensor(next_prev_w).to(self.device)
        greedy_action = torch.FloatTensor(greedy_action).to(self.device) 

        if torch.isnan(greedy_action).any():
            fallback = torch.zeros_like(greedy_action)
            fallback[:, 0] = 1.0 
            greedy_action = torch.where(torch.isnan(greedy_action), fallback, greedy_action)

        # Clamp to [0, 1] range
        greedy_action = torch.clamp(greedy_action, 0.0, 1.0)
        
        sums = greedy_action.sum(dim=1, keepdim=True)
        greedy_action = greedy_action / (sums + 1e-8)

        with torch.no_grad():
            curr_ipm_pred = ipm_model(state)
            next_ipm_pred = ipm_model(next_state)

        # Critic Update 
        with torch.no_grad():
            next_action = self.actor_target(next_state, next_prev_w, next_ipm_pred)
            target_q = self.critic_target(next_state, next_prev_w, next_ipm_pred, next_action)
            target_q = reward + config.GAMMA * target_q
        
        current_q = self.critic(state, prev_w, curr_ipm_pred, action)
        critic_loss = F.mse_loss(current_q, target_q)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        actor_actions = self.actor(state, prev_w, curr_ipm_pred)
        rl_loss = -self.critic(state, prev_w, curr_ipm_pred, actor_actions).mean()
        
        # BCM Loss
        bcm_loss = F.binary_cross_entropy(actor_actions, greedy_action)
        
        total_actor_loss = rl_loss + (self.bcm_lambda * bcm_loss)
        
        self.actor_optimizer.zero_grad()
        total_actor_loss.backward()
        self.actor_optimizer.step()
        
        self._soft_update(self.actor, self.actor_target)
        self._soft_update(self.critic, self.critic_target)
        
        return critic_loss.item(), rl_loss.item(), bcm_loss.item()

    def _hard_update(self, source, target):
        target.load_state_dict(source.state_dict())

    def _soft_update(self, source, target):
        for param, target_param in zip(source.parameters(), target.parameters()):
            target_param.data.copy_(
                config.TAU * param.data + (1 - config.TAU) * target_param.data
            )