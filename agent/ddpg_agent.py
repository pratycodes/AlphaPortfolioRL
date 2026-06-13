import copy
from contextlib import contextmanager

import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from models.actor_critic import Actor, Critic
from config.settings import config
from data.universe import ipm_feature_dim

class DDPGAgent:
    def __init__(self, num_assets, window_size):
        self.device = torch.device(config.DEVICE)
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
        self.entropy_coef = float(getattr(config, "POLICY_ENTROPY_COEF", 0.0))
        self.max_grad_norm = float(getattr(config, "MAX_GRAD_NORM", 0.0))
        self.last_update_stats = {}

    def update(self, batch, ipm_model):
        if len(batch) == 10:
            state, prev_w, action, reward, next_state, next_prev_w, greedy_action, done, priority_indices, importance_weights = batch
        else:
            state, prev_w, action, reward, next_state, next_prev_w, greedy_action, done = batch
            priority_indices = None
            importance_weights = None
        
        state = torch.FloatTensor(state).to(self.device)
        prev_w = torch.FloatTensor(prev_w).to(self.device)
        action = torch.FloatTensor(action).to(self.device)
        reward = torch.FloatTensor(reward).unsqueeze(1).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        next_prev_w = torch.FloatTensor(next_prev_w).to(self.device)
        greedy_action = torch.FloatTensor(greedy_action).to(self.device) 
        done = torch.FloatTensor(done).unsqueeze(1).to(self.device)
        if importance_weights is None:
            sample_weights = torch.ones_like(done)
        else:
            sample_weights = torch.FloatTensor(importance_weights).unsqueeze(1).to(self.device)

        if torch.isnan(greedy_action).any():
            fallback = torch.zeros_like(greedy_action)
            fallback[:, 0] = 1.0 
            greedy_action = torch.where(torch.isnan(greedy_action), fallback, greedy_action)

        # Clamp to [0, 1] range
        greedy_action = torch.clamp(greedy_action, 0.0, 1.0)
        
        sums = greedy_action.sum(dim=1, keepdim=True)
        greedy_action = greedy_action / (sums + 1e-8)

        with torch.no_grad():
            if config.USE_IPM:
                curr_ipm_pred = ipm_model(state)
                next_ipm_pred = ipm_model(next_state)
            else:
                pred_dim = ipm_feature_dim(config)
                curr_ipm_pred = torch.zeros(state.size(0), pred_dim, device=self.device)
                next_ipm_pred = torch.zeros(next_state.size(0), pred_dim, device=self.device)

        # Critic Update 
        with torch.no_grad(), self._temporary_eval(self.actor_target, self.critic_target):
            next_action = self.actor_target(next_state, next_prev_w, next_ipm_pred)
            target_q = self.critic_target(next_state, next_prev_w, next_ipm_pred, next_action)
            target_q = reward + config.GAMMA * (1.0 - done) * target_q
        
        current_q = self.critic(state, prev_w, curr_ipm_pred, action)
        td_error = target_q - current_q
        critic_loss = (sample_weights * td_error.pow(2)).mean()
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        critic_grad_norm = self._clip_and_measure_grad_norm(self.critic.parameters())
        self.critic_optimizer.step()
        
        actor_actions = self.actor(state, prev_w, curr_ipm_pred)
        rl_loss = -self.critic(state, prev_w, curr_ipm_pred, actor_actions).mean()
        policy_entropy = -(actor_actions * torch.log(actor_actions + 1e-8)).sum(dim=1).mean()
        
        if config.USE_BCM:
            bcm_loss = -(greedy_action * torch.log(actor_actions + 1e-8)).sum(dim=1).mean()
        else:
            bcm_loss = torch.zeros((), device=self.device)
        entropy_loss = -self.entropy_coef * policy_entropy
        total_actor_loss = rl_loss + (self.bcm_lambda * bcm_loss) + entropy_loss
        
        self.actor_optimizer.zero_grad()
        total_actor_loss.backward()
        actor_grad_norm = self._clip_and_measure_grad_norm(self.actor.parameters())
        self.actor_optimizer.step()
        
        self._soft_update(self.actor, self.actor_target)
        self._soft_update(self.critic, self.critic_target)

        self.last_update_stats = {
            "policy_entropy": policy_entropy.item(),
            "normalized_policy_entropy": (policy_entropy / np.log(self.action_dim)).item(),
            "actor_grad_norm": actor_grad_norm,
            "critic_grad_norm": critic_grad_norm,
            "entropy_loss": entropy_loss.item(),
            "td_errors": td_error.detach().abs().cpu().numpy().reshape(-1),
            "priority_indices": priority_indices,
        }
        
        return critic_loss.item(), rl_loss.item(), bcm_loss.item()

    def _hard_update(self, source, target):
        target.load_state_dict(source.state_dict())

    def _soft_update(self, source, target):
        for param, target_param in zip(source.parameters(), target.parameters()):
            target_param.data.copy_(
                config.TAU * param.data + (1 - config.TAU) * target_param.data
            )

    @contextmanager
    def _temporary_eval(self, *modules):
        training_modes = [module.training for module in modules]
        try:
            for module in modules:
                module.eval()
            yield
        finally:
            for module, was_training in zip(modules, training_modes):
                module.train(was_training)

    def _clip_and_measure_grad_norm(self, parameters):
        params = [param for param in parameters if param.grad is not None]
        if not params:
            return 0.0

        if self.max_grad_norm > 0:
            total_norm = torch.nn.utils.clip_grad_norm_(params, self.max_grad_norm)
            return float(total_norm.detach().cpu().item())

        total = torch.zeros((), device=self.device)
        for param in params:
            total = total + param.grad.detach().pow(2).sum()
        return float(torch.sqrt(total).cpu().item())

    def create_perturbed_actor(self, noise_std):
        perturbed_actor = copy.deepcopy(self.actor).to(self.device)
        with torch.no_grad():
            for param in perturbed_actor.parameters():
                if param.requires_grad:
                    param.add_(torch.randn_like(param) * noise_std)
        perturbed_actor.eval()
        return perturbed_actor

    def parameter_noise_distance(self, perturbed_actor, states, prev_weights, ipm_model):
        if len(states) == 0:
            return 0.0

        state_tensor = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        prev_weight_tensor = torch.as_tensor(prev_weights, dtype=torch.float32, device=self.device)

        actor_was_training = self.actor.training
        ipm_was_training = ipm_model.training
        self.actor.eval()
        ipm_model.eval()

        with torch.no_grad():
            if getattr(config, "USE_IPM", False):
                ipm_pred = ipm_model(state_tensor)
            else:
                ipm_pred = torch.zeros(state_tensor.size(0), ipm_feature_dim(config), device=self.device)
            base_actions = self.actor(state_tensor, prev_weight_tensor, ipm_pred)
            perturbed_actions = perturbed_actor(state_tensor, prev_weight_tensor, ipm_pred)
            distance = torch.sqrt(torch.mean(torch.sum((base_actions - perturbed_actions) ** 2, dim=1)))

        if actor_was_training:
            self.actor.train()
        if ipm_was_training:
            ipm_model.train()
        return float(distance.detach().cpu().item())
