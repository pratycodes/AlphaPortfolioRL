import torch
import torch.nn as nn

from SparseNetwork4DRL import SparseLinear, build_mlp
from config.settings import config
from data.universe import feature_dim, ipm_feature_dim, market_feature_dim


def project_portfolio_weights(weights, max_weight=None, max_cash_weight=None):
    weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-8)
    cash = weights[:, :1]
    risky = weights[:, 1:]

    if max_weight is not None:
        max_weight_tensor = torch.as_tensor(max_weight, dtype=weights.dtype, device=weights.device)
        capped_risky = torch.minimum(risky, max_weight_tensor)
        cash = cash + (risky.sum(dim=1, keepdim=True) - capped_risky.sum(dim=1, keepdim=True))
        risky = capped_risky

    if max_cash_weight is not None:
        max_cash_tensor = torch.as_tensor(max_cash_weight, dtype=weights.dtype, device=weights.device)
        excess_cash = torch.clamp(cash - max_cash_tensor, min=0.0)
        cash = torch.minimum(cash, max_cash_tensor.expand_as(cash))
        risky = _redistribute_excess_to_risky(risky, excess_cash, max_weight)

    projected = torch.cat([cash, risky], dim=1)
    return projected / (projected.sum(dim=1, keepdim=True) + 1e-8)


def apply_equal_weight_anchor(weights, anchor_weight=0.0, max_weight=None, max_cash_weight=None):
    anchor_weight = float(anchor_weight)
    if anchor_weight <= 0.0 or weights.size(1) <= 1:
        return weights

    risky_count = weights.size(1) - 1
    anchor = torch.zeros_like(weights)
    anchor[:, 1:] = 1.0 / risky_count
    anchored = (anchor_weight * anchor) + ((1.0 - anchor_weight) * weights)
    return project_portfolio_weights(anchored, max_weight=max_weight, max_cash_weight=max_cash_weight)


def _redistribute_excess_to_risky(risky, excess, max_weight):
    if risky.size(1) == 0:
        return risky

    remaining = excess
    for _ in range(risky.size(1) + 1):
        if torch.max(remaining).item() <= 1e-8:
            break

        if max_weight is None:
            preference = torch.where(risky > 1e-8, risky, torch.ones_like(risky))
            risky = risky + remaining * preference / (preference.sum(dim=1, keepdim=True) + 1e-8)
            break

        max_weight_tensor = torch.as_tensor(max_weight, dtype=risky.dtype, device=risky.device)
        room = torch.clamp(max_weight_tensor - risky, min=0.0)
        if torch.max(room).item() <= 1e-8:
            break

        preference = torch.where(room > 1e-8, torch.clamp(risky, min=1e-8), torch.zeros_like(risky))
        fallback = room
        preference_sum = preference.sum(dim=1, keepdim=True)
        allocation_base = torch.where(preference_sum > 1e-8, preference, fallback)
        allocation = remaining * allocation_base / (allocation_base.sum(dim=1, keepdim=True) + 1e-8)
        allocation = torch.minimum(allocation, room)
        risky = risky + allocation
        remaining = remaining - allocation.sum(dim=1, keepdim=True)

    return risky


def _sparse_dims(base_dims):
    if not getattr(config, "USE_SPARSE_NETWORK", False):
        return base_dims
    multiplier = float(getattr(config, "SPARSE_WIDTH_MULTIPLIER", 1.0))
    return [max(1, int(dim * multiplier)) for dim in base_dims]

class Actor(nn.Module):
    def __init__(self, num_assets, window_size, action_dim):
        super(Actor, self).__init__()
        self.feature_dim = feature_dim(config)
        self.ipm_dim = ipm_feature_dim(config)
        self.market_dim = market_feature_dim(config)
        self.fe_lstm = nn.LSTM(input_size=window_size,
                               hidden_size=20, 
                               batch_first=True,
                               bidirectional=True)
        
        input_dim_fa = (40 * self.feature_dim) + action_dim + self.ipm_dim + self.market_dim
        
        self.fa_net = build_mlp(
            input_dim_fa,
            _sparse_dims([256, 128, 64, 32]),
            action_dim,
            use_sparse=getattr(config, "USE_SPARSE_NETWORK", False),
            density=getattr(config, "SPARSE_DENSITY", 0.35),
            seed=getattr(config, "SEED", None),
            topology=getattr(config, "SPARSE_TOPOLOGY", "erdos_renyi"),
            dropout=getattr(config, "DROPOUT", 0.0),
        )
        
        self.softmax = nn.Softmax(dim=1)
        self._init_weights()

    def _init_weights(self):
        linear_layers = []
        for m in self.fa_net.modules():
            if isinstance(m, (nn.Linear, SparseLinear)):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
                linear_layers.append(m)
        
        last_layer = linear_layers[-1]
        last_layer.weight.data.uniform_(-3e-3, 3e-3)
        last_layer.bias.data.uniform_(-3e-3, 3e-3)

    def forward(self, state, prev_weights, ipm_pred):
        fe_input = state.transpose(1, 2)
        fe_out, _ = self.fe_lstm(fe_input)
        fe_out = fe_out.reshape(fe_out.size(0), -1)
        market_indicator = _market_indicator_from_state(state)
        
        combined = torch.cat([fe_out, prev_weights, ipm_pred, market_indicator], dim=1)
        logits = self.fa_net(combined)
        actions = self.softmax(logits)
        projected_actions = project_portfolio_weights(
            actions,
            max_weight=getattr(config, "MAX_WEIGHT", None),
            max_cash_weight=getattr(config, "MAX_CASH_WEIGHT", None),
        )
        return apply_equal_weight_anchor(
            projected_actions,
            anchor_weight=getattr(config, "BASELINE_ANCHOR_WEIGHT", 0.0),
            max_weight=getattr(config, "MAX_WEIGHT", None),
            max_cash_weight=getattr(config, "MAX_CASH_WEIGHT", None),
        )

class Critic(nn.Module):
    def __init__(self, num_assets, window_size, action_dim):
        super(Critic, self).__init__()
        self.feature_dim = feature_dim(config)
        self.ipm_dim = ipm_feature_dim(config)
        self.market_dim = market_feature_dim(config)
        
        self.fe_lstm = nn.LSTM(
            input_size=window_size,
            hidden_size=20,
            batch_first=True,
            bidirectional=True,
        )
        
        input_dim_fa = (40 * self.feature_dim) + action_dim + self.ipm_dim + self.market_dim + action_dim
        
        self.fa_net = build_mlp(
            input_dim_fa,
            _sparse_dims([256, 128, 64, 32]),
            1,
            use_sparse=getattr(config, "USE_SPARSE_NETWORK", False),
            density=getattr(config, "SPARSE_DENSITY", 0.35),
            seed=getattr(config, "SEED", None) + 1000 if getattr(config, "SEED", None) is not None else None,
            topology=getattr(config, "SPARSE_TOPOLOGY", "erdos_renyi"),
            dropout=getattr(config, "DROPOUT", 0.0),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.fa_net.modules():
            if isinstance(m, (nn.Linear, SparseLinear)):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
                
    def forward(self, state, prev_weights, ipm_pred, action):
        fe_input = state.transpose(1, 2)
        fe_out, _ = self.fe_lstm(fe_input)
        fe_out = fe_out.reshape(fe_out.size(0), -1)
        market_indicator = _market_indicator_from_state(state)
        combined = torch.cat([fe_out, prev_weights, ipm_pred, market_indicator, action], dim=1)
        q_val = self.fa_net(combined)
        return q_val


def _market_indicator_from_state(state):
    if not getattr(config, "USE_MARKET_FEATURE", False):
        return torch.zeros(state.size(0), 0, dtype=state.dtype, device=state.device)

    market_close_index = state.size(2) - 1
    if state.size(1) < 2:
        return torch.ones(state.size(0), 1, dtype=state.dtype, device=state.device)

    previous_ratio = torch.clamp(state[:, -2, market_close_index], min=1e-8)
    return (1.0 / previous_ratio).unsqueeze(1)
