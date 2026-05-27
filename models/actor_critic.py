import torch
import torch.nn as nn

from SparseNetwork4DRL import SparseLinear, build_mlp
from config.settings import config

class Actor(nn.Module):
    def __init__(self, num_assets, window_size, action_dim):
        super(Actor, self).__init__()
        self.fe_lstm = nn.LSTM(input_size=num_assets * 3, 
                               hidden_size=20, 
                               batch_first=True)
        
        input_dim_fa = 20 + action_dim + (num_assets * 3)
        
        self.fa_net = build_mlp(
            input_dim_fa,
            [256, 128, 64],
            action_dim,
            use_sparse=getattr(config, "USE_SPARSE_NETWORK", False),
            density=getattr(config, "SPARSE_DENSITY", 0.35),
            seed=getattr(config, "SEED", None),
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
        fe_out, _ = self.fe_lstm(state)
        fe_out = fe_out[:, -1, :] 
        
        combined = torch.cat([fe_out, prev_weights, ipm_pred], dim=1)
        logits = self.fa_net(combined)
        actions = self.softmax(logits)
        return actions

class Critic(nn.Module):
    def __init__(self, num_assets, window_size, action_dim):
        super(Critic, self).__init__()
        
        self.fe_lstm = nn.LSTM(input_size=num_assets * 3, hidden_size=20, batch_first=True)
        
        input_dim_fa = 20 + action_dim + (num_assets * 3) + action_dim
        
        self.fa_net = build_mlp(
            input_dim_fa,
            [256, 128],
            1,
            use_sparse=getattr(config, "USE_SPARSE_NETWORK", False),
            density=getattr(config, "SPARSE_DENSITY", 0.35),
            seed=getattr(config, "SEED", None) + 1000 if getattr(config, "SEED", None) is not None else None,
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.fa_net.modules():
            if isinstance(m, (nn.Linear, SparseLinear)):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
                
    def forward(self, state, prev_weights, ipm_pred, action):
        fe_out, _ = self.fe_lstm(state)
        fe_out = fe_out[:, -1, :]
        combined = torch.cat([fe_out, prev_weights, ipm_pred, action], dim=1)
        q_val = self.fa_net(combined)
        return q_val
