import torch
import torch.nn as nn
import torch.nn.functional as F

class Actor(nn.Module):
    def __init__(self, num_assets, window_size, action_dim):
        super(Actor, self).__init__()
        self.fe_lstm = nn.LSTM(input_size=num_assets * 3, 
                               hidden_size=20, 
                               batch_first=True)
        
        input_dim_fa = 20 + action_dim + (num_assets * 3)
        
        self.fa_net = nn.Sequential(
            nn.Linear(input_dim_fa, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 64),
            nn.LeakyReLU(),
            nn.Linear(64, action_dim) 
        )
        
        self.softmax = nn.Softmax(dim=1)
        self._init_weights()

    def _init_weights(self):
        for m in self.fa_net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
        
        last_layer = self.fa_net[-1]
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
        
        self.fa_net = nn.Sequential(
            nn.Linear(input_dim_fa, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 1) 
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.fa_net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
                
    def forward(self, state, prev_weights, ipm_pred, action):
        fe_out, _ = self.fe_lstm(state)
        fe_out = fe_out[:, -1, :]
        combined = torch.cat([fe_out, prev_weights, ipm_pred, action], dim=1)
        q_val = self.fa_net(combined)
        return q_val