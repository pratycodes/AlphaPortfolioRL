import torch
import torch.nn as nn
import numpy as np
import logging
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
from config.settings import config
from data.features import ohlc_feature_matrix
from data.universe import feature_assets, feature_dim, ipm_feature_dim

logger = logging.getLogger("quant_rl")

class IPM(nn.Module):
    def __init__(self, num_assets, window_size, hidden_dim=100):
        super(IPM, self).__init__()
        self.input_dim = feature_dim(config)
        self.output_dim = ipm_feature_dim(config)
        self.delay = 3
        self.register_buffer("decays", torch.tensor([0.1, 0.2, 0.5, 0.8], dtype=torch.float32))

        self.bias = nn.Parameter(torch.zeros(self.output_dim))
        self.delay_weights = nn.Parameter(torch.zeros(self.delay - 1, self.output_dim, self.output_dim))
        self.trace_weights = nn.Parameter(torch.zeros(len(self.decays), self.output_dim, self.output_dim))
        self.log_variance = nn.Parameter(torch.zeros(self.output_dim))
        self.rnn = nn.RNN(
            input_size=self.output_dim,
            hidden_size=hidden_dim,
            nonlinearity="tanh",
            batch_first=True,
        )
        self.rnn_to_bias = nn.Linear(hidden_dim, self.output_dim)

        nn.init.normal_(self.rnn.weight_ih_l0, mean=0.0, std=0.1)
        nn.init.normal_(self.rnn.weight_hh_l0, mean=0.0, std=1.0)
        nn.init.zeros_(self.rnn.bias_ih_l0)
        nn.init.zeros_(self.rnn.bias_hh_l0)
        nn.init.zeros_(self.rnn_to_bias.weight)
        nn.init.zeros_(self.rnn_to_bias.bias)
        for parameter in self.rnn.parameters():
            parameter.requires_grad = False

    def forward(self, x):
        sequence = self._price_change_sequence(x)
        rnn_out, _ = self.rnn(sequence)
        prediction = self.bias + self.rnn_to_bias(rnn_out[:, -1, :])

        for lag in range(1, self.delay):
            lag_values = sequence[:, -lag, :]
            prediction = prediction + torch.einsum("bi,oi->bo", lag_values, self.delay_weights[lag - 1])

        traces = self._eligibility_traces(sequence)
        for index in range(traces.size(1)):
            prediction = prediction + torch.einsum("bi,oi->bo", traces[:, index, :], self.trace_weights[index])
        return prediction
    
    def pretrain_ipm(ipm, df, device, epochs):
        logger.info(f"Pre-training IPM for {epochs} epochs.")
        ipm.train()
        for p in ipm.parameters():
            p.requires_grad = True
            
        optimizer = optim.RMSprop(ipm.parameters(), lr=1e-3)
        obs_data, close_data = _paper_state_values(df)
        target_data = ohlc_feature_matrix(df, assets=feature_assets(config))
        
        X_list = []
        y_list = []
        window_size = config.WINDOW_SIZE
        
        for i in range(len(df) - window_size - 1):
            window = obs_data[i : i + window_size]
            current_close = close_data[i + window_size - 1]
            norm_window = window / (np.repeat(current_close, 3) + 1e-8)

            current_features = target_data[i + window_size - 1]
            next_features = target_data[i + window_size]
            target_relative = (next_features - current_features) / (current_features + 1e-8)
            target_relative = np.nan_to_num(target_relative, nan=0.0, posinf=0.0, neginf=0.0)
            
            X_list.append(norm_window)
            y_list.append(target_relative)
            
        X = torch.tensor(np.array(X_list), dtype=torch.float32).to(device)
        y = torch.tensor(np.array(y_list), dtype=torch.float32).to(device)
        
        dataset = TensorDataset(X, y)
        loader = DataLoader(dataset, batch_size=64, shuffle=True)
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            for batch_X, batch_y in loader:
                optimizer.zero_grad()
                preds = ipm(batch_X)
                loss = ipm.negative_log_likelihood(preds, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                
        logger.info("IPM Pre-training Completed.")
        ipm.eval()
        if getattr(config, "USE_ONLINE_IPM", False):
            for p in ipm.parameters():
                p.requires_grad = True
            logger.info("IPM left trainable for online updates.")
        else:
            for p in ipm.parameters():
                p.requires_grad = False
            logger.info("IPM Frozen for RL training.")

    @staticmethod
    def online_update(ipm, optimizer, state, target, device):
        ipm.train()
        state_tensor = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        target_tensor = torch.as_tensor(target, dtype=torch.float32, device=device).unsqueeze(0)

        optimizer.zero_grad()
        prediction = ipm(state_tensor)
        loss = ipm.negative_log_likelihood(prediction, target_tensor)
        loss.backward()
        optimizer.step()
        ipm.eval()
        return float(loss.detach().cpu().item())

    def _eligibility_traces(self, x):
        decays = self.decays.to(dtype=x.dtype, device=x.device)
        traces = torch.zeros(x.size(0), len(decays), x.size(2), dtype=x.dtype, device=x.device)
        for t in range(x.size(1)):
            traces = (decays.view(1, -1, 1) * traces) + x[:, t : t + 1, :]
        return traces

    def negative_log_likelihood(self, prediction, target):
        variance = torch.exp(self.log_variance).clamp_min(1e-8)
        return 0.5 * (((target - prediction).pow(2) / variance) + self.log_variance).mean()

    def _price_change_sequence(self, state):
        risky_start = 3
        risky_end = risky_start + self.output_dim
        risky_features = state[:, :, risky_start:risky_end]
        previous = torch.clamp(risky_features[:, :-1, :], min=1e-8)
        changes = (risky_features[:, 1:, :] - risky_features[:, :-1, :]) / previous
        if changes.size(1) == 0:
            return torch.zeros(
                state.size(0),
                1,
                self.output_dim,
                dtype=state.dtype,
                device=state.device,
            )
        first = torch.zeros(
            state.size(0),
            1,
            self.output_dim,
            dtype=state.dtype,
            device=state.device,
        )
        return torch.cat([first, changes], dim=1)


def _paper_state_values(df):
    assets = feature_assets(config)
    matrix_assets = list(assets)
    market_ticker = getattr(config, "MARKET_FEATURE_TICKER", None)
    if getattr(config, "USE_MARKET_FEATURE", False) and market_ticker in df.columns.get_level_values(0):
        matrix_assets.append(market_ticker)

    real_values = ohlc_feature_matrix(df, assets=matrix_assets)
    cash = np.ones((len(df), 3), dtype=float)
    values = np.concatenate([cash, real_values], axis=1)

    close = df.xs("Close", level=1, axis=1).loc[:, matrix_assets].to_numpy(dtype=float)
    close = np.concatenate([np.ones((len(df), 1)), close], axis=1)
    return values, close
