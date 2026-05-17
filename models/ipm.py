import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from utils.logger import setup_logger
import torch.optim as optim
from config.settings import config
from data.features import ohlc_feature_matrix

logger = setup_logger()

class IPM(nn.Module):
    def __init__(self, num_assets, window_size, hidden_dim=64):
        super(IPM, self).__init__()

        self.lstm = nn.LSTM(input_size=num_assets * 3, 
                            hidden_size=hidden_dim, 
                            batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_assets * 3) 
    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :] 
        prediction = self.fc(out)
        return prediction
    
    def pretrain_ipm(ipm, df, device, epochs):
        logger.info(f"Pre-training IPM for {epochs} epochs.")
        ipm.train()
        for p in ipm.parameters():
            p.requires_grad = True
            
        optimizer = optim.Adam(ipm.parameters(), lr=1e-3)
        criterion = nn.MSELoss()

        obs_data = ohlc_feature_matrix(df)
        
        X_list = []
        y_list = []
        window_size = config.WINDOW_SIZE
        
        for i in range(len(df) - window_size - 1):
            window = obs_data[i : i + window_size]
            norm_window = window / (window[0] + 1e-8)
            
            current_features = obs_data[i + window_size - 1]
            next_features = obs_data[i + window_size]
            target_relative = next_features / (current_features + 1e-8)
            
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
                loss = criterion(preds, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                
        logger.info("IPM Pre-training Completed.")
        ipm.eval()
        for p in ipm.parameters():
            p.requires_grad = False
        logger.info("IPM Frozen for RL training.")
