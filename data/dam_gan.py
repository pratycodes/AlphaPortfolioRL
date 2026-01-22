import torch
import torch.nn as nn

class RGAN_Generator(nn.Module):
    """
    Recurrent Generator for Data Augmentation.
    Generates synthetic price relative sequences.
    """
    def __init__(self, noise_dim, hidden_dim, seq_len, num_features):
        super(RGAN_Generator, self).__init__()
        self.seq_len = seq_len
        self.noise_dim = noise_dim
        
        # LSTM Generator
        self.lstm = nn.LSTM(input_size=noise_dim, 
                            hidden_size=hidden_dim, 
                            num_layers=1, 
                            batch_first=True)
        
        self.fc = nn.Linear(hidden_dim, num_features)

    def forward(self, z):
        lstm_out, _ = self.lstm(z)
        return torch.tanh(self.fc(lstm_out))

class RGAN_Discriminator(nn.Module):
    """
    Recurrent Discriminator.
    """
    def __init__(self, input_dim, hidden_dim, seq_len):
        super(RGAN_Discriminator, self).__init__()
        
        self.lstm = nn.LSTM(input_size=input_dim, 
                            hidden_size=hidden_dim, 
                            num_layers=1, 
                            batch_first=True)
        
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]
        return torch.sigmoid(self.fc(last_hidden))