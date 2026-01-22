import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from config.settings import config
from data.fetcher import fetch_data
from data.dam_gan import RGAN_Generator, RGAN_Discriminator
from utils.logger import setup_logger

logger = setup_logger(name="dam_trainer")

def prepare_gan_data(df, window_size):
    raw_values = df.xs('Close', level=1, axis=1).values
    returns = np.diff(raw_values, axis=0) / (raw_values[:-1] + 1e-8)
    
    windows = []
    for i in range(len(returns) - window_size):
        windows.append(returns[i : i + window_size])
    
    return np.array(windows, dtype=np.float32)

def train_gan():
    logger.info("Fetching Data for GAN Training.")
    df = fetch_data(config.ASSETS, config.TRAIN_START_DATE, config.TRAIN_END_DATE)
    
    real_data = prepare_gan_data(df, config.GAN_SEQ_LEN)
    # Shape: (Batch, Seq, Assets)
    dataset = TensorDataset(torch.from_numpy(real_data))
    loader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    
    num_features = len(config.ASSETS)
    
    generator = RGAN_Generator(config.GAN_NOISE_DIM, config.GAN_HIDDEN_DIM, config.GAN_SEQ_LEN, num_features)
    discriminator = RGAN_Discriminator(num_features, config.GAN_HIDDEN_DIM, config.GAN_SEQ_LEN)
    
    g_optim = optim.Adam(generator.parameters(), lr=config.GAN_LR)
    d_optim = optim.Adam(discriminator.parameters(), lr=config.GAN_LR)
    
    loss_fn = nn.BCELoss()
    
    logger.info("Starting GAN Training...")
    
    for epoch in range(config.GAN_EPOCHS):
        for i, (real_batch,) in enumerate(loader):
            batch_size = real_batch.size(0)
            
            # Train Discriminator 
            d_optim.zero_grad()
            
            real_labels = torch.ones(batch_size, 1)
            real_preds = discriminator(real_batch)
            d_real_loss = loss_fn(real_preds, real_labels)
            
            # Fake Data
            noise = torch.randn(batch_size, config.GAN_SEQ_LEN, config.GAN_NOISE_DIM)
            fake_batch = generator(noise)
            fake_labels = torch.zeros(batch_size, 1)
            fake_preds = discriminator(fake_batch.detach()) 
            d_fake_loss = loss_fn(fake_preds, fake_labels)
            
            d_loss = d_real_loss + d_fake_loss
            d_loss.backward()
            d_optim.step()
            
            # Train Generator 
            g_optim.zero_grad()
            
            preds = discriminator(fake_batch)
            g_loss = loss_fn(preds, real_labels)
            
            g_loss.backward()
            g_optim.step()
            
        if (epoch + 1) % 10 == 0:
            logger.info(f"Epoch {epoch+1} | D Loss: {d_loss.item():.4f} | G Loss: {g_loss.item():.4f}")

    torch.save(generator.state_dict(), "models/dam_generator.pth")
    logger.info("GAN Generator saved to models/dam_generator.pth")
