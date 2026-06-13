import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import logging
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset
from config.settings import config
from data.fetcher import fetch_data
from data.features import ohlc_feature_matrix
from data.dam_gan import RGAN_Generator, RGAN_Discriminator
from data.universe import data_tickers, feature_assets

logger = logging.getLogger("quant_rl")

def prepare_gan_data(df, window_size, assets=None):
    raw_values = ohlc_feature_matrix(df, assets=assets)
    returns = np.diff(raw_values, axis=0) / (raw_values[:-1] + 1e-8)
    returns = np.nan_to_num(returns, nan=0.0, posinf=0.0, neginf=0.0)
    
    windows = []
    for i in range(len(returns) - window_size):
        windows.append(returns[i : i + window_size])
    
    return np.array(windows, dtype=np.float32)

def train_gan(output_path="models/dam_generator.pth"):
    logger.info("Fetching Data for GAN Training.")
    gan_assets = feature_assets(config)
    df = fetch_data(data_tickers(config), config.TRAIN_START_DATE, config.TRAIN_END_DATE)
    
    real_data = prepare_gan_data(df, config.GAN_SEQ_LEN, assets=gan_assets)
    # Shape: (Batch, Seq, Assets)
    dataset = TensorDataset(torch.from_numpy(real_data))
    loader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    
    num_features = len(gan_assets) * 3
    
    generator = RGAN_Generator(config.GAN_NOISE_DIM, config.GAN_HIDDEN_DIM, config.GAN_SEQ_LEN, num_features)
    discriminator = RGAN_Discriminator(num_features, config.GAN_HIDDEN_DIM, config.GAN_SEQ_LEN)
    
    g_optim = optim.Adam(generator.parameters(), lr=config.GAN_LR)
    d_optim = optim.Adam(discriminator.parameters(), lr=config.GAN_LR)
    
    loss_fn = nn.BCELoss()
    
    logger.info("Starting GAN Training.")
    
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
            adversarial_loss = loss_fn(preds, real_labels)
            mmd_loss = _mmd_rbf(real_batch, fake_batch)
            g_loss = adversarial_loss + config.GAN_MMD_LAMBDA * mmd_loss
            
            g_loss.backward()
            g_optim.step()
            
        if (epoch + 1) % 10 == 0:
            logger.info(f"Epoch {epoch+1} | D Loss: {d_loss.item():.4f} | G Loss: {g_loss.item():.4f}")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(generator.state_dict(), output_path)
    logger.info(f"GAN Generator saved to {output_path}")


def _mmd_rbf(real_batch, fake_batch, bandwidths=(0.1, 0.5, 1.0, 2.0)):
    real = real_batch.reshape(real_batch.size(0), -1)
    fake = fake_batch.reshape(fake_batch.size(0), -1)

    xx = torch.cdist(real, real).pow(2)
    yy = torch.cdist(fake, fake).pow(2)
    xy = torch.cdist(real, fake).pow(2)

    loss = torch.zeros((), device=real_batch.device)
    for bandwidth in bandwidths:
        gamma = 1.0 / (2.0 * bandwidth * bandwidth)
        loss = loss + torch.exp(-gamma * xx).mean()
        loss = loss + torch.exp(-gamma * yy).mean()
        loss = loss - 2.0 * torch.exp(-gamma * xy).mean()
    return loss / len(bandwidths)
