import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import os

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Hyperparameters
latent_dim = 100
img_channels = 3
img_size = 32
batch_size = 128
epochs = 200
lr_g = 0.0002
lr_d = 0.0002
beta1 = 0.5
beta2 = 0.999
lambda_reg = 0.1
lambda_dood1 = 0.1
lambda_dood2 = 0.1
fgsm_epsilon = 0.03
tau = 0.1  # Threshold for GMM density

# Create directories
os.makedirs('models', exist_ok=True)
os.makedirs('generated_samples', exist_ok=True)
