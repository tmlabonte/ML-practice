import torch
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import transforms
import torch.nn.functional as F
from torch import nn
import pdb

INPUT_DIM = 28 * 28
BATCH_SIZE = 32

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.linear1 = nn.Linear(INPUT_DIM, 100)
        self.linear2 = nn.Linear(100, 25)

    def forward(self, X):
        hidden = F.relu(self.linear1(X))
        latent = F.relu(self.linear2(hidden))

        return latent
    
class Decoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.linear1 = nn.Linear(25, 100)
        self.linear2 = nn.Linear(100, INPUT_DIM)

    def forward(self, X):
        hidden = F.relu(self.linear1(X))
        reconstructed = F.relu(self.linear2(hidden))
        
        return reconstructed 

class VAE(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = Encoder()
        self.decoder = Decoder()

        self.mean = nn.Linear(25, 25)
        self.log_var = nn.Linear(25, 25)

    # X is the input images
    def forward(self, X):
        hidden = self.encoder(X)

        mu = self.mean(hidden)
        log_var = self.log_var(hidden)

        var = torch.exp(log_var)

        noise = torch.from_numpy(np.random.normal(0, 1, size=var.size())).float()
        latent = mu + var * noise

        reconstructed = self.decoder(latent)

        self.mu = mu
        self.var = var

        return reconstructed

def latent_loss(z_mean, z_stdev):
    mean_sq = z_mean * z_mean
    stdev_sq = z_stdev * z_stdev

    return 0.5 * torch.mean(mean_sq + stdev_sq - torch.log(stdev_sq) - 1)

recon_loss = nn.MSELoss()

transform = transforms.Compose([transforms.ToTensor()])

mnist = torchvision.datasets.MNIST('./', download=True, transform=transform)

dataloader = torch.utils.data.DataLoader(mnist, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

vae = VAE()

optimizer = optim.Adam(vae.parameters(), lr=0.001)

for epoch_i in range(30):
    for data_batch in dataloader:
        X, y = data_batch
        X = X.reshape(BATCH_SIZE, INPUT_DIM)

        optimizer.zero_grad()

        X_reconstructed = vae(X)

        loss = recon_loss(X_reconstructed, X) + latent_loss(vae.mu, vae.var)
        loss.backward()

        optimizer.step()

    print(epoch_i, loss.item())

