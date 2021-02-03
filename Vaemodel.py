import torch
from torch import nn
import pandas as pd, numpy as np
import torch.nn.functional as F


class VAE(nn.Module):

    def __init__(self,latent_dim):
        super(VAE, self).__init__()
        
        # Encoder Layers
        self.enc_layer1 = nn.Linear(18,10)
        self.enc_bn = nn.BatchNorm1d(num_features=10)
        self.enc_layer2 = nn.Linear(10,latent_dim)
        self.enc_layer3 = nn.Linear(10,latent_dim)

        # Decoder Layers
        self.dec_layer1 = nn.Linear(latent_dim,10)
        self.dec_bn = nn.BatchNorm1d(num_features=10)
        self.dec_layer2 = nn.Linear(10,18)
    
    def encode(self,x):
        x = self.enc_layer1(x)
        x = F.relu(x)
        x = self.enc_bn(x)

        # Calc the mu and sigma - note log sigma is taken to avoid negative variance estimates
        z_mu = self.enc_layer2(x)
        z_log_sigma = self.enc_layer3(x)

        # Add the random value sampled from the norm distribution
        epsilon = torch.cuda.FloatTensor(z_log_sigma.size()).normal_()
        z_sigma = z_log_sigma.mul(0.5).exp_()
        z = z_mu + epsilon.mul(z_sigma)
        return z, z_mu, z_log_sigma
    
    def decode(self,x):
        x = self.dec_layer1(x)
        x = self.dec_bn(x)
        x = F.relu(x)
        x = self.dec_layer2(x)
        return x
    
    def forward(self, x):
        ''' Returns z_mu and z_log_sigma to calculate the loss'''
        z,z_mu,z_log_sigma = self.encode(x)
        return self.decode(z), z_mu, z_log_sigma
    
    def loss_function(self,recon_x, x, mu, logvar):
        """
        recon_x: generating images
        x: origin images
        mu: latent mean
        logvar: latent log variance
        """
        reconstruction_function = nn.MSELoss(size_average=False)
        BCE = reconstruction_function(recon_x, x)  # mse loss
        # loss = 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
        KLD = torch.sum(KLD_element).mul_(-0.5)
        # KL divergence
        return BCE + KLD