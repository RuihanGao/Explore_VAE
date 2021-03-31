import os, sys
import numpy as np
import pickle
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torch.distributions as torchD

import torch, seaborn as sns
import pandas as pd

from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import ListedColormap

from utils import *

"""
Copy the network structures here for easier import during testing and plotting
"""

class MLP_Encoder(nn.Module):
    def __init__(self, **kwargs):
        super(MLP_Encoder, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(in_features=np.prod(kwargs["input_shape"]), out_features=400),
            nn.ReLU(),
            nn.Linear(in_features=400, out_features=kwargs["latent_dim"])
        )
    
    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        latent = self.model(x)
        return latent

class MLP_Decoder(nn.Module):
    def __init__(self, **kwargs):
        super(MLP_Decoder, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(in_features=kwargs["latent_dim"], out_features=400),
            nn.ReLU(),
            nn.Linear(in_features=400, out_features=np.prod(kwargs["input_shape"])),
            nn.Sigmoid() # push the pixels in range (0,1)
        )
        self.output_shape = kwargs["input_shape"]
    
    def forward(self, latent):
        x_bar = self.model(latent)
        x_bar = x_bar.view([-1]+ self.output_shape)      
        return x_bar
    
class MLP_AE(nn.Module):
    def __init__(self, **kwargs):
        # kwargs["input_shape"] = [1,28,28]
        # kwargs["latent_dim"] = 4
        super(MLP_AE, self).__init__()
        self.encoder = MLP_Encoder(**kwargs)
        self.decoder = MLP_Decoder(**kwargs)
        
    def forward(self, x):
        latent = self.encoder(x)
        x_bar = self.decoder(latent)
        return latent, x_bar

    def sample_latent_embedding(self, latent, sd=1, N_samples=1):
        """
        AE returns scalar value and we use that as mean and predefined default value for standard deviation (sd)
        """
        dist = torchD.Normal(latent, sd)
        embedding = dist.sample((N_samples,))
        return embedding

class MLP_V_Encoder(nn.Module):
    def __init__(self, **kwargs):
        super(MLP_V_Encoder, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(in_features=np.prod(kwargs["input_shape"]), out_features=kwargs["enc_dim"]),
            nn.ReLU(),
        )
    
    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        enc_out = self.model(x)
        return enc_out

class MLP_V_Decoder(nn.Module):
    def __init__(self, **kwargs):
        super(MLP_V_Decoder, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(in_features=kwargs["latent_dim"], out_features=kwargs["enc_dim"]),
            nn.ReLU(),
            nn.Linear(in_features=kwargs["enc_dim"], out_features=np.prod(kwargs["input_shape"])),
            nn.Sigmoid() # push the pixels in range (0,1)   
        )
        self.output_shape = kwargs["input_shape"]
    
    def forward(self, latent):
        x_bar = self.model(latent)
        x_bar = x_bar.view([-1]+ self.output_shape)      
        return x_bar
    
class MLP_VAE(nn.Module):
    """
    TODO: check whether to use sum or mean for the probability part
    """
    def __init__(self, **kwargs):
        # kwargs["input_shape"] = [1,28,28]
        # kwargs["latent_dim"] = 4
        super(MLP_VAE, self).__init__()
        self.encoder = MLP_V_Encoder(**kwargs)
        self.decoder = MLP_V_Decoder(**kwargs)
        
        # distribution layers
        self.enc_dim = kwargs["enc_dim"]
        self.latent_dim = kwargs["latent_dim"]
        self.enc_to_mean = nn.Linear(self.enc_dim, self.latent_dim)
        self.enc_to_logvar = nn.Linear(self.enc_dim, self.latent_dim)
    
    def encode(self, x):
        enc_out = self.encoder(x)
        mean = self.enc_to_mean(enc_out)
        logvar = self.enc_to_logvar(enc_out)
        return mean, logvar
    
    def decode(self, latent):
        return self.decoder(latent)
    
    def pxz_likelihood(self, x, x_bar, scale=1., dist_type="Gaussian"):
        """
        compute the likelihood p(x|z) based on predefined distribution, given a latent vector z
        default scale = 1, can be broadcasted to the shape of x_bar
        """
        if dist_type == "Gaussian":
            dist = torch.distributions.Normal(loc=x_bar, scale=scale)
        else:
            raise NotImplementedError("unknown distribution for p(x|z) {}".format(dist_type))
        
        log_pxz = dist.log_prob(x)
        return log_pxz.sum() # log_pxz.sum((1,2,3))
    
    def kl_divergence(self, mean, logvar):
        """
        Monte Carlo way to solve KL divergence
        """
        pz = torchD.Normal(torch.zeros_like(mean), scale=1)
        std = torch.exp(0.5*logvar)
        qzx = torchD.Normal(loc=mean, scale=std)
        
        z = qzx.rsample() # reparameterized sampling, shape [32,2]
        
        # clamp the log prob to avoid -inf
        qzx_lp = qzx.log_prob(z).clamp(min=-1e10, max=0.)
        pz_lp = pz.log_prob(z).clamp(min=-1e10, max=0.)

        kl = qzx_lp - pz_lp
        if torch.isnan(qzx_lp).any():
            raise ValueError("nan in qzx_lp")
        if torch.isnan(pz_lp).any():
            raise ValueError("nan in pz_lp")
        if torch.isnan(kl.mean()).any():
            raise ValueError("nan in kl")
        return kl.sum()
    
    def reparameterize(self, mean, logvar):
        # assume Gaussian for p(epsilon)
        sd = torch.exp(0.5*logvar)
        # use randn_like to sample N(0,1) of the same size as std/mean
        # default only sample once, otherwise should try sample multiple times take mean
        eps = torch.randn_like(sd) 
        return mean + sd * eps
    
    def sample_latent_embedding(self, mean, logvar, method="reparameterize"):
        """
        Write a sampling function to make function name consistent
        """
        if method=="reparameterize":
            return self.reparameterize(mean, logvar)
        else:
            raise NotImplementedError("Unrecognized method for sampling latent embedding {}".format(method))
    
    def forward(self, x, if_plot_pq=False):
        latent_mean, latent_logvar = self.encode(x)
        latent = self.reparameterize(latent_mean, latent_logvar)
        x_bar = self.decoder(latent)
        
        if if_plot_pq:
            plot_p_q(latent_mean, latent_logvar)
            
        return latent, x_bar, latent_mean, latent_logvar
    
class MLP_CV_Encoder(nn.Module):
    def __init__(self, **kwargs):
        super(MLP_CV_Encoder, self).__init__()
        
        self.model = nn.Sequential(
            nn.Linear(in_features=(np.prod(kwargs["input_shape"])+1), out_features=kwargs["enc_dim"]),
            nn.ReLU(),
        )
    
    def forward(self, x, y):
        y = torch.unsqueeze(y,1)
        x = torch.flatten(x, start_dim=1)
        x = torch.cat([x, y], dim = 1)
        enc_out = self.model(x)
        return enc_out

class MLP_CV_Decoder(nn.Module):
    def __init__(self, **kwargs):
        super(MLP_CV_Decoder, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(in_features=(kwargs["latent_dim"]+1), out_features=kwargs["enc_dim"]),
            nn.ReLU(),
            nn.Linear(in_features=kwargs["enc_dim"], out_features=np.prod(kwargs["input_shape"])),
            nn.Sigmoid() # push the pixels in range (0,1)   
        )
        self.output_shape = kwargs["input_shape"]
    
    def forward(self, latent, y):
        y = torch.unsqueeze(y,1)
        latent = torch.cat([latent, y], dim=1)
        x_bar = self.model(latent)
        x_bar = x_bar.view([-1]+ self.output_shape)      
        return x_bar
    
class MLP_CVAE(nn.Module):
    """
    TODO: check whether to use sum or mean for the probability part
    """
    def __init__(self, **kwargs):
        # e.g.
        # kwargs["input_shape"] = [1,28,28]
        # kwargs["latent_dim"] = 4
        super(MLP_CVAE, self).__init__()
        self.encoder = MLP_CV_Encoder(**kwargs)
        self.decoder = MLP_CV_Decoder(**kwargs)
        
        # distribution layers
        self.enc_dim = kwargs["enc_dim"]
        self.latent_dim = kwargs["latent_dim"]
        self.enc_to_mean = nn.Linear(self.enc_dim, self.latent_dim)
        self.enc_to_logvar = nn.Linear(self.enc_dim, self.latent_dim)
    
    def encode(self, x, y):
        enc_out = self.encoder(x, y)
        mean = self.enc_to_mean(enc_out)
        logvar = self.enc_to_logvar(enc_out)
        return mean, logvar
    
    def decode(self, latent, y):
        return self.decoder(latent, y)
    
    def pxz_likelihood(self, x, x_bar, scale=1., dist_type="Gaussian"):
        """
        compute the likelihood p(x|z) based on predefined distribution, given a latent vector z
        default scale = 1, can be broadcasted to the shape of x_bar
        """
        if dist_type == "Gaussian":
            dist = torch.distributions.Normal(loc=x_bar, scale=scale)
        else:
            raise NotImplementedError("unknown distribution for p(x|z) {}".format(dist_type))
        
        log_pxz = dist.log_prob(x)
        return log_pxz.sum() # log_pxz.sum((1,2,3))
    
    def kl_divergence(self, mean, logvar):
        """
        Monte Carlo way to solve KL divergence
        """
        pz = torchD.Normal(torch.zeros_like(mean), scale=1)
        std = torch.exp(0.5*logvar)
        qzx = torchD.Normal(loc=mean, scale=std)
        
        z = qzx.rsample() # reparameterized sampling, shape [32,2]
        
        # clamp the log prob to avoid -inf
        qzx_lp = qzx.log_prob(z).clamp(min=-1e10, max=0.)
        pz_lp = pz.log_prob(z).clamp(min=-1e10, max=0.)

        kl = qzx_lp - pz_lp
        if torch.isnan(qzx_lp).any():
            raise ValueError("nan in qzx_lp")
        if torch.isnan(pz_lp).any():
            raise ValueError("nan in pz_lp")
        if torch.isnan(kl.mean()).any():
            raise ValueError("nan in kl")

        return kl.sum()
    
    def reparameterize(self, mean, logvar):
        # assume Gaussian for p(epsilon)
        sd = torch.exp(0.5*logvar)
        # use randn_like to sample N(0,1) of the same size as std/mean
        # default only sample once, otherwise should try sample multiple times take mean
        eps = torch.randn_like(sd) 
        return mean + sd * eps
    
    def sample_latent_embedding(self, mean, logvar, method="reparameterize"):
        """
        Write a sampling function to make function name consistent
        """
        if method=="reparameterize":
            return self.reparameterize(mean, logvar)
        
        else:
            raise NotImplementedError("Unrecognized method for sampling latent embedding {}".format(method))
    
    def forward(self, x, y, if_plot_pq=False):
        latent_mean, latent_logvar = self.encode(x, y)
        latent = self.reparameterize(latent_mean, latent_logvar)
        x_bar = self.decoder(latent, y)
        
        if if_plot_pq:
            plot_p_q(latent_mean, latent_logvar)
            
        return latent, x_bar, latent_mean, latent_logvar
