import os, sys
import numpy as np
import pickle
import matplotlib.pyplot as plt

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

def plot_latent(labels, latents, dtype="array", cmap="tab10", add_legend=False, suptitle_app="_train", ax=None, bbox_to_anchor=(1.05, 1), **plt_kwargs):
    """
    plot 2D visualization for latent space
    params dtype: "array" or "tensor"
    """
    
    if dtype == "tensor":
        labels = labels.cpu().detach().numpy()
        latents = latents.cpu().detach().numpy()
#     print("labels {} {}".format(type(labels), labels.shape))
#     print("latents {} {}".format(type(latents), latents.shape))
    
    if ax is None:
        # without specifying ax, create a new plot here. Otherwise, can use plt.gca(), see https://towardsdatascience.com/creating-custom-plotting-functions-with-matplotlib-1f4b8eba6aa1
        fig, ax = plt.subplots()
        # ax.scatter(test_latents[:,0], test_latents[:,1], c=test_labels) # fail to set labels
        for y in np.unique(labels):
            i = np.where(labels == y)
            ax.scatter(latents[i,0], latents[i,1], label=y, cmap=cmap, **plt_kwargs)
        if add_legend:
            ax.legend(bbox_to_anchor=bbox_to_anchor)
        plt.title("Latent space"+suptitle_app)
        plt.show()
    else:
        for y in np.unique(labels):
            i = np.where(labels==y)
            ax.scatter(latents[i,0], latents[i,1], label=y, cmap=cmap, **plt_kwargs)
        ax.set_title("Latent space"+suptitle_app)
        if add_legend:
            ax.legend(bbox_to_anchor=bbox_to_anchor)
        
        return ax
        
    
    
def plot_p_q(mean, logvar, pz_scale=1., N_samples=100, add_legend=False, suptitle_app="_train"):
    
    """
    Visualize the p(z) and q(z|x) distributions
    """

    pz = torchD.Normal(torch.zeros_like(mean), scale=pz_scale) # assume constant scale
    std = torch.exp(0.5*logvar)
    qzx = torchD.Normal(loc=mean, scale=std)

    print("Plot bivariate latent distributions")
    print("pz batch_shape {}, event_shape {}".format(pz.batch_shape, pz.event_shape))
    print("qzx batch_shape {}, event_shape {}".format(qzx.batch_shape, qzx.event_shape))
    pz_samples = pz.sample((N_samples,)).cpu().detach().numpy() #shape (1000, 32, 2)
    qzx_samples = qzx.sample((N_samples,)).cpu().detach().numpy()

    sample_dim, batch_dim, latent_dim = pz_samples.shape
    print("check p, q shape, pz {}, qzx {}".format(pz_samples.shape, qzx_samples.shape))
    
    # 1D histograms as subplots
    fig, axes = plt.subplots(nrows=2, ncols=latent_dim, figsize=(12, 12))
    for i in range(latent_dim):
        sns.histplot(pz_samples[...,i], kde=True, ax=axes[0,i], legend=add_legend)
        sns.histplot(qzx_samples[...,i], kde=True, ax=axes[1,i], legend=add_legend)
        
    cols_header = ["Latent {}".format(i) for i in range(latent_dim)]
    rows_header = ["pz", "qzx"]

    for ax, col in zip(axes[0], cols_header):
        ax.set_title(col)

    for ax, row in zip(axes[:,0], rows_header):
        ax.set_ylabel(row, rotation=0, size='large')

    plt.suptitle("Bivariate Latent Distributions"+suptitle_app)
    plt.show()
    
    # 2D histplot by seaborn  
    df_pz = pd.DataFrame(pz_samples.reshape(-1, latent_dim), columns=["Latent {}".format(i) for i in range(latent_dim)]) #
    df_pz.index = np.tile(np.arange(pz_samples.shape[1]), pz_samples.shape[0]) + 1
    df_pz.index.name = 'Batch'
    
    df_qzx = pd.DataFrame(qzx_samples.reshape(-1, latent_dim), columns=["Latent {}".format(i) for i in range(latent_dim)]) #
    df_qzx.index = np.tile(np.arange(qzx_samples.shape[1]), qzx_samples.shape[0]) + 1
    df_qzx.index.name = 'Batch'
    
    fig, axes = plt.subplots(nrows=2,ncols=1, figsize=(12, 12))
    sns.histplot(df_pz, x="Latent 0", y="Latent 1", hue="Batch", kde=True, ax=axes[0], palette="bright", legend=add_legend)
    sns.histplot(df_qzx, x="Latent 0", y="Latent 1", hue="Batch", kde=True, ax=axes[1], palette="bright", legend=add_legend)

    plt.suptitle("Scatterplot of samples"+suptitle_app)
    plt.show()
    
def sample_latent_embedding(latent, sd=1, N_samples=1):
    """
    AE returns scalar value and we use that as mean and predefined default value for standard deviation (sd)
    equivalently, use model.sample_latent_embedding
    """
    dist = torchD.Normal(latent, sd)
    embedding = dist.sample((N_samples,))
#         print("sample z for AE, sample_shape {}, batch_shape {}, event_shape {}".format(embedding.shape, dist.batch_shape, dist.event_shape))
    return embedding


def generate_data_cond_y(model, y, num_samples=10, latent_dim=2, device="cpu"):
    """
    Prepared for CVAE to generate images with specified label
    """
    if isinstance(y, int):
        y = np.repeat(y, num_samples)
    else:
        num_samples = len(y)

    y = torch.from_numpy(y).float().to(device)
    z = torch.randn(num_samples, latent_dim).to(device)
    assert z.shape[0]==y.shape[0], "inconsistent shape z {}, y {}".format(z.shape, y.shape)
    recon_from_embeddings = model.decoder(z, y)
    y = y.cpu().detach().numpy(). astype(int)
    plt.figure(figsize=(num_samples,1))
    for i, recon in enumerate(recon_from_embeddings.cpu().detach().numpy()):
        plt.subplot(1, num_samples, i+1)
        plt.imshow(recon[0])
        plt.title("{}".format(y[i]))
#     plt.suptitle("Sampling for label {}".format(y))