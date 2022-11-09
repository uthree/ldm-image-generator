import torch
import torch.nn as nn
from vae import VAE
from ddpm import DDPM

class LDM():
    def __init__(self, ddpm: DDPM, vae: VAE):
        self.ddpm = ddpm
        self.vae = vae

    def caluclate_loss(self, image, condition=None):
        z, _ = self.vae.encode(image)
        return self.ddpm.caluclate_loss(z, condition=condition)

    def sample(self, latent_shape=(1, 8, 64, 64), seed=1, condition=None, style=None):
        z = self.ddpm.sample(x_shape=latent_shape, condition=condition, seed=seed)
        return self.vae.decode(z, style=style)
