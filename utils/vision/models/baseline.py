import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.distributions import transforms as tT
from torchvision.datasets import MNIST #
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from mpl_toolkits.axes_grid1 import ImageGrid
from torchvision.utils import save_image, make_grid
from torch.distributions import Normal, Independent, kl
from torch.autograd import Variable
import math
import torch.nn.functional as F


class VAE(nn.Module):
    def __init__(self,  device=None, input_dim = None, latent_dim = None):
        super(VAE, self).__init__()
        self.device = device
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        self.encoder_1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size = 3, padding = 1, stride=2 ), 
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(kernel_size =  2),
            nn.Conv2d(32, 64, kernel_size = 3, padding = 1, stride=2), 
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 64, kernel_size = 3, padding = 1, stride=2), 
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(kernel_size =  2),
            nn.Conv2d(64, 128, kernel_size = 3, padding = 1, stride=2), 
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Flatten(),
        )

        # latent mean and variance
        self.mean_layer = nn.Linear(128,32)
        self.logvar_layer = nn.Linear(128,32)

        self.decoder_input = nn.Linear(32, 128 * 8 * 8)

        self.decoder_1 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size = 3, padding = 1, stride=2), # deconv1
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(64, 64, kernel_size = 3,  stride=2), # deconv3
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(64, 32, kernel_size = 3,  stride=2), # deconv4
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(32, self.input_dim,  kernel_size = 2, stride=1), 
        )


    def encoder(self,x):
        s = self.encoder_1(x)
        return s

    def decoder(self, x):
        # reshape
        x_hat = self.decoder_1(x)
        return x_hat


    def calc_same_pad(self, i: int, k: int, s: int, d: int) -> int:
        return max((math.ceil(i / s) - 1) * s + (k - 1) * d + 1 - i, 0)
    
    def apply_same_padding(self, x: torch.Tensor):
        # same padding 1
        ih, iw = x.size()[-2:]

        kernel_size = 3
        stride = 2
        dilation  = 1
        

        pad_h = self.calc_same_pad(i=ih, k=kernel_size, s=stride, d=dilation)
        pad_w = self.calc_same_pad(i=iw, k=kernel_size, s=stride, d=dilation)

        if pad_h > 0 or pad_w > 0:
            x = F.pad(
                x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2]
            )
        return x

    def gaussian_noise(self,input, dim, mean, std):

        #output = Variable(ins.data.new(ins.size()).normal_(mean, std))
        output = 0
        return output

    def encode(self, x):
        x = self.encoder(x)
        mean , logvar = self.mean_layer(x), self.logvar_layer(x)
        return mean, logvar
    

    def reparameterization(self, mean, var):
        std = torch.exp(0.5*var)
        epsilon = torch.randn_like(std).to(self.device)      
        z =  mean + std*epsilon
        return z

    
    def decode(self, x):
        decoder_input = self.decoder_input(x)
        decoder_input = decoder_input.view(-1,128, 8, 8)
        d =  self.decoder(decoder_input)
        return d
    
    def get_latent_space_vector(self,x):
        mean, log_var = self.encode(x)
        return self.reparameterization(mean, log_var)
    
    def forward(self,x):
        mean, log_var = self.encode(x)
        z = self.reparameterization(mean, log_var)
        x_hat = self.decode(z)
        return x_hat, mean, log_var