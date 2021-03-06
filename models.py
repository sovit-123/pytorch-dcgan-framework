import torch.nn as nn
import torch.nn.functional as F
import torch

class Generator(nn.Module):
    # init takes in the latent vector size
    def __init__(self, nz, n_channels=1):
        super(Generator, self).__init__()
        self.nz = nz
        self.first_out_channels = 1024
        self.n_channels = n_channels
        self.kernel_size = 4

        self.gen_model = nn.Sequential(
            # nz will be the input to the first convolution
            nn.ConvTranspose2d(
                self.nz, self.first_out_channels, kernel_size=self.kernel_size, 
                stride=1, padding=0, bias=False),
            nn.BatchNorm2d(self.first_out_channels),
            nn.ReLU(True),
            nn.ConvTranspose2d(
                self.first_out_channels, self.first_out_channels//2, 
                kernel_size=self.kernel_size, 
                stride=2, padding=1, bias=False),
            nn.BatchNorm2d(self.first_out_channels//2),
            nn.ReLU(True),
            nn.ConvTranspose2d(
                self.first_out_channels//2, self.first_out_channels//4, 
                kernel_size=self.kernel_size, 
                stride=2, padding=1, bias=False),
            nn.BatchNorm2d(self.first_out_channels//4),
            nn.ReLU(True),
            nn.ConvTranspose2d(
                self.first_out_channels//4, self.first_out_channels//8, 
                kernel_size=self.kernel_size, 
                stride=2, padding=1, bias=False),
            nn.BatchNorm2d(self.first_out_channels//8),
            nn.ReLU(True),
            nn.ConvTranspose2d(
                self.first_out_channels//8, self.n_channels, kernel_size=self.kernel_size, 
                stride=2, padding=1, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        out = self.gen_model(x)
        return out


class Discriminator(nn.Module):
    def __init__(self, n_channels=1):
        super(Discriminator, self).__init__()
        self.n_channels = n_channels
        self.first_out_channels = 64
        self.kernel_size = 4
        self.leaky_relu_neg_slope = 0.2

        self.conv_layers = nn.Sequential(
            nn.Conv2d(
                self.n_channels, self.first_out_channels, 
                kernel_size=self.kernel_size, 
                stride=2, padding=1, bias=False),
            nn.LeakyReLU(self.leaky_relu_neg_slope, inplace=True),
            nn.Conv2d(
                self.first_out_channels, self.first_out_channels*2, 
                kernel_size=self.kernel_size, 
                stride=2, padding=1, bias=False),
            nn.BatchNorm2d(self.first_out_channels*2),
            nn.LeakyReLU(self.leaky_relu_neg_slope, inplace=True),
            nn.Conv2d(
                self.first_out_channels*2, self.first_out_channels*4, 
                kernel_size=self.kernel_size, 
                stride=2, padding=1, bias=False),
            nn.BatchNorm2d(self.first_out_channels*4),
            nn.LeakyReLU(self.leaky_relu_neg_slope, inplace=True),
            nn.Conv2d(
                self.first_out_channels*4, self.first_out_channels*8, 
                kernel_size=self.kernel_size, 
                stride=2, padding=1, bias=False),
            nn.BatchNorm2d(self.first_out_channels*8),
            nn.LeakyReLU(self.leaky_relu_neg_slope, inplace=True),
        )

        self.linear_layers = nn.Sequential(
            nn.Linear(8192, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = torch.flatten(x, 1)
        x = self.linear_layers(x)
        return x