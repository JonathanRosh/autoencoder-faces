import torch
import torch.nn as nn


class Autoencoder(nn.Module):
    """
    Simple conv autoencoder:
    - 4 downsampling conv blocks
    - linear bottleneck
    - 4 upsampling deconv blocks
    """

    def __init__(self, latent_dim=256, base_channels=16):
        super(Autoencoder, self).__init__()

        # Encoder: 256x256 -> 16x16
        self.encoder = nn.Sequential(
            nn.Conv2d(3, base_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(base_channels, base_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(base_channels, base_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(base_channels, base_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # bottleneck: (C,16,16) -> latent_dim -> (C,16,16)
        self.latent_dim = latent_dim
        self.flat_size = base_channels * 16 * 16

        self.latent_layer = nn.Linear(self.flat_size, self.latent_dim)
        self.unlatent_layer = nn.Linear(self.latent_dim, self.flat_size)

        # Decoder: 16x16 -> 256x256
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(base_channels, base_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.LeakyReLU(0.2, inplace=True),

            nn.ConvTranspose2d(base_channels, base_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.LeakyReLU(0.2, inplace=True),

            nn.ConvTranspose2d(base_channels, base_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.LeakyReLU(0.2, inplace=True),

            nn.ConvTranspose2d(base_channels, 3, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh(),
        )

    def forward(self, x):
        encoded = self.encoder(x)

        B, C, H, W = encoded.shape
        flat = encoded.view(B, -1)
        latent = self.latent_layer(flat)
        unlatent = self.unlatent_layer(latent)
        unflat = unlatent.view(B, C, H, W)

        output = self.decoder(unflat)
        return output
