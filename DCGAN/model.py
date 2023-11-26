"""
Discriminator and Generator implementation from DCGAN paper

"""

import torch
import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self, channels_img, features_d):
        """
        Initialize the Discriminator

        :param channels_img: Number of channels of the image
        :param features_d: Dim of output features
        """
        super(Discriminator, self).__init__()
        self.disc = nn.Sequential(
            # Input shape: N X channels_img * 64 X 64
            nn.Conv2d(
                channels_img,
                features_d,
                kernel_size=4,
                stride=2,
                padding=1
            ),  # 32 X 32
            nn.LeakyReLU(0.2),
            self._block(features_d, features_d * 2, 4, 2, 1),  # 16 X 16
            self._block(features_d * 2, features_d * 2 * 2, 4, 2, 1),  # 8 X 8
            self._block(features_d * 2 * 2, features_d * 2 * 2 * 2, 4, 2, 1),  # 4 X 4
            nn.Conv2d(
                features_d * 2 * 2 * 2,
                1,
                kernel_size=4,
                stride=2,
                padding=0
            ),  # 1 X 1
            nn.Sigmoid(),
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(in_channels,
                      out_channels,
                      kernel_size,
                      stride,
                      padding,
                      bias=False,
                      ),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2),

        )

    def forward(self, x):
        return self.disc(x)


class Generator(nn.Module):
    def __init__(self, z_dim, channels_img, features_g):
        """

        :param z_dim:
        :param channels_img:
        :param features_g:
        """
        super(Generator, self).__init__()
        self.gen = nn.Sequential(
            # Input N X z_dim X 1 X 1
            self._block(z_dim, features_g * 16, 4, 1, 0),  # N X features_g * 16 X 4 X 4
            self._block(features_g * 16, features_g * 8, 4, 2, 1),  # 8 X 8
            self._block(features_g * 8, features_g * 4, 4, 2, 1),  # 16 X 16
            self._block(features_g * 4, features_g * 2, 4, 2, 1),  # 32 X 32
            nn.ConvTranspose2d(
                features_g * 2,
                channels_img,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.Tanh(),  # Bound output to [-1, 1]
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels,
                               out_channels,
                               kernel_size,
                               stride,
                               padding,
                               bias=False,
                               ),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(),
        )

    def forward(self, x):
        return self.gen(x)


def initialize_weights(model):
    """
    Initializes weights according to the DCGAN paper

    :param model: Created model
    :return:
    """
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            # Initialize weights with mean=0 and SD=0.2
            nn.init.normal_(m.weight.data, 0.0, 0.02)


def test():
    # 8 images in batch with dimensions as 64 X 64 X 3
    N, in_channels, H, W = 8, 3, 64, 64

    # Noise dimension
    noise_dim = 100
    x = torch.randn((N, in_channels, H, W))
    disc = Discriminator(in_channels, 8)
    initialize_weights(disc)
    assert disc(x).shape == (N, 1, 1, 1), "Discriminator test failed"
    gen = Generator(noise_dim, in_channels, 8)
    initialize_weights(gen)
    z = torch.randn((N, noise_dim, 1, 1))
    assert gen(z).shape == (N, in_channels, H, W), "Generator test failed"
    print("Success, tests passed!")


if __name__ == "__main__":
    test()
