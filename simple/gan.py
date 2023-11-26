"""
Code to train my first generative adversarial network
using Pytorch on MNIST dataset

GAN consists of two Neural Networks that are trained simultaneously
 -- Generator (Generates fake based on random noise distribution)
 -- Discriminator (Distinguish between generator output and actual data)

"""

""" System Module """
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
# For Standard datasets
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torchvision import transforms as trn
from torch.utils.tensorboard import SummaryWriter


""" Hyper parameters """

out_features_dim = 128
param_relu = 0.01
device = "cuda" if torch.cuda.is_available() else "cpu"
lr = 1e-2
xcap_dim = 64  # 128, 256
image_dim = 28 * 28 * 1  # MNIST image dimension
batch_size = 32
num_epochs = 100


class Discriminator(nn.Module):
    def __init__(self, img_dim):
        """
        Initialization for Discriminator class object. You'll
        get image features as input

        :param img_dim: image dimensions (wd X ht X c)
        """
        super(Discriminator, self).__init__()
        self.disc = nn.Sequential(
            nn.Linear(img_dim, out_features_dim),
            nn.LeakyReLU(param_relu),
            nn.Linear(out_features_dim, 1),  # 1 because we want binary classification for fake and real
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.disc(x)


class Generator(nn.Module):
    def __init__(self, xcap_dim, img_dim):
        """
        Initialization for Generator class object. You'll
        get image features as input

        :param img_dim: input image dimensions (wd X ht X c)
        :param xcap_dim: output image dimensions (wd X ht X c) flattened to d-dim
        """
        super(Generator, self).__init__()
        self.gen = nn.Sequential(
            nn.Linear(xcap_dim, 2 * out_features_dim),
            nn.LeakyReLU(param_relu),
            # 28 X 28 X 1 for MNIST grey scale image when flattened to 784, so d = 784
            nn.Linear(2 * out_features_dim, img_dim),
            # Ensure output in [-1, 1] as MNIST image bit map is normalized b/w -1 and 1
            # P.S. To use transforms.Normalize((0.1307,), (0.3081,))
            # you should multiply nn.Tanh() with 2.83 ≈ nn.Tanh() * 2.83 ≈ (-2.83, 2.83)
            nn.Tanh(),
        )

    def forward(self, x):
        return self.gen(x)


def main():
    my_disc = Discriminator(image_dim).to(device)
    my_gen = Generator(xcap_dim, image_dim).to(device)

    # Add fixed noise to see the change w.r.t to epochs
    fixed_noise = torch.rand((batch_size, xcap_dim)).to(device)

    # Transforms (only normalize)
    transforms = trn.Compose(
        [trn.ToTensor(), trn.Normalize(0.5, 0.5)],
    )
    dataset = datasets.MNIST(root='dataset/',
                             transform=transforms,
                             download=True
                             )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    opt_disc = optim.Adam(my_disc.parameters(), lr=lr)
    opt_gen = optim.Adam(my_gen.parameters(), lr=lr)

    # Binary classification so binary cross entropy
    my_loss = nn.BCELoss()

    # For tensorboard
    writer_fake = SummaryWriter(f"logs/fake")  # folder to store fake images
    writer_real = SummaryWriter(f"logs/real")  # folder to store real images

    step = 0

    # Training for Gen and Disc
    for epoch in range(num_epochs):
        for batch_idx, (real, _) in enumerate(loader):
            # Flatten 28X28X1 to 784
            real = real.view(-1, 784).to(device)
            bs = real.shape[0]
            #print(f"Our batch size: {bs}\n")

            ### Train Discriminator: MAX{ log(D(real)) + log(1 - D(G(x_cap))}
            # Standard Normal Gaussian noise
            noise = torch.rand(bs, xcap_dim).to(device)
            gen_fake = my_gen(noise)
            disc_real = my_disc(real).view(-1)

            ### 1 * log(D(real)) + (1-1) * log (1-D(G(x_cap)))
            # calculated for loss of discriminator as per BCE
            # Min(-l) = Max(l)
            loss_disc_real = my_loss(disc_real, torch.ones_like(disc_real))

            # We need gen_fake in training generator so detach it before Adam clears all gradient for this step
            disc_fake = my_disc(gen_fake).view(-1)
            ### 0 * log(D(real)) + (1-0) * log (1 - D(G(x_cap)))
            loss_disc_fake = my_loss(disc_fake, torch.zeros_like(disc_fake))

            loss_disc = (loss_disc_real + loss_disc_fake) / 2

            my_disc.zero_grad()
            loss_disc.backward(retain_graph=True)
            opt_disc.step()

            ### Train Generator: MIN {log(1 - D(G(x_cap))} <-->MAX {log(D(G(x_cap)))}
            disc_fake_from_gen = my_disc(gen_fake).view(-1)
            # Follow reasoning as in  loss_disc_real
            loss_gen = my_loss(disc_fake_from_gen, torch.ones_like(disc_fake_from_gen))

            my_gen.zero_grad()
            loss_gen.backward()
            opt_gen.step()

            # See visualization in TensorBoard

            if batch_idx % 625 == 0:
                print(
                    f"Epoch [{epoch+1}/{num_epochs}] Batch {batch_idx}/{len(loader)} \
                          Loss D: {loss_disc:.4f}, Loss G: {loss_gen:.4f}"
                )
                with torch.no_grad():
                    fake = my_gen(fixed_noise).reshape(-1, 1, 28, 28)
                    data = real.reshape(-1, 1, 28, 28)

                    img_grid_fake = torchvision.utils.make_grid(fake, normalize=True)
                    img_grid_real = torchvision.utils.make_grid(data, normalize=True)

                    writer_fake.add_image(
                        "Mnist Fake Images", img_grid_fake, global_step=step
                    )
                    writer_real.add_image(
                        "Mnist Real Images", img_grid_real, global_step=step
                    )
                    step += 1


if __name__ == '__main__':
    main()
