import torch
import torch.nn as nn


# def weights_init(m):
#     if type(m) == nn.Linear:
#         nn.init.xavier_uniform_(m.weight)
#         torch.nn.init.zeros_(m.bias)
from Constants import Constants


class Generator(torch.nn.Module):
    """
    A three hidden-layer generative neural network
    """

    def __init__(self, out_nodes=25):
        super(Generator, self).__init__()
        n_features = Constants.GAN_GENERATOR_IN_NODES

        self.hidden0 = nn.Sequential(
            nn.Linear(n_features, n_features),
            nn.LeakyReLU(0.2)
        )

        self.hidden1 = nn.Sequential(
            nn.Linear(n_features, n_features),
            nn.LeakyReLU(0.2)
        )

        self.hidden2 = nn.Sequential(
            nn.Linear(n_features, n_features),
            nn.LeakyReLU(0.2)
        )

        self.out = nn.Sequential(
            nn.Linear(n_features, out_nodes),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.hidden0(x)
        x = self.hidden1(x)
        x = self.hidden2(x)
        x = self.out(x)
        return x


class Discriminator(torch.nn.Module):
    """
    A three hidden-layer discriminative neural network
    """

    def __init__(self, in_nodes=25):
        super(Discriminator, self).__init__()
        n_out = 1

        self.hidden0 = nn.Sequential(
            nn.Linear(in_nodes, 25),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3)
        )
        self.hidden1 = nn.Sequential(
            nn.Linear(25, 25),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3)
        )
        self.hidden2 = nn.Sequential(
            nn.Linear(25, 25),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3)
        )
        self.out = nn.Sequential(
            torch.nn.Linear(25, n_out),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        x = self.hidden0(x)
        x = self.hidden1(x)
        x = self.hidden2(x)
        x = self.out(x)
        return x
