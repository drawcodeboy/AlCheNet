import math

import torch
import torch.nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

class VaDE(torch.nn.Module):
    """Variational Deep Embedding(VaDE).

    Args:
        n_classes (int): Number of clusters.
        data_dim (int): Dimension of observed data.
        latent_dim (int): Dimension of latent space.
    """
    def __init__(self, n_classes, data_dim, latent_dim):
        super(VaDE, self).__init__()

        self._pi = Parameter(torch.zeros(n_classes))
        self.mu = Parameter(torch.randn(n_classes, latent_dim))
        self.logvar = Parameter(torch.randn(n_classes, latent_dim))

        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(data_dim, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 2048),
            torch.nn.ReLU(),
        )
        self.encoder_mu = torch.nn.Linear(2048, latent_dim)
        self.encoder_logvar = torch.nn.Linear(2048, latent_dim)

        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(latent_dim, 2048),
            torch.nn.ReLU(),
            torch.nn.Linear(2048, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, data_dim),
            torch.nn.Sigmoid(),
        )

    @property
    def weights(self):
        return torch.softmax(self._pi, dim=0)

    def encode(self, x):
        h = self.encoder(x)
        mu = self.encoder_mu(h)
        logvar = self.encoder_logvar(h)
        return mu, logvar

    def decode(self, z):
        return self.decoder(z)
    
    def reparameterize(mu, logvar):
        """Reparameterization trick.
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar

    def classify(self, x, n_samples=8):
        with torch.no_grad():
            mu, logvar = self.encode(x)
            z = torch.stack(
                [_reparameterize(mu, logvar) for _ in range(n_samples)], dim=1)
            z = z.unsqueeze(2)
            h = z - self.mu
            h = torch.exp(-0.5 * torch.sum(h * h / self.logvar.exp(), dim=3))
            # Same as `torch.sqrt(torch.prod(self.logvar.exp(), dim=1))`
            h = h / torch.sum(0.5 * self.logvar, dim=1).exp()
            p_z_given_c = h / (2 * math.pi)
            p_z_c = p_z_given_c * self.weights
            y = p_z_c / torch.sum(p_z_c, dim=2, keepdim=True)
            y = torch.sum(y, dim=1)
            pred = torch.argmax(y, dim=1)
        return pred