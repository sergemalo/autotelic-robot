"""
Beta-VAE encoder trained online from replay buffer images.

Serves the same interface as R3MEncoder but is trainable and
additionally exposes sample() for goal imagination (future use).
"""
import logging
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from encoders.base import BaseEncoder

logger = logging.getLogger(__name__)


class ConvEncoder(nn.Module):
    """Simple convolutional encoder: (B,3,H,W) → (B, hidden_dim)."""

    def __init__(self, channels: List[int]):
        super().__init__()
        layers = []
        in_ch = 3
        for out_ch in channels:
            layers += [
                nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1),
                nn.ReLU(inplace=True),
            ]
            in_ch = out_ch
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ConvDecoder(nn.Module):
    """Simple transposed-conv decoder: (B, latent_dim) → (B, 3, H, W)."""

    def __init__(self, latent_dim: int, channels: List[int], output_size: int):
        super().__init__()
        self.first_ch = channels[0]
        # Compute the spatial size after encoding (depends on # of stride-2 layers)
        n_layers = len(channels) + 1  # +1 for the latent projection
        self.spatial = output_size // (2 ** (n_layers - 1))

        self.fc = nn.Linear(latent_dim, channels[0] * self.spatial * self.spatial)

        layers = []
        in_ch = channels[0]
        for out_ch in channels[1:]:
            layers += [
                nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1),
                nn.ReLU(inplace=True),
            ]
            in_ch = out_ch
        layers += [
            nn.ConvTranspose2d(in_ch, 3, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        ]
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.fc(z)
        x = x.view(x.size(0), self.first_ch, self.spatial, self.spatial)
        return self.net(x)


class VAEEncoder(BaseEncoder):
    """
    Beta-VAE encoder trained online.

    The VAE is periodically retrained on images from the replay buffer.
    encode() returns the mean of the posterior (no sampling at test/act time).
    """

    def __init__(
        self,
        latent_dim: int,
        encoder_channels: List[int],
        decoder_channels: List[int],
        beta: float,
        camera_key: str,
        device: torch.device,
        image_size: int = 224,
    ):
        super().__init__(camera_key=camera_key, device=device)

        self._latent_dim = latent_dim
        self.beta = beta

        self.conv_encoder = ConvEncoder(encoder_channels).to(device)

        # Compute flattened conv output size with a dummy forward pass
        with torch.no_grad():
            dummy = torch.zeros(1, 3, image_size, image_size, device=device)
            conv_out = self.conv_encoder(dummy)
            self._conv_flat = int(np.prod(conv_out.shape[1:]))

        self.fc_mu = nn.Linear(self._conv_flat, latent_dim).to(device)
        self.fc_logvar = nn.Linear(self._conv_flat, latent_dim).to(device)

        self.decoder = ConvDecoder(
            latent_dim=latent_dim,
            channels=decoder_channels,
            output_size=image_size,
        ).to(device)

        logger.info(
            "VAE encoder created: latent_dim=%d, beta=%.1f, conv_flat=%d",
            latent_dim, beta, self._conv_flat,
        )

    @property
    def latent_dim(self) -> int:
        return self._latent_dim

    def _encode_raw(self, img: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (mu, logvar) for a batch of images."""
        h = self.conv_encoder(img)
        h = h.view(h.size(0), -1)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterise(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def encode(self, obs: Dict[str, np.ndarray]) -> torch.Tensor:
        """
        Returns the posterior mean (no noise) — used for policy/reward.
        """
        img = self.obs_to_image_tensor(obs)
        with torch.no_grad():
            mu, _ = self._encode_raw(img)
        return mu

    def encode_tensor(self, img: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Used during VAE training — returns (mu, logvar)."""
        return self._encode_raw(img)

    def loss(self, img: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Beta-VAE loss.

        Returns:
            total_loss, reconstruction_loss, kl_loss
        """
        mu, logvar = self._encode_raw(img)
        z = self.reparameterise(mu, logvar)
        recon = self.decode(z)

        recon_loss = F.binary_cross_entropy(recon, img, reduction="sum") / img.size(0)
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        total = recon_loss + self.beta * kl_loss

        return total, recon_loss, kl_loss