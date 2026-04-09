"""
Base encoder interface.
All encoders must implement this contract so the rest of the
system never depends on a concrete implementation.
"""
from abc import ABC, abstractmethod
from typing import Dict

import numpy as np
import torch
import torch.nn as nn


class BaseEncoder(ABC, nn.Module):
    """
    Abstract encoder: raw obs dict → latent vector z.

    Subclasses must implement:
        - encode(obs) → Tensor[B, latent_dim]
        - latent_dim (property)
    """

    def __init__(self, camera_key: str, device: torch.device):
        super().__init__()
        self.camera_key = camera_key
        self.device = device

    @property
    @abstractmethod
    def latent_dim(self) -> int:
        """Dimensionality of the latent space."""
        ...

    @abstractmethod
    def encode(self, obs: Dict[str, np.ndarray]) -> torch.Tensor:
        """
        Encode a batch of observations into latent vectors.

        Args:
            obs: dict with at minimum self.camera_key → np.ndarray (B, H, W, 3) uint8

        Returns:
            z: Tensor of shape (B, latent_dim) on self.device
        """
        ...

    def obs_to_image_tensor(self, obs: Dict[str, np.ndarray]) -> torch.Tensor:
        """
        Utility: extract the camera image from an obs dict and convert to
        a normalised float tensor of shape (B, 3, H, W) in [0, 1].

        Handles both single obs (H, W, 3) and batched (B, H, W, 3).
        """
        img = obs[self.camera_key]  # uint8 numpy
        if img.ndim == 3:
            img = img[np.newaxis]  # (1, H, W, 3)
        # (B, H, W, 3) → (B, 3, H, W)
        img = torch.from_numpy(img).permute(0, 3, 1, 2).float().to(self.device)
        img = img / 255.0
        return img

    def forward(self, obs: Dict[str, np.ndarray]) -> torch.Tensor:
        """Alias for encode — allows use as nn.Module."""
        return self.encode(obs)
