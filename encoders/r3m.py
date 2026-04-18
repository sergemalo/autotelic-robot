"""
R3M encoder wrapper.
Loads a pretrained R3M model and uses it as a frozen feature extractor.

R3M paper: https://arxiv.org/abs/2203.12601
"""
import logging
from typing import Dict

import numpy as np
import torch

from encoders.base import BaseEncoder

logger = logging.getLogger(__name__)

# R3M output dims per backbone
_R3M_DIMS = {
    "resnet18": 512,
    "resnet34": 512,
    "resnet50": 2048,
}


class R3MEncoder(BaseEncoder):
    """
    Frozen R3M encoder.

    The R3M model is loaded via the official r3m package.
    Gradients are never computed through this encoder.
    """

    def __init__(self, model_name: str, camera_key: str, device: torch.device):
        super().__init__(camera_key=camera_key, device=device)

        if model_name not in _R3M_DIMS:
            raise ValueError(
                f"Unknown R3M model '{model_name}'. "
                f"Choose from {list(_R3M_DIMS.keys())}."
            )

        self._latent_dim = _R3M_DIMS[model_name]
        logger.info("Loading R3M model: %s (latent_dim=%d)", model_name, self._latent_dim)

        from r3m import load_r3m  # imported here to keep dependency optional
        self.model = load_r3m(model_name)
        self.model = self.model.to(device)
        self.model.eval()

        # Freeze all parameters
        for param in self.model.parameters():
            param.requires_grad = False

        logger.info("R3M encoder loaded and frozen.")

    @property
    def latent_dim(self) -> int:
        return self._latent_dim

    @torch.no_grad()
    def encode(self, obs: Dict[str, np.ndarray]) -> torch.Tensor:
        """
        Args:
            obs: dict containing self.camera_key → uint8 numpy image

        Returns:
            z: Tensor (B, latent_dim)
        """
        img = self.obs_to_image_tensor(obs)  # (B, 3, H, W) in [0, 1]
        # R3M expects [0, 255] range — rescale back
        img = img * 255.0
        z = self.model(img)                  # (B, latent_dim)
        return z