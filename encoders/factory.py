"""
Factory function: instantiate the correct encoder from Hydra config.
"""
import logging

import torch
from omegaconf import DictConfig

from encoders.base import BaseEncoder

logger = logging.getLogger(__name__)


def make_encoder(cfg: DictConfig, device: torch.device) -> BaseEncoder:
    """
    Build and return an encoder instance from config.

    Args:
        cfg:    cfg.encoder sub-config
        device: torch device

    Returns:
        Concrete BaseEncoder subclass instance
    """
    name = cfg.encoder.name
    logger.info("Building encoder: %s", name)

    if name == "r3m":
        from encoders.r3m import R3MEncoder
        return R3MEncoder(
            model_name=cfg.encoder.model_name,
            camera_key=cfg.encoder.camera_key,
            device=device,
        )

    if name == "vae":
        from encoders.vae import VAEEncoder
        return VAEEncoder(
            latent_dim=cfg.encoder.representation_size,
            camera_key=cfg.encoder.camera_key,
            device=device,
            vae_path=cfg.encoder.vae_path,
            vae_imgsize=cfg.encoder.imsize,
        )

    raise ValueError(f"Unknown encoder '{name}'. Choices: r3m, vae.")
  