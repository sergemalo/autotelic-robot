"""
Factory function: instantiate the SAC agent from Hydra config.
"""
import logging

import torch
from omegaconf import DictConfig

from agents.sac import SACAgent

logger = logging.getLogger(__name__)


def make_agent(cfg: DictConfig, latent_dim: int, device: torch.device) -> SACAgent:
    """
    Build and return a SACAgent from config.

    Args:
        cfg:        full Hydra config (cfg.agent sub-config is used)
        latent_dim: latent dim from the encoder (injected at runtime)
        device:     torch device

    Returns:
        SACAgent instance
    """
    logger.info("Building SAC agent: latent_dim=%d", latent_dim)
    return SACAgent(
        latent_dim=latent_dim,
        action_dim=cfg.env.action_dim,
        hidden_dims=list(cfg.agent.hidden_dims),
        activation=cfg.agent.activation,
        actor_lr=cfg.agent.actor_lr,
        critic_lr=cfg.agent.critic_lr,
        alpha_lr=cfg.agent.alpha_lr,
        gamma=cfg.agent.gamma,
        tau=cfg.agent.tau,
        init_temperature=cfg.agent.init_temperature,
        target_entropy=cfg.agent.target_entropy,
        device=device,
        pretrained_bc_path=cfg.agent.pretrained_bc_path,
    )
