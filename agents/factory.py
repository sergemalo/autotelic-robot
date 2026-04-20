"""
Factory function: instantiate the SAC agent from Hydra config.
"""
import logging

import torch
from omegaconf import DictConfig

from agents.networks import PrivilegedStateEncoder
from agents.sac import SACAgent

logger = logging.getLogger(__name__)


def make_agent(cfg: DictConfig, device: torch.device) -> SACAgent:
    """
    Build and return a SACAgent from config.

    Args:
        cfg:    full Hydra config (cfg.agent and cfg.env sub-configs are used)
        device: torch device

    Returns:
        SACAgent instance
    """
    obs_dim = PrivilegedStateEncoder.OBS_DIM  # 34, fixed by the encoder layout
    logger.info("Building SAC agent: obs_dim=%d", obs_dim)

    return SACAgent(
        object_pos_key=cfg.reward.object_pos_key,
        obs_dim=obs_dim,
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
        normaliser_clip=cfg.agent.get("normaliser_clip", 10.0),
        pretrained_bc_path=cfg.agent.get("pretrained_bc_path", None),
        obj_to_eef_key=cfg.agent.get("obj_to_eef_key", None),
    )