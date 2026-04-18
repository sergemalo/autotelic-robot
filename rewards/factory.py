"""
Factory function: instantiate the correct reward from Hydra config.
"""
import logging

from omegaconf import DictConfig

from rewards.base import BaseReward

logger = logging.getLogger(__name__)


def make_reward(cfg: DictConfig) -> BaseReward:
    """
    Build and return a reward instance from config.

    Args:
        cfg: full Hydra config (cfg.reward sub-config is used)

    Returns:
        Concrete BaseReward subclass instance
    """
    name = cfg.reward.name
    logger.info("Building reward: %s", name)

    if name == "privileged":
        from rewards.privileged import PrivilegedReward
        return PrivilegedReward(
            object_pos_key=cfg.reward.object_pos_key,
            reward_scale=cfg.reward.reward_scale,
            reward_type=cfg.reward.reward_type,
            sparse_threshold=cfg.reward.sparse_threshold,
        )

    if name == "latent":
        from rewards.latent import LatentReward
        return LatentReward(
            reward_scale=cfg.reward.reward_scale,
            distance_metric=cfg.reward.distance_metric,
        )

    raise ValueError(f"Unknown reward '{name}'. Choices: privileged, latent.")
