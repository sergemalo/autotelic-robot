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

    if name == "privileged_obj":
        from rewards.privileged_obj import PrivilegedRewardObj
        return PrivilegedRewardObj(
            object_pos_key=cfg.reward.object_pos_key,
            reward_scale=cfg.reward.reward_scale,
            reward_offset=cfg.reward.reward_offset,
            reward_type=cfg.reward.reward_type,
            sparse_threshold=cfg.reward.sparse_threshold,
        )
    
    if name == "privileged_arm":
        from rewards.privileged_arm import PrivilegedRewardArm
        return PrivilegedRewardArm(
            eef_pos_key=cfg.reward.eef_pos_key,
            eef_ori_key=cfg.reward.eef_ori_key,
            goal_eef_pos_key=cfg.reward.goal_eef_pos_key,
            goal_eef_ori_key=cfg.reward.goal_eef_ori_key,
            reward_scale=cfg.reward.reward_scale,
            reward_type=cfg.reward.reward_type,
            sparse_threshold=cfg.reward.sparse_threshold,
            pos_offset=cfg.reward.pos_offset,
            ori_offset=cfg.reward.ori_offset,
        )
    
    if name == "latent":
        from rewards.latent import LatentReward
        return LatentReward(
            reward_scale=cfg.reward.reward_scale,
            distance_metric=cfg.reward.distance_metric,
        )

    raise ValueError(f"Unknown reward '{name}'. Choices: privileged, latent.")
