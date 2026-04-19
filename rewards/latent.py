"""
Latent distance reward: r = -||z_next - z_goal||

This is the original RIG reward, operating in the encoder's latent space.
"""
import logging
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F

from rewards.base import BaseReward

logger = logging.getLogger(__name__)


class LatentReward(BaseReward):
    """
    r = -||z_next - z_goal||₂   (euclidean)
    or
    r = cosine_similarity(z_next, z_goal) - 1   (cosine, in [-2, 0])
    """

    def __init__(
        self,
        reward_scale: float = 1.0,
        distance_metric: str = "euclidean",
    ):
        self.reward_scale = reward_scale
        self.distance_metric = distance_metric
        logger.info(
            "LatentReward: metric=%s, scale=%.2f",
            distance_metric, reward_scale,
        )

    def compute(
        self,
        obs: Dict[str, np.ndarray],
        next_obs: Dict[str, np.ndarray],
        goal_obs: Dict[str, np.ndarray],
        z: torch.Tensor,
        z_next: torch.Tensor,
        z_goal: torch.Tensor,
    ) -> float:
        if self.distance_metric == "euclidean":
            dist = torch.norm(z_next - z_goal, dim=-1)
            reward = -dist.item()
        elif self.distance_metric == "cosine":
            sim = F.cosine_similarity(z_next, z_goal, dim=-1)
            reward = (sim - 1.0).item()
        else:
            raise ValueError(f"Unknown distance_metric: {self.distance_metric}")

        return self.reward_scale * reward

    def reset(self) -> None:
        """
        Call at the start of each episode so rest_z re-initialises
        correctly for the new scene / object placement.
        """
        pass