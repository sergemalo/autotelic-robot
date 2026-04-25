"""
Privileged reward: uses ground-truth object position from the obs dict.

The object position for the task's target object is directly available
as a key in the LIBERO obs dict (e.g. "akita_black_bowl_1_pos").
The goal position is taken from the goal_obs dict under the same key.
"""
import logging
from typing import Dict

import numpy as np
import torch

from rewards.base import BaseReward

logger = logging.getLogger(__name__)


class PrivilegedRewardObj(BaseReward):
    """
    r = -||pos_next - pos_goal||₂  (negative Euclidean distance)

    Optionally can be converted to a sparse reward by thresholding.
    """

    def __init__(
        self,
        object_pos_key: str,
        reward_scale: float = 1.0,
        reward_offset: float = 0.0,
        reward_type: str = "negative_distance",
        sparse_threshold: float = 0.05,
    ):
        if object_pos_key is None:
            raise ValueError(
                "reward.object_pos_key must be set in config when using "
                "the privileged reward object. "
                "Example: reward.object_pos_key=akita_black_bowl_1_pos"
            )
        self.object_pos_key = object_pos_key
        self.reward_scale = reward_scale
        self.reward_offset = reward_offset
        self.reward_type = reward_type
        self.sparse_threshold = sparse_threshold

        logger.info(
            "PrivilegedRewardObj: key=%s, type=%s, scale=%.2f, offset=%.2f",
            object_pos_key, reward_type, reward_scale, reward_offset,
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
        pos_next = next_obs[self.object_pos_key]   # (3,)
        pos_goal = goal_obs[self.object_pos_key]   # (3,)

        dist = float(np.linalg.norm(pos_next - pos_goal))

        if self.reward_type == "negative_distance":
            reward = -dist + self.reward_offset
        elif self.reward_type == "sparse":
            reward = 1.0 if dist < self.sparse_threshold else 0.0
        else:
            raise ValueError(f"Unknown reward_type: {self.reward_type}")

        return self.reward_scale * reward
