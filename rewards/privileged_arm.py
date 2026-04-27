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


class PrivilegedRewardArm(BaseReward):

    def __init__(
        self,
        eef_pos_key: str,
        eef_ori_key: str,
        goal_eef_pos_key: str,
        goal_eef_ori_key: str,
        reward_scale: float = 1.0,
        reward_type: str = "negative_distance",
        sparse_threshold: float = 0.05,
        pos_offset: float = 1.0,
        ori_offset: float = 1.0,
    ):
        if eef_pos_key is None or eef_ori_key is None or goal_eef_pos_key is None or goal_eef_ori_key is None:
            raise ValueError(
                "reward.eef_pos_key, reward.eef_ori_key, reward.goal_eef_pos_key, and reward.goal_eef_ori_key must be set in config when using "
                "the privileged reward arm. "
                "Example: reward.eef_pos_key=robot0_eef_pos"
            )
        self.eef_pos_key = eef_pos_key
        self.eef_ori_key = eef_ori_key
        self.goal_eef_pos_key = goal_eef_pos_key
        self.goal_eef_ori_key = goal_eef_ori_key
        self.reward_scale = reward_scale
        self.reward_type = reward_type
        self.sparse_threshold = sparse_threshold
        self.pos_offset = pos_offset
        self.ori_offset = ori_offset

        logger.info(
            "PrivilegedRewardArm: eef_pos_key=%s, eef_ori_key=%s, goal_eef_pos_key=%s, goal_eef_ori_key=%s, type=%s, scale=%.2f, pos_offset=%.2f, ori_offset=%.2f",
            eef_pos_key, eef_ori_key, goal_eef_pos_key, goal_eef_ori_key, reward_type, reward_scale, pos_offset, ori_offset,
        )

    def quaternion_geodesic_distance(self, q1: np.ndarray, q2: np.ndarray) -> float:
        """
        Geodesic distance between two unit quaternions.
        Returns a value in [0, 1], where 0 = identical, 1 = maximally opposed.
        Works regardless of quaternion convention (xyzw or wxyz) since it only
        uses the dot product.
        """
        q1 = q1 / (np.linalg.norm(q1) + 1e-8)
        q2 = q2 / (np.linalg.norm(q2) + 1e-8)
        # Clamp to [-1, 1] to guard against floating point error in arccos
        dot = np.clip(np.abs(np.dot(q1, q2)), 0.0, 1.0)
        # arccos maps [0,1] -> [pi/2, 0]; normalize to [0, 1]
        return (2.0 / np.pi) * np.arccos(dot)


    def compute(
        self,
        obs: Dict[str, np.ndarray],
        next_obs: Dict[str, np.ndarray],
        goal_obs: Dict[str, np.ndarray],
        z: torch.Tensor,
        z_next: torch.Tensor,
        z_goal: torch.Tensor,
        ) -> float:
        eef_pos  = obs[self.eef_pos_key]
        eef_ori  = obs[self.eef_ori_key]
        goal_pos = goal_obs[self.goal_eef_pos_key]
        goal_ori = goal_obs[self.goal_eef_ori_key]

        pos_dist = np.linalg.norm(eef_pos - goal_pos)
        ori_dist = self.quaternion_geodesic_distance(eef_ori, goal_ori)

        # Both terms are >= 0 before offset, offset guarantees floor at 0
        pos_reward = self.pos_offset - pos_dist   # in [-inf, pos_offset], floored by offset choice
        ori_reward = self.ori_offset - ori_dist   # in [ori_offset - 1, ori_offset]

        reward = 0.5 * (pos_reward + ori_reward) * self.reward_scale
        return reward

    def compute_batch(
        self,
        next_obs: Dict[str, np.ndarray],
        goal_obs: Dict[str, np.ndarray],
        device: torch.device,
    ) -> torch.Tensor:
        """Compute privileged arm rewards for a relabeled batch as (B, 1)."""
        eef_pos = torch.as_tensor(next_obs[self.eef_pos_key], dtype=torch.float32, device=device)
        eef_ori = torch.as_tensor(next_obs[self.eef_ori_key], dtype=torch.float32, device=device)
        goal_pos = torch.as_tensor(goal_obs[self.goal_eef_pos_key], dtype=torch.float32, device=device)
        goal_ori = torch.as_tensor(goal_obs[self.goal_eef_ori_key], dtype=torch.float32, device=device)

        pos_dist = torch.linalg.vector_norm(eef_pos - goal_pos, dim=-1, keepdim=True)

        # Quaternion geodesic distance in [0, 1]
        eef_ori = eef_ori / (torch.linalg.vector_norm(eef_ori, dim=-1, keepdim=True) + 1e-8)
        goal_ori = goal_ori / (torch.linalg.vector_norm(goal_ori, dim=-1, keepdim=True) + 1e-8)
        dot = torch.sum(eef_ori * goal_ori, dim=-1, keepdim=True).abs().clamp(0.0, 1.0)
        ori_dist = (2.0 / np.pi) * torch.arccos(dot)

        pos_reward = self.pos_offset - pos_dist
        ori_reward = self.ori_offset - ori_dist
        reward = 0.5 * (pos_reward + ori_reward) * self.reward_scale
        return reward