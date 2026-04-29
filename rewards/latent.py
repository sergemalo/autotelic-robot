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
        Distance-based latent reward with optional non-negative shaping.

        Base distance:
            - euclidean: ||z_next - z_goal||₂
            - cosine:    1 - cosine_similarity(z_next, z_goal)

        Reward:
            r = reward_scale * max(0, reward_offset - normalized_distance)  (if non_negative)
            r = reward_scale * (reward_offset - normalized_distance)         (otherwise)
    """

    def __init__(
        self,
        reward_scale: float = 1.0,
        distance_metric: str = "euclidean",
        reward_offset: float = 0.0,
        non_negative: bool = False,
        normalize_distance: bool = False,
        distance_norm: str = "sqrt_latent_dim",
    ):
        self.reward_scale = reward_scale
        self.distance_metric = distance_metric
        self.reward_offset = reward_offset
        self.non_negative = non_negative
        self.normalize_distance = normalize_distance
        self.distance_norm = distance_norm
        logger.info(
            (
                "LatentReward: metric=%s, scale=%.2f, offset=%.2f, "
                "non_negative=%s, normalize_distance=%s, distance_norm=%s"
            ),
            distance_metric,
            reward_scale,
            reward_offset,
            non_negative,
            normalize_distance,
            distance_norm,
        )

    def _compute_distance(self, z_next: torch.Tensor, z_goal: torch.Tensor) -> torch.Tensor:
        """Compute per-sample latent distance as shape (B, 1)."""
        if self.distance_metric == "euclidean":
            dist = torch.norm(z_next - z_goal, dim=-1, keepdim=True)
            if self.normalize_distance:
                if self.distance_norm == "sqrt_latent_dim":
                    denom = max(2 * float(z_next.shape[-1]) ** 0.5, 1e-8)
                    dist = dist / denom
                else:
                    raise ValueError(f"Unknown distance_norm: {self.distance_norm}")
            return dist

        if self.distance_metric == "cosine":
            sim = F.cosine_similarity(z_next, z_goal, dim=-1).unsqueeze(-1)
            # Cosine distance in [0, 2]
            dist = 1.0 - sim
            if self.normalize_distance:
                # Normalized cosine distance in [0, 1]
                dist = 0.5 * dist
            return dist

        raise ValueError(f"Unknown distance_metric: {self.distance_metric}")

    def compute_batch(self, z_next: torch.Tensor, z_goal: torch.Tensor) -> torch.Tensor:
        """Compute reward for a batch of latent pairs as shape (B, 1)."""
        dist = self._compute_distance(z_next, z_goal)
        reward = self.reward_offset - dist
        if self.non_negative:
            reward = torch.clamp_min(reward, 0.0)
        return self.reward_scale * reward

    def compute(
        self,
        obs: Dict[str, np.ndarray],
        next_obs: Dict[str, np.ndarray],
        goal_obs: Dict[str, np.ndarray],
        z: torch.Tensor,
        z_next: torch.Tensor,
        z_goal: torch.Tensor,
    ) -> float:
        reward = self.compute_batch(z_next, z_goal)
        return reward.squeeze(-1).item()
