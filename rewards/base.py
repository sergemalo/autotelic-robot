"""
Base reward interface.
All reward functions receive the current obs, next obs, and goal obs,
plus the encoded latents, and return a scalar reward per transition.
"""
from abc import ABC, abstractmethod
from typing import Dict

import numpy as np
import torch


class BaseReward(ABC):
    """
    Abstract reward function.

    compute() is called once per environment step with both the
    raw obs dicts and the pre-computed latent vectors so that
    each reward type can use whichever it needs.
    """

    @abstractmethod
    def compute(
        self,
        obs: Dict[str, np.ndarray],
        next_obs: Dict[str, np.ndarray],
        goal_obs: Dict[str, np.ndarray],
        z: torch.Tensor,
        z_next: torch.Tensor,
        z_goal: torch.Tensor,
    ) -> float:
        """
        Compute a scalar reward for a single transition.

        Args:
            obs:      current observation dict
            next_obs: next observation dict
            goal_obs: goal observation dict
            z:        latent encoding of obs        (1, latent_dim)
            z_next:   latent encoding of next_obs   (1, latent_dim)
            z_goal:   latent encoding of goal_obs   (1, latent_dim)

        Returns:
            scalar float reward
        """
        ...

    @abstractmethod
    def reset(self) -> None:
        """
        Call at the start of each episode so rest_z re-initialises
        correctly for the new scene / object placement.
        """
        ...