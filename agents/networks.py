"""
SAC actor and critic networks.
Both operate entirely in latent space: input is [z, z_goal].
"""
import logging
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

logger = logging.getLogger(__name__)

LOG_STD_MIN = -5
LOG_STD_MAX = 2


def build_mlp(
    input_dim: int,
    output_dim: int,
    hidden_dims: List[int],
    activation: str = "relu",
) -> nn.Sequential:
    """Build a fully connected MLP with LayerNorm after each hidden layer."""
    act_fn = {"relu": nn.ReLU, "tanh": nn.Tanh, "elu": nn.ELU}[activation]
    layers = []
    in_dim = input_dim
    for h in hidden_dims:
        layers += [nn.Linear(in_dim, h), nn.LayerNorm(h), act_fn()]
        in_dim = h
    layers.append(nn.Linear(in_dim, output_dim))
    return nn.Sequential(*layers)


class SACActorNetwork(nn.Module):
    """
    Gaussian policy: (z, z_goal) → (mu, log_std) → action in [-1, 1]^action_dim
    """

    def __init__(
        self,
        latent_dim: int,
        action_dim: int,
        hidden_dims: List[int],
        activation: str = "relu",
    ):
        super().__init__()
        input_dim = latent_dim * 2  # [z, z_goal]
        self.net = build_mlp(input_dim, hidden_dims[-1], hidden_dims[:-1], activation)
        self.mu_head = nn.Linear(hidden_dims[-1], action_dim)
        self.log_std_head = nn.Linear(hidden_dims[-1], action_dim)

        logger.debug(
            "SACActorNetwork: input_dim=%d, action_dim=%d, hidden=%s",
            input_dim, action_dim, hidden_dims,
        )

    def forward(
        self, z: torch.Tensor, z_goal: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (mu, log_std)."""
        x = torch.cat([z, z_goal], dim=-1)
        h = self.net(x)
        mu = self.mu_head(h)
        log_std = self.log_std_head(h).clamp(LOG_STD_MIN, LOG_STD_MAX)
        return mu, log_std

    def sample(
        self, z: torch.Tensor, z_goal: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample action using reparameterisation trick.

        Returns:
            action:   tanh-squashed sample, shape (B, action_dim)
            log_prob: log probability of the action, shape (B, 1)
        """
        mu, log_std = self.forward(z, z_goal)
        std = log_std.exp()
        dist = Normal(mu, std)
        x = dist.rsample()
        action = torch.tanh(x)

        # Log prob with tanh correction
        log_prob = dist.log_prob(x) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)

        return action, log_prob

    def deterministic_action(
        self, z: torch.Tensor, z_goal: torch.Tensor
    ) -> torch.Tensor:
        """Return tanh(mu) — used for evaluation."""
        mu, _ = self.forward(z, z_goal)
        return torch.tanh(mu)


class SACCriticNetwork(nn.Module):
    """
    Twin Q-networks: (z, a, z_goal) → (Q1, Q2)
    Using two critics is the standard SAC / TD3 trick to reduce overestimation.
    """

    def __init__(
        self,
        latent_dim: int,
        action_dim: int,
        hidden_dims: List[int],
        activation: str = "relu",
    ):
        super().__init__()
        input_dim = latent_dim * 2 + action_dim  # [z, z_goal, a]

        self.q1 = build_mlp(input_dim, 1, hidden_dims, activation)
        self.q2 = build_mlp(input_dim, 1, hidden_dims, activation)

        logger.debug(
            "SACCriticNetwork: input_dim=%d, hidden=%s",
            input_dim, hidden_dims,
        )

    def forward(
        self,
        z: torch.Tensor,
        action: torch.Tensor,
        z_goal: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (Q1, Q2)."""
        x = torch.cat([z, z_goal, action], dim=-1)
        return self.q1(x), self.q2(x)
