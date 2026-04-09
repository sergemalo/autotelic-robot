"""
Soft Actor-Critic (SAC) agent.
Operates entirely in encoder latent space.
Supports loading a BC-pretrained actor at initialisation.
"""
import logging
import os
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from omegaconf import DictConfig

from agents.networks import SACActorNetwork, SACCriticNetwork

logger = logging.getLogger(__name__)


class SACAgent:
    """
    SAC with automatic entropy tuning.

    Actor input:  [z, z_goal]      where z, z_goal ∈ R^latent_dim
    Critic input: [z, z_goal, a]
    """

    def __init__(
        self,
        latent_dim: int,
        action_dim: int,
        hidden_dims: List[int],
        activation: str,
        actor_lr: float,
        critic_lr: float,
        alpha_lr: float,
        gamma: float,
        tau: float,
        init_temperature: float,
        target_entropy,          # float or "auto"
        device: torch.device,
        pretrained_bc_path: Optional[str] = None,
    ):
        self.gamma = gamma
        self.tau = tau
        self.device = device

        # ---- Networks ------------------------------------------------
        self.actor = SACActorNetwork(
            latent_dim, action_dim, hidden_dims, activation
        ).to(device)

        self.critic = SACCriticNetwork(
            latent_dim, action_dim, hidden_dims, activation
        ).to(device)

        self.critic_target = SACCriticNetwork(
            latent_dim, action_dim, hidden_dims, activation
        ).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        for p in self.critic_target.parameters():
            p.requires_grad = False

        # ---- Entropy temperature ------------------------------------
        if target_entropy == "auto":
            self.target_entropy = -float(action_dim)
        else:
            self.target_entropy = float(target_entropy)

        self.log_alpha = torch.tensor(
            [float(torch.log(torch.tensor(init_temperature)))],
            dtype=torch.float32,
            device=device,
            requires_grad=True,
        )

        # ---- Optimisers ---------------------------------------------
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=alpha_lr)

        # ---- BC pretraining -----------------------------------------
        if pretrained_bc_path is not None:
            self._load_bc_weights(pretrained_bc_path)

        logger.info(
            "SACAgent: latent_dim=%d, action_dim=%d, target_entropy=%.2f",
            latent_dim, action_dim, self.target_entropy,
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    @torch.no_grad()
    def select_action(
        self,
        z: torch.Tensor,
        z_goal: torch.Tensor,
        deterministic: bool = False,
    ) -> torch.Tensor:
        """
        Select action given latent state and goal.

        Args:
            z:            (1, latent_dim)
            z_goal:       (1, latent_dim)
            deterministic: if True, return tanh(mu) without noise

        Returns:
            action: (action_dim,) numpy array in [-1, 1]
        """
        if deterministic:
            action = self.actor.deterministic_action(z, z_goal)
        else:
            action, _ = self.actor.sample(z, z_goal)
        return action.squeeze(0).cpu().numpy()

    # ------------------------------------------------------------------
    # Training step
    # ------------------------------------------------------------------

    def update(
        self,
        z: torch.Tensor,
        actions: torch.Tensor,
        z_next: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        z_goal: torch.Tensor,
    ) -> Dict[str, float]:
        """
        One gradient update step for critic, actor, and alpha.

        All tensor inputs are (B, dim) on self.device.

        Returns:
            dict of scalar metrics for logging
        """
        metrics = {}

        # ---- Critic update ------------------------------------------
        critic_loss, metrics_c = self._critic_loss(
            z, actions, z_next, rewards, dones, z_goal
        )
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        metrics.update(metrics_c)

        # ---- Actor + alpha update -----------------------------------
        actor_loss, alpha_loss, metrics_a = self._actor_alpha_loss(z, z_goal)
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        metrics.update(metrics_a)

        # ---- Soft target update -------------------------------------
        self._soft_update_target()

        return metrics

    def _critic_loss(
        self,
        z: torch.Tensor,
        actions: torch.Tensor,
        z_next: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        z_goal: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict]:
        with torch.no_grad():
            next_actions, next_log_pi = self.actor.sample(z_next, z_goal)
            q1_target, q2_target = self.critic_target(z_next, next_actions, z_goal)
            q_target = torch.min(q1_target, q2_target)
            backup = rewards + self.gamma * (1.0 - dones) * (
                q_target - self.alpha.detach() * next_log_pi
            )

        q1, q2 = self.critic(z, actions, z_goal)
        loss = F.mse_loss(q1, backup) + F.mse_loss(q2, backup)

        return loss, {
            "critic_loss": loss.item(),
            "q1_mean": q1.mean().item(),
            "q2_mean": q2.mean().item(),
        }

    def _actor_alpha_loss(
        self,
        z: torch.Tensor,
        z_goal: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        actions, log_pi = self.actor.sample(z, z_goal)
        q1, q2 = self.critic(z, actions, z_goal)
        q_min = torch.min(q1, q2)

        actor_loss = (self.alpha.detach() * log_pi - q_min).mean()
        alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()

        return actor_loss, alpha_loss, {
            "actor_loss": actor_loss.item(),
            "alpha_loss": alpha_loss.item(),
            "alpha": self.alpha.item(),
            "entropy": -log_pi.mean().item(),
        }

    def _soft_update_target(self):
        for param, target_param in zip(
            self.critic.parameters(), self.critic_target.parameters()
        ):
            target_param.data.copy_(
                self.tau * param.data + (1.0 - self.tau) * target_param.data
            )

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
                "critic_target": self.critic_target.state_dict(),
                "log_alpha": self.log_alpha.data,
                "actor_optimizer": self.actor_optimizer.state_dict(),
                "critic_optimizer": self.critic_optimizer.state_dict(),
                "alpha_optimizer": self.alpha_optimizer.state_dict(),
            },
            path,
        )
        logger.info("Checkpoint saved: %s", path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
        self.critic_target.load_state_dict(ckpt["critic_target"])
        self.log_alpha.data.copy_(ckpt["log_alpha"])
        self.actor_optimizer.load_state_dict(ckpt["actor_optimizer"])
        self.critic_optimizer.load_state_dict(ckpt["critic_optimizer"])
        self.alpha_optimizer.load_state_dict(ckpt["alpha_optimizer"])
        logger.info("Checkpoint loaded: %s", path)

    def _load_bc_weights(self, path: str):
        """Load only the actor weights from a BC checkpoint."""
        ckpt = torch.load(path, map_location=self.device)
        # Support both a full SAC checkpoint and a bare actor state dict
        state_dict = ckpt.get("actor", ckpt)
        self.actor.load_state_dict(state_dict)
        logger.info("BC pretrained actor weights loaded from: %s", path)
