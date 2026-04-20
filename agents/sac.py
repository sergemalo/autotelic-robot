"""
Soft Actor-Critic (SAC) agent — privileged state mode.

update() and select_action() accept raw LIBERO obs dicts directly.
Internally the agent:
  1. Encodes obs dicts → 34-float vectors  (via PrivilegedStateEncoder)
  2. Updates RunningMeanStd stats          (on every update() batch)
  3. Normalises the vectors               (zero mean / unit std, clipped)
  4. Runs actor / critic forward passes

This means the caller (trainer) never needs to touch encoding or
normalisation — it just passes obs dicts straight through.
"""
import logging
import os
from typing import Dict, Optional, Tuple, List

import numpy as np
import torch
import torch.nn.functional as F

from agents.networks import (
    PrivilegedStateEncoder,
    RunningMeanStd,
    SACActorNetwork,
    SACCriticNetwork,
)

logger = logging.getLogger(__name__)


class SACAgent:
    """
    SAC with automatic entropy tuning, privileged state observations,
    online normalisation, and raw-obs-dict interface.

    Args:
        object_pos_key: obs-dict key for the target object position,
                        e.g. "chocolate_pudding_1_pos".
                        Passed to PrivilegedStateEncoder.
        obs_dim:        must equal PrivilegedStateEncoder.OBS_DIM (34).
                        Kept explicit so the config documents what goes in.
        action_dim:     dimension of the action space.
        hidden_dims:    MLP hidden layer sizes, e.g. [512, 512, 512].
        activation:     "relu" | "elu" | "tanh".
        normaliser_clip: clip value for RunningMeanStd (default 10.0).
        ... (standard SAC hyperparameters)
    """

    def __init__(
        self,
        object_pos_key: str,
        obs_dim: int,
        action_dim: int,
        hidden_dims: List[int],
        activation: str,
        actor_lr: float,
        critic_lr: float,
        alpha_lr: float,
        gamma: float,
        tau: float,
        init_temperature: float,
        target_entropy,            # float or "auto"
        device: torch.device,
        normaliser_clip: float = 10.0,
        pretrained_bc_path: Optional[str] = None,
        obj_to_eef_key: Optional[str] = None,
    ):
        self.gamma  = gamma
        self.tau    = tau
        self.device = device

        # ── Encoder (stateless, pure numpy) ───────────────────────────
        self.encoder = PrivilegedStateEncoder(
            object_pos_key=object_pos_key,
            obj_to_eef_key=obj_to_eef_key,
        )
        assert obs_dim == PrivilegedStateEncoder.OBS_DIM, (
            f"obs_dim={obs_dim} does not match "
            f"PrivilegedStateEncoder.OBS_DIM={PrivilegedStateEncoder.OBS_DIM}"
        )

        # ── Online normaliser (stateful, owned by the agent) ──────────
        self.normaliser = RunningMeanStd(dim=obs_dim, clip=normaliser_clip)

        # ── Networks ──────────────────────────────────────────────────
        self.actor = SACActorNetwork(
            input_dim=obs_dim,
            action_dim=action_dim,
            hidden_dims=hidden_dims,
            activation=activation,
        ).to(device)

        self.critic = SACCriticNetwork(
            input_dim=obs_dim,
            action_dim=action_dim,
            hidden_dims=hidden_dims,
            activation=activation,
        ).to(device)

        self.critic_target = SACCriticNetwork(
            input_dim=obs_dim,
            action_dim=action_dim,
            hidden_dims=hidden_dims,
            activation=activation,
        ).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        for p in self.critic_target.parameters():
            p.requires_grad = False

        # ── Entropy temperature ───────────────────────────────────────
        self.target_entropy = (
            -float(action_dim) if target_entropy == "auto" else float(target_entropy)
        )
        self.log_alpha = torch.tensor(
            [float(torch.log(torch.tensor(init_temperature)))],
            dtype=torch.float32, device=device, requires_grad=True,
        )

        # ── Optimisers ────────────────────────────────────────────────
        self.actor_optimizer  = torch.optim.Adam(self.actor.parameters(),  lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.alpha_optimizer  = torch.optim.Adam([self.log_alpha],         lr=alpha_lr)

        if pretrained_bc_path is not None:
            self._load_bc_weights(pretrained_bc_path)

        logger.info(
            "SACAgent: obs_dim=%d  action_dim=%d  target_entropy=%.2f",
            obs_dim, action_dim, self.target_entropy,
        )

    # ── Properties ────────────────────────────────────────────────────

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    # ── Internal helpers ──────────────────────────────────────────────

    def _encode_and_normalise(
        self,
        obs: Dict[str, np.ndarray],
        goal_obs: Dict[str, np.ndarray],
    ) -> torch.Tensor:
        """
        Encode a single obs/goal pair → normalise → (1, obs_dim) tensor.
        Used by select_action().
        """
        raw = self.encoder.encode(obs, goal_obs)          # (34,) float32
        normed = self.normaliser.normalise(raw)           # (34,) float32
        return torch.tensor(normed, dtype=torch.float32, device=self.device).unsqueeze(0)

    def _encode_batch_and_normalise(
        self,
        obs: Dict[str, np.ndarray],
        goal_obs: Dict[str, np.ndarray],
        update_stats: bool = False,
    ) -> torch.Tensor:
        """
        Encode a batched obs dict → optionally update normaliser stats
        → normalise → (B, obs_dim) tensor on self.device.

        Args:
            obs:          Dict[str, np.ndarray] with shape (B, *feature_shape) per key
            goal_obs:     same structure
            update_stats: if True, update RunningMeanStd with this batch
                          (True only for current obs, not next_obs, to avoid
                          counting each transition twice)
        """
        raw = self.encoder.encode_batch(obs, goal_obs)  # (B, 34) float32

        if update_stats:
            self.normaliser.update(raw)

        normed = self.normaliser.normalise(raw)  # (B, 34) float32
        return torch.tensor(normed, dtype=torch.float32, device=self.device)

    # ── Action selection ──────────────────────────────────────────────

    @torch.no_grad()
    def select_action(
        self,
        obs: Dict[str, np.ndarray],
        goal_obs: Dict[str, np.ndarray],
        deterministic: bool = False,
    ) -> np.ndarray:
        """
        Select action given raw obs and goal_obs dicts.

        Args:
            obs:           current observation dict from env.step() / env.reset()
            goal_obs:      goal observation dict
            deterministic: if True, return tanh(mu) without noise (for eval)

        Returns:
            action: (action_dim,) float32 array in [-1, 1]
        """
        state = self._encode_and_normalise(obs, goal_obs)  # (1, 34)

        if deterministic:
            action = self.actor.deterministic_action(state)
        else:
            action, _ = self.actor.sample(state)

        return action.squeeze(0).cpu().numpy()

    # ── Training step ─────────────────────────────────────────────────

    def update(
        self,
        obs:       Dict[str, np.ndarray],   # (B, *feature_shape) per key
        actions:   np.ndarray,              # (B, action_dim)
        next_obs:  Dict[str, np.ndarray],   # (B, *feature_shape) per key
        rewards:   np.ndarray,              # (B,) or (B, 1)
        dones:     np.ndarray,              # (B,) or (B, 1)
        goal_obs:  Dict[str, np.ndarray],   # (B, *feature_shape) per key
    ) -> Dict[str, float]:
        """
        One gradient update step.

        Args:
            obs:       list of B current observation dicts
            actions:   (B, action_dim) numpy array
            next_obs:  list of B next observation dicts
            rewards:   (B,) or (B, 1) numpy array
            dones:     (B,) or (B, 1) numpy array
            goal_obs:  list of B goal observation dicts
                       (same goal for current and next obs — standard GC-RL)

        Returns:
            dict of scalar training metrics for logging.
        """
        # ── Encode + normalise ────────────────────────────────────────
        # Update running stats only on the current obs batch.
        # next_obs uses the same goal so shares the same goal distribution;
        # updating stats on both would double-count each transition.
        states      = self._encode_batch_and_normalise(obs,      goal_obs, update_stats=True)
        next_states = self._encode_batch_and_normalise(next_obs, goal_obs, update_stats=False)

        # ── Convert remaining inputs to tensors ───────────────────────
        actions_t = actions
        rewards_t = rewards
        dones_t = torch.tensor(
            np.asarray(dones, dtype=np.float32).reshape(-1, 1),
            dtype=torch.float32, device=self.device,
        )

        metrics = {}

        # ── Critic update ─────────────────────────────────────────────
        critic_loss, metrics_c = self._critic_loss(
            states, actions_t, next_states, rewards_t, dones_t
        )
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        metrics.update(metrics_c)

        # ── Actor + alpha update ──────────────────────────────────────
        actor_loss, alpha_loss, metrics_a = self._actor_alpha_loss(states)
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        metrics.update(metrics_a)

        # ── Soft target update ────────────────────────────────────────
        self._soft_update_target()

        # Expose normaliser sample count for monitoring
        metrics["normaliser_count"] = float(self.normaliser.count)

        return metrics

    # ── Loss functions ────────────────────────────────────────────────

    def _critic_loss(
        self,
        states:      torch.Tensor,
        actions:     torch.Tensor,
        next_states: torch.Tensor,
        rewards:     torch.Tensor,
        dones:       torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict]:
        with torch.no_grad():
            next_actions, next_log_pi = self.actor.sample(next_states)
            q1_t, q2_t = self.critic_target(next_states, next_actions)
            q_target = torch.min(q1_t, q2_t)
            backup = rewards + self.gamma * (1.0 - dones) * (
                q_target - self.alpha.detach() * next_log_pi
            )

        q1, q2 = self.critic(states, actions)
        loss = F.mse_loss(q1, backup) + F.mse_loss(q2, backup)

        return loss, {
            "critic_loss": loss.item(),
            "q1_mean":     q1.mean().item(),
            "q2_mean":     q2.mean().item(),
        }

    def _actor_alpha_loss(
        self,
        states: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        actions, log_pi = self.actor.sample(states)
        q1, q2 = self.critic(states, actions)
        q_min  = torch.min(q1, q2)

        actor_loss = (self.alpha.detach() * log_pi - q_min).mean()
        alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()

        return actor_loss, alpha_loss, {
            "actor_loss": actor_loss.item(),
            "alpha_loss": alpha_loss.item(),
            "alpha":      self.alpha.item(),
            "entropy":    -log_pi.mean().item(),
        }

    def _soft_update_target(self):
        for p, tp in zip(self.critic.parameters(), self.critic_target.parameters()):
            tp.data.copy_(self.tau * p.data + (1.0 - self.tau) * tp.data)

    # ── Checkpointing ─────────────────────────────────────────────────

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(
            {
                "actor":            self.actor.state_dict(),
                "critic":           self.critic.state_dict(),
                "critic_target":    self.critic_target.state_dict(),
                "log_alpha":        self.log_alpha.data,
                "actor_optimizer":  self.actor_optimizer.state_dict(),
                "critic_optimizer": self.critic_optimizer.state_dict(),
                "alpha_optimizer":  self.alpha_optimizer.state_dict(),
                "normaliser":       self.normaliser.state_dict(),  # ← include stats
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
        if "normaliser" in ckpt:
            self.normaliser.load_state_dict(ckpt["normaliser"])
        else:
            logger.warning("Checkpoint has no normaliser state — stats reset to zero.")
        logger.info("Checkpoint loaded: %s", path)

    def _load_bc_weights(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        state_dict = ckpt.get("actor", ckpt)
        self.actor.load_state_dict(state_dict)
        logger.info("BC pretrained actor weights loaded from: %s", path)
