"""
SAC actor and critic networks.

Two modes are supported:
  - Latent mode  (original): input is [z, z_goal]  where z ∈ R^latent_dim
  - Privileged mode (new):   input is a 34-float ground-truth state vector
                              built by PrivilegedStateEncoder

The same build_mlp / SACActorNetwork / SACCriticNetwork classes are used in
both modes — the only difference is the input_dim passed at construction.
"""
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal

logger = logging.getLogger(__name__)

LOG_STD_MIN = -5
LOG_STD_MAX = 2


# ──────────────────────────────────────────────────────────────────────────────
# Running normalizer
# ──────────────────────────────────────────────────────────────────────────────

class RunningMeanStd:
    """
    Welford online algorithm for running mean and variance.

    Tracks per-dimension statistics over all samples seen so far.
    Used to normalise the privileged state vector to zero mean / unit std
    before it enters the actor and critic networks.

    Args:
        dim:  number of dimensions to track (e.g. 34)
        eps:  small constant added to std to avoid division by zero
        clip: clip normalised values to [-clip, clip] after normalisation
    """

    def __init__(self, dim: int, eps: float = 1e-8, clip: float = 10.0):
        self.dim   = dim
        self.eps   = eps
        self.clip  = clip
        self.count = 0
        self.mean  = np.zeros(dim, dtype=np.float64)
        self.M2    = np.zeros(dim, dtype=np.float64)  # sum of squared deviations

    def update(self, batch: np.ndarray) -> None:
        """
        Update running stats with a (B, dim) or (dim,) array.
        Uses Welford's online algorithm — numerically stable for large counts.
        """
        batch = np.asarray(batch, dtype=np.float64)
        if batch.ndim == 1:
            batch = batch[np.newaxis, :]
        for x in batch:
            self.count += 1
            delta       = x - self.mean
            self.mean  += delta / self.count
            self.M2    += delta * (x - self.mean)

    @property
    def var(self) -> np.ndarray:
        if self.count < 2:
            return np.ones(self.dim, dtype=np.float64)
        return self.M2 / (self.count - 1)

    @property
    def std(self) -> np.ndarray:
        return np.sqrt(self.var)

    def normalise(self, x: np.ndarray) -> np.ndarray:
        """Normalise a (dim,) or (B, dim) numpy array, then clip."""
        normed = (x - self.mean) / (self.std + self.eps)
        return np.clip(normed, -self.clip, self.clip).astype(np.float32)

    def normalise_tensor(self, x: torch.Tensor) -> torch.Tensor:
        """Same as normalise() but for a torch.Tensor on any device."""
        mean = torch.tensor(self.mean, dtype=x.dtype, device=x.device)
        std  = torch.tensor(self.std,  dtype=x.dtype, device=x.device)
        return (x - mean).div(std + self.eps).clamp(-self.clip, self.clip)

    # -- Serialisation --------------------------------------------------

    def state_dict(self) -> dict:
        return {
            "count": self.count,
            "mean":  self.mean.copy(),
            "M2":    self.M2.copy(),
            "eps":   self.eps,
            "clip":  self.clip,
        }

    def load_state_dict(self, d: dict) -> None:
        self.count = d["count"]
        self.mean  = d["mean"].copy()
        self.M2    = d["M2"].copy()
        self.eps   = d.get("eps",  self.eps)
        self.clip  = d.get("clip", self.clip)


# ──────────────────────────────────────────────────────────────────────────────
# MLP builder
# ──────────────────────────────────────────────────────────────────────────────

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


# ──────────────────────────────────────────────────────────────────────────────
# Privileged state encoder
# ──────────────────────────────────────────────────────────────────────────────

class PrivilegedStateEncoder:
    """
    Builds the 34-float privileged observation vector from a raw LIBERO obs dict.

    Vector layout:
        [0:7]   joint_pos          robot0_joint_pos
        [7:14]  joint_vel          robot0_joint_vel
        [14]    gripper_qpos       robot0_gripper_qpos[0]
        [15:18] eef_pos            robot0_eef_pos
        [18:22] eef_quat           robot0_eef_quat
        [22:25] obj_pos            <object_pos_key>
        [25:28] obj_minus_eef      <object>_to_robot0_eef_pos  (precomputed by LIBERO)
        [28:31] goal_pos           goal_obs[object_pos_key]
        [31:34] goal_minus_obj     goal_pos - obj_pos
        ──────
        total   34

    Normalisation is intentionally NOT applied here — SACAgent owns the
    RunningMeanStd and applies it after encoding, so this class is stateless.

    Args:
        object_pos_key: obs key for object position, e.g. "chocolate_pudding_1_pos"
        obj_to_eef_key: obs key for precomputed (obj - eef) vector.
                        LIBERO names this "<object>_to_robot0_eef_pos".
                        Auto-derived from object_pos_key if not provided.
    """

    OBS_DIM = 34

    def __init__(
        self,
        object_pos_key: str,
        obj_to_eef_key: Optional[str] = None,
    ):
        self.object_pos_key = object_pos_key

        if obj_to_eef_key is not None:
            self.obj_to_eef_key = obj_to_eef_key
        else:
            base = object_pos_key[:-4] if object_pos_key.endswith("_pos") else object_pos_key
            self.obj_to_eef_key = f"{base}_to_robot0_eef_pos"

        logger.info(
            "PrivilegedStateEncoder: object_pos_key=%s  obj_to_eef_key=%s  obs_dim=%d",
            object_pos_key, self.obj_to_eef_key, self.OBS_DIM,
        )

    def encode(
        self,
        obs: Dict[str, np.ndarray],
        goal_obs: Dict[str, np.ndarray],
    ) -> np.ndarray:
        """
        Build the raw (un-normalised) 34-float vector from one obs/goal pair.

        Returns:
            state: (34,) float32 array
        """
        joint_pos      = np.asarray(obs["robot0_joint_pos"],         dtype=np.float32)  # (7,)
        joint_vel      = np.asarray(obs["robot0_joint_vel"],         dtype=np.float32)  # (7,)
        gripper_qpos   = np.asarray([obs["robot0_gripper_qpos"][0]], dtype=np.float32)  # (1,)
        eef_pos        = np.asarray(obs["robot0_eef_pos"],           dtype=np.float32)  # (3,)
        eef_quat       = np.asarray(obs["robot0_eef_quat"],          dtype=np.float32)  # (4,)
        obj_pos        = np.asarray(obs[self.object_pos_key],        dtype=np.float32)  # (3,)
        obj_minus_eef  = np.asarray(obs[self.obj_to_eef_key],        dtype=np.float32)  # (3,)
        goal_pos       = np.asarray(goal_obs[self.object_pos_key],   dtype=np.float32)  # (3,)
        goal_minus_obj = (goal_pos - obj_pos).astype(np.float32)                        # (3,)

        state = np.concatenate([
            joint_pos,       # 7
            joint_vel,       # 7
            gripper_qpos,    # 1
            eef_pos,         # 3
            eef_quat,        # 4
            obj_pos,         # 3
            obj_minus_eef,   # 3
            goal_pos,        # 3
            goal_minus_obj,  # 3
        ])  # total: 34

        assert state.shape == (self.OBS_DIM,), \
            f"Expected ({self.OBS_DIM},), got {state.shape}"
        return state

    def encode_batch(
        self,
        obs_list: List[Dict[str, np.ndarray]],
        goal_obs_list: List[Dict[str, np.ndarray]],
    ) -> np.ndarray:
        """
        Encode a list of obs dicts into a (B, 34) float32 array.
        Normalisation is NOT applied — SACAgent handles that.
        """
        return np.stack([self.encode(o, g) for o, g in zip(obs_list, goal_obs_list)])


# ──────────────────────────────────────────────────────────────────────────────
# Actor
# ──────────────────────────────────────────────────────────────────────────────

class SACActorNetwork(nn.Module):
    """Gaussian policy. Expects a pre-normalised (B, input_dim) tensor."""

    def __init__(
        self,
        input_dim: int,
        action_dim: int,
        hidden_dims: List[int],
        activation: str = "relu",
    ):
        super().__init__()
        self.net          = build_mlp(input_dim, hidden_dims[-1], hidden_dims[:-1], activation)
        self.mu_head      = nn.Linear(hidden_dims[-1], action_dim)
        self.log_std_head = nn.Linear(hidden_dims[-1], action_dim)
        logger.debug("SACActorNetwork: input_dim=%d  action_dim=%d  hidden=%s",
                     input_dim, action_dim, hidden_dims)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h       = self.net(x)
        mu      = self.mu_head(h)
        log_std = self.log_std_head(h).clamp(LOG_STD_MIN, LOG_STD_MAX)
        return mu, log_std

    def sample(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mu, log_std = self.forward(x)
        std    = log_std.exp()
        dist   = Normal(mu, std)
        u      = dist.rsample()
        action = torch.tanh(u)
        log_prob = (dist.log_prob(u) - torch.log(1 - action.pow(2) + 1e-6)).sum(-1, keepdim=True)
        return action, log_prob

    def deterministic_action(self, x: torch.Tensor) -> torch.Tensor:
        mu, _ = self.forward(x)
        return torch.tanh(mu)


# ──────────────────────────────────────────────────────────────────────────────
# Critic
# ──────────────────────────────────────────────────────────────────────────────

class SACCriticNetwork(nn.Module):
    """Twin Q-networks. Input is [normalised_state, action]."""

    def __init__(
        self,
        input_dim: int,
        action_dim: int,
        hidden_dims: List[int],
        activation: str = "relu",
    ):
        super().__init__()
        total = input_dim + action_dim
        self.q1 = build_mlp(total, 1, hidden_dims, activation)
        self.q2 = build_mlp(total, 1, hidden_dims, activation)
        logger.debug("SACCriticNetwork: input_dim=%d  action_dim=%d  total=%d  hidden=%s",
                     input_dim, action_dim, total, hidden_dims)

    def forward(
        self, obs_vec: torch.Tensor, action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([obs_vec, action], dim=-1)
        return self.q1(x), self.q2(x)
