"""
Replay buffer with RIG-style goal relabeling.

Each transition stores:
    - obs dict (image + proprio + object positions)
    - action
    - next_obs dict
    - done flag
    - episode_id  (to support future-state relabeling)
    - step_in_ep  (step index within episode)

Goal relabeling strategies:
    - buffer:  sample z_goal from a random past transition
    - future:  sample z_goal from a future state in the same episode
    - mixed:   50% buffer + 50% future
"""
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)


class ReplayBuffer:
    """
    Fixed-capacity replay buffer with goal relabeling support.

    Images are stored as uint8 to minimise memory usage.
    """

    def __init__(
        self,
        capacity: int,
        goal_sampling_strategy: str = "mixed",
        future_fraction: float = 0.5,
    ):
        assert goal_sampling_strategy in ("buffer", "future", "mixed"), (
            f"Unknown goal_sampling_strategy: {goal_sampling_strategy}"
        )
        self.capacity = capacity
        self.goal_sampling_strategy = goal_sampling_strategy
        self.future_fraction = future_fraction

        # Storage — populated lazily on first push
        self._obs: Optional[Dict[str, np.ndarray]] = None
        self._next_obs: Optional[Dict[str, np.ndarray]] = None
        self._actions: Optional[np.ndarray] = None
        self._rewards: Optional[np.ndarray] = None
        self._dones: Optional[np.ndarray] = None
        self._episode_ids: Optional[np.ndarray] = None
        self._step_in_eps: Optional[np.ndarray] = None

        self._ptr = 0
        self._size = 0

        # Episode index: maps episode_id → list of buffer indices
        self._episode_index: Dict[int, List[int]] = {}
        self._current_episode_id = 0
        self._current_episode_indices: List[int] = []

        logger.info(
            "ReplayBuffer: capacity=%d, strategy=%s, future_fraction=%.2f",
            capacity, goal_sampling_strategy, future_fraction,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _init_storage(self, obs: Dict[str, np.ndarray], action: np.ndarray):
        """Allocate storage arrays on first push."""
        self._actions = np.zeros((self.capacity, action.shape[0]), dtype=np.float32)
        self._rewards = np.zeros((self.capacity, 1), dtype=np.float32)
        self._dones = np.zeros((self.capacity, 1), dtype=np.float32)
        self._episode_ids = np.zeros(self.capacity, dtype=np.int64)
        self._step_in_eps = np.zeros(self.capacity, dtype=np.int32)

        # One array per obs key
        self._obs = {}
        self._next_obs = {}
        for key, val in obs.items():
            dtype = val.dtype
            shape = val.shape
            self._obs[key] = np.zeros((self.capacity, *shape), dtype=dtype)
            self._next_obs[key] = np.zeros((self.capacity, *shape), dtype=dtype)

        logger.debug("ReplayBuffer storage initialised.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def push(
        self,
        obs: Dict[str, np.ndarray],
        action: np.ndarray,
        next_obs: Dict[str, np.ndarray],
        reward: float,
        done: bool,
        step_in_ep: int,
    ):
        """Add a single transition."""
        if self._obs is None:
            self._init_storage(obs, action)

        idx = self._ptr

        for key in self._obs:
            self._obs[key][idx] = obs[key]
            self._next_obs[key][idx] = next_obs[key]

        self._actions[idx] = action
        self._rewards[idx] = reward
        self._dones[idx] = float(done)
        self._episode_ids[idx] = self._current_episode_id
        self._step_in_eps[idx] = step_in_ep

        # Maintain episode index
        # Remove old entry if we are overwriting
        old_ep = self._episode_ids[idx] if self._size == self.capacity else -1
        if old_ep >= 0 and old_ep in self._episode_index:
            try:
                self._episode_index[old_ep].remove(idx)
            except ValueError:
                pass
            if not self._episode_index[old_ep]:
                del self._episode_index[old_ep]

        self._current_episode_indices.append(idx)

        self._ptr = (self._ptr + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def end_episode(self):
        """Call at the end of each episode to flush the episode index."""
        if self._current_episode_indices:
            self._episode_index[self._current_episode_id] = list(
                self._current_episode_indices
            )
        self._current_episode_id += 1
        self._current_episode_indices = []

    def sample(
        self,
        batch_size: int,
        encoder,          # BaseEncoder — used to encode goal images
        device: torch.device,
    ) -> Tuple[
        torch.Tensor,  # z        (B, latent_dim)
        torch.Tensor,  # actions  (B, action_dim)
        torch.Tensor,  # z_next   (B, latent_dim)
        torch.Tensor,  # rewards  (B, 1)
        torch.Tensor,  # dones    (B, 1)
        torch.Tensor,  # z_goal   (B, latent_dim)
        Dict,          # raw goal obs (for privileged reward recomputation)
    ]:
        assert self._size >= batch_size, (
            f"Buffer has only {self._size} transitions, need {batch_size}."
        )

        idxs = np.random.randint(0, self._size, size=batch_size)

        obs_batch = {k: self._obs[k][idxs] for k in self._obs}
        next_obs_batch = {k: self._next_obs[k][idxs] for k in self._next_obs}
        actions = torch.FloatTensor(self._actions[idxs]).to(device)
        rewards = torch.FloatTensor(self._rewards[idxs]).to(device)
        dones = torch.FloatTensor(self._dones[idxs]).to(device)

        # Sample goal obs according to strategy
        goal_obs_batch = self._sample_goals(idxs)

        # Encode
        z = encoder.encode(obs_batch)
        z_next = encoder.encode(next_obs_batch)
        z_goal = encoder.encode(goal_obs_batch)

        return z, actions, z_next, rewards, dones, z_goal, goal_obs_batch

    def _sample_goals(self, idxs: np.ndarray) -> Dict[str, np.ndarray]:
        """
        For each transition index, sample a goal observation.
        Returns a batched obs dict.
        """
        strategy = self.goal_sampling_strategy
        batch_size = len(idxs)

        if strategy == "buffer":
            goal_idxs = np.random.randint(0, self._size, size=batch_size)

        elif strategy == "future":
            goal_idxs = self._sample_future_goals(idxs)

        elif strategy == "mixed":
            use_future = np.random.rand(batch_size) < self.future_fraction
            future_idxs = self._sample_future_goals(idxs)
            buffer_idxs = np.random.randint(0, self._size, size=batch_size)
            goal_idxs = np.where(use_future, future_idxs, buffer_idxs)

        return {k: self._next_obs[k][goal_idxs] for k in self._next_obs}

    def _sample_future_goals(self, idxs: np.ndarray) -> np.ndarray:
        """
        For each idx, sample a future state from the same episode.
        Falls back to a random buffer idx if the episode has only one step.
        """
        goal_idxs = np.empty_like(idxs)
        for i, idx in enumerate(idxs):
            ep_id = self._episode_ids[idx]
            ep_indices = self._episode_index.get(int(ep_id), [idx])
            step = int(self._step_in_eps[idx])
            future = [j for j in ep_indices if self._step_in_eps[j] > step]
            if future:
                goal_idxs[i] = np.random.choice(future)
            else:
                goal_idxs[i] = np.random.randint(0, self._size)
        return goal_idxs

    def __len__(self) -> int:
        return self._size