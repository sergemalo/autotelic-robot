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
        self._z: Optional[torch.Tensor] = None
        self._next_obs: Optional[Dict[str, np.ndarray]] = None
        self._next_z: Optional[torch.Tensor] = None
        self._actions: Optional[np.ndarray] = None
        self._rewards: Optional[np.ndarray] = None
        self._dones: Optional[np.ndarray] = None
        self._episode_ids: Optional[np.ndarray] = None
        self._step_in_eps: Optional[np.ndarray] = None

        self._z_goal_original: Optional[List] = None  # Store original goals
        self._goal_obs_original: Optional[Dict[str, np.ndarray]] = None  # For privileged reward
        self._is_warmup: Optional[np.ndarray] = None  # Flag warmup transitions

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
        # For z and next_z, we have tensors of size latent_dim on the GPU, so we will store them as tensors directly rather than numpy arrays.
        self._z = [None] * self.capacity
        self._next_z = [None] * self.capacity

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


        self._z_goal_original = [None] * self.capacity
        self._is_warmup = np.zeros(self.capacity, dtype=bool)

        # For privileged reward recomputation
        self._goal_obs_original = {}
        for key in obs.keys():
            self._goal_obs_original[key] = np.zeros((self.capacity, *obs[key].shape), dtype=obs[key].dtype)

        logger.debug("ReplayBuffer storage initialised.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def push(
        self,
        obs: Dict[str, np.ndarray],
        z: torch.Tensor,
        action: np.ndarray,
        next_obs: Dict[str, np.ndarray],
        next_z: torch.Tensor,
        reward: float,
        done: bool,
        step_in_ep: int,
        z_goal: Optional[torch.Tensor] = None, 
        goal_obs: Optional[Dict] = None,
        is_warmup: bool = False
    ):
        """Add a single transition."""
        if self._obs is None:
            self._init_storage(obs, action)

        idx = self._ptr

        for key in self._obs:
            self._obs[key][idx] = obs[key]
            self._next_obs[key][idx] = next_obs[key]

        self._z[idx] = z.squeeze(0)
        self._next_z[idx] = next_z.squeeze(0)
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


        self._is_warmup[idx] = is_warmup
        if z_goal is not None:
            self._z_goal_original[idx] = z_goal.squeeze(0)
            if goal_obs is not None:
                for key in self._goal_obs_original:
                    self._goal_obs_original[key][idx] = goal_obs[key]

        if self._size == 1:
            self._log_sample_size()

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
        device: torch.device,
    ) -> Tuple[
        torch.Tensor,  # z        (B, latent_dim)
        torch.Tensor,  # actions  (B, action_dim)
        Dict,          # next_obs (dict of multiple different modalities, each (B, ...))
        torch.Tensor,  # z_next   (B, latent_dim)
        torch.Tensor,  # rewards  (B, 1)
        torch.Tensor,  # dones    (B, 1)
        torch.Tensor,  # z_goal   (B, latent_dim)
        Dict,          # raw goal obs (for privileged reward recomputation)
        np.ndarray,    # relabel mask (True if relabeled, False if original), CPU version
    ]:
        assert self._size >= batch_size, (
            f"Buffer has only {self._size} transitions, need {batch_size}."
        )

        idxs = np.random.randint(0, self._size, size=batch_size) # sample indices 

        # convert to tensors and move to device
        obs_batch = {key: arr[idxs] for key, arr in self._obs.items()}
        z_batch = torch.stack([self._z[i] for i in idxs]).to(device)
        next_obs_batch = {key: arr[idxs] for key, arr in self._next_obs.items()}
        next_z_batch = torch.stack([self._next_z[i] for i in idxs]).to(device)
        actions = torch.FloatTensor(self._actions[idxs]).to(device)
        rewards = torch.FloatTensor(self._rewards[idxs]).to(device)
        dones = torch.FloatTensor(self._dones[idxs]).to(device)

        #------ Goal relabeling ---------------------------------------
        # warmup transitions are always relabeled
        is_warmup = self._is_warmup[idxs] 
        use_relabeled = is_warmup.copy()   # start with warmup mask (True for warmup, False for non-warmup)

        # For non-warmup: 50% keep original, 50% relabel
        non_warmup_mask = ~is_warmup  # only consider non-warmup transitions for random relabeling
        relabel_dice = np.random.rand(batch_size) < 0.5 # 50% chance to relabel for non-warmup transitions
        use_relabeled[non_warmup_mask] = relabel_dice[non_warmup_mask] # combine with warmup mask to get final relabeling decision


        # Construct goal batches based on relabeling decisions
        z_goal_batch = []
        goal_obs_batch = {k: [] for k in self._next_obs}
        
        for i, idx in enumerate(idxs):
            if use_relabeled[i]:
                # Relabel: sample new goal (future or buffer)
                goal_idx = self._sample_relabeled_goal_idx(idx)
                z_goal_batch.append(self._z[goal_idx])
                for key in goal_obs_batch:
                    goal_obs_batch[key].append(self._next_obs[key][goal_idx])
            else:
                # Keep original
                z_goal_batch.append(self._z_goal_original[idx])
                for key in goal_obs_batch:
                    goal_obs_batch[key].append(self._goal_obs_original[key][idx])
        
        z_goal_batch = torch.stack(z_goal_batch).to(device)
        goal_obs_batch = {k: np.array(v) for k, v in goal_obs_batch.items()}

        
        return obs_batch, z_batch, actions, next_obs_batch, next_z_batch, rewards, dones, z_goal_batch, goal_obs_batch, use_relabeled
    
    def _sample_relabeled_goal_idx(self, idx: int) -> int:
        """Sample either future or buffer goal (50/50 mix)."""
        if np.random.rand() < self.future_fraction:
            return self._sample_future_goal_idx(idx)
        else:
            return np.random.randint(0, self._size)

    def _sample_future_goal_idx(self, idx: int) -> int:
        """Sample from future in same episode."""
        ep_id = self._episode_ids[idx]
        ep_indices = self._episode_index.get(int(ep_id), [idx])
        step = int(self._step_in_eps[idx])
        future = [j for j in ep_indices if self._step_in_eps[j] > step]

        if future: 
            return np.random.choice(future)
        else: # if no future (e.g. idx is from last step of episode), fall back to buffer sampling
            return np.random.randint(0, self._size)

    def reset(self):
        self._ptr = 0
        
    def __len__(self) -> int:
        return self._size
    
    def _log_sample_size(self):
        sample_size_bytes = 0
        for key in self._obs:
            sample_size_bytes += self._obs[key][0].nbytes
            sample_size_bytes += self._next_obs[key][0].nbytes

        sample_size_bytes += self._actions[0].nbytes
        sample_size_bytes += self._rewards[0].nbytes
        sample_size_bytes += self._dones[0].nbytes
        sample_size_bytes += self._episode_ids[0].nbytes
        sample_size_bytes += self._step_in_eps[0].nbytes

        sample_size_gpu_bytes = self._z[0].element_size() * self._z[0].nelement() * 2

        logger.info(f"Estimated size per sample: {sample_size_bytes / 1e3:.2f} KB (CPU) + {sample_size_gpu_bytes / 1e3:.2f} KB (GPU)")
        logger.info(f"Estimated total buffer size: {(sample_size_bytes * self.capacity) / 1e9:.2f} GB (CPU) + {(sample_size_gpu_bytes * self.capacity) / 1e9:.2f} GB (GPU)")