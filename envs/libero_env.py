"""
LIBERO environment wrapper.

Wraps OffScreenRenderEnv into a clean interface that:
  - returns obs dicts consistently
  - handles episode resets and goal obs management
  - exposes check_success()
"""
import logging
import os
from typing import Dict, Optional, Tuple

import numpy as np
from omegaconf import DictConfig

logger = logging.getLogger(__name__)


class LiberoEnv:
    """
    Thin wrapper around LIBERO's OffScreenRenderEnv.

    Attributes:
        action_dim:  dimensionality of the action space (7)
        obs:         current observation dict (set after reset/step)
    """

    def __init__(self, cfg: DictConfig):
        from libero.libero import benchmark
        from libero.libero.envs import OffScreenRenderEnv
        from libero.libero import get_libero_path

        self.cfg = cfg
        self.episode_step = 0
        self.action_dim = cfg.env.action_dim

        # ---- Load task -----------------------------------------------
        benchmark_dict = benchmark.get_benchmark_dict()
        task_suite_name = cfg.env.task_suite
        task_suite = benchmark_dict[task_suite_name]()

        task_id = cfg.env.task_id
        task = task_suite.get_task(task_id)

        bddl_file = os.path.join(
            get_libero_path("bddl_files"),
            task.problem_folder,
            task.bddl_file,
        )

        logger.info(
            "Loading LIBERO task: suite=%s, id=%d, description='%s'",
            task_suite_name, task_id, task.language,
        )

        # ---- Build environment ---------------------------------------
        env_args = {
            "bddl_file_name": bddl_file,
            "camera_heights": cfg.env.camera_heights,
            "camera_widths": cfg.env.camera_widths,
        }
        self._env = OffScreenRenderEnv(**env_args)
        self._env.seed(cfg.seed)

        # Store initial states for reproducible resets
        self._init_states = task_suite.get_task_init_states(task_id)
        self._n_init_states = len(self._init_states)
        self._init_state_idx = 0

        self.task_language = task.language
        self.obs: Optional[Dict[str, np.ndarray]] = None

        logger.info("LiberoEnv ready. %d initial states available.", self._n_init_states)

    def reset(self, init_state_idx: Optional[int] = None) -> Dict[str, np.ndarray]:
        """
        Reset the environment.

        Args:
            init_state_idx: which initial state to use. If None, cycles
                            through states sequentially.

        Returns:
            obs dict
        """
        self._env.reset()

        if init_state_idx is not None:
            idx = init_state_idx % self._n_init_states
        else:
            idx = self._init_state_idx % self._n_init_states
            self._init_state_idx += 1

        self._env.set_init_state(self._init_states[idx])
        self.episode_step = 0

        # Take a no-op step to get a clean obs
        obs, _, _, _ = self._env.step([0.0] * self.action_dim)
        self.obs = obs
        return obs

    def step(
        self, action: np.ndarray
    ) -> Tuple[Dict[str, np.ndarray], float, bool, dict]:
        """
        Step the environment.

        Args:
            action: (action_dim,) array in [-1, 1]

        Returns:
            (next_obs, reward, done, info)
            Note: reward here is LIBERO's sparse +1 success reward.
                  Your reward function replaces/augments this externally.
        """
        obs, reward, done, info = self._env.step(action.tolist())
        self.episode_step += 1

        truncated = self.episode_step >= self.cfg.env.episode_length
        done = bool(done) or truncated

        self.obs = obs
        return obs, float(reward), done, info

    def check_success(self) -> bool:
        """Query LIBERO's built-in task success condition."""
        return bool(self._env.check_success())

    def close(self):
        self._env.close()
        logger.debug("LiberoEnv closed.")
