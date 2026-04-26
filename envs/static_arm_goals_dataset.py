"""
StaticArmGoalsDataset: loads a pre-generated arm-pose dataset from a .npz file
and exposes train / eval subsets for use in training and evaluation loops.

The .npz file is expected to have been produced by gen_arm_dataset.py and
contain at minimum the following keys:
    images       uint8  (N, H, W, 3)
    eef_pos      float32 (N, 3)
    eef_quat     float32 (N, 4)
    obs_dicts    object  (N,)   — array of dicts

The dataset is split into train/eval subsets by randomly shuffling the indices
with a fixed seed, then slicing at the configured split percentage. This gives
reproducible splits as long as the same .npz file and split seed are used.

Usage:
    dataset = StaticArmGoalsDataset(cfg)
    dataset.load()

    goal = dataset.sample_goal(split="train")
    goal = dataset.sample_goal(split="eval")

    goal.image      # uint8  (H, W, 3)
    goal.position   # float32 (3,)   EEF xyz
    goal.quat       # float32 (4,)   EEF quaternion (w, x, y, z)
    goal.obs        # dict[str, np.ndarray]

Config keys used (cfg.goals):
    dataset_path       path to the .npz file
    train_split        fraction of data used for training  (e.g. 0.9)
    cfg.seed         seed used for the train/eval shuffle (default: 0)
    camera_key         obs key for the image (used for documentation only;
                       images are stored pre-extracted in the .npz)
"""
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
from omegaconf import DictConfig
from tqdm import tqdm

from .goals_dataset import GoalSample

logger = logging.getLogger(__name__)

from intrinsic_motivation.learning_progress import compute_lp


class StaticArmGoalsDataset:
    """
    Loads a pre-generated arm-pose dataset from disk and exposes
    reproducible train / eval subsets.

    Attributes:
        n_total   total number of samples in the .npz file
        n_train   number of samples in the train subset
        n_eval    number of samples in the eval subset
    """

    VALID_SPLITS = ("train", "eval")

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self._seed = int(self.cfg.get("seed", 0))
        self._sample_rngs = {
            "train": np.random.default_rng(self._seed),
            "eval": np.random.default_rng(self._seed + 1),
        }
        self._train_goals: List[GoalSample] = []
        self._eval_goals:  List[GoalSample] = []
        self.n_total = 0
        self.n_train = 0
        self.n_eval  = 0
        self._results_queues: List = []  # For tracking results for each goal
        self._lps: List = []  # For tracking learning progress for each goal

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load(self) -> None:
        """
        Load the .npz file, shuffle indices reproducibly, and populate
        the train / eval subsets.
        """
        path = self.cfg.goals.dataset_path
        train_split = float(self.cfg.goals.train_split)

        logger.info("Loading arm dataset from: %s", path)
        data = np.load(path, allow_pickle=True)

        images    = data["images"]      # (N, H, W, 3)  uint8
        eef_pos   = data["eef_pos"]     # (N, 3)
        eef_quat  = data["eef_quat"]    # (N, 4)
        obs_dicts = data["obs_dicts"]   # (N,) object array of dicts

        self.n_total = len(images)
        logger.info("Total samples in file: %d", self.n_total)

        # --- Reproducible shuffle ---
        rng = np.random.default_rng(self._seed)
        indices = np.arange(self.n_total)
        rng.shuffle(indices)

        # Reset sampling RNGs on each load so repeated runs are reproducible.
        self._sample_rngs = {
            "train": np.random.default_rng(self._seed),
            "eval": np.random.default_rng(self._seed + 1),
        }

        n_train = int(self.n_total * train_split)
        train_idx = indices[:n_train]
        eval_idx  = indices[n_train:]

        self.n_train = len(train_idx)
        self.n_eval  = len(eval_idx)

        logger.info(
            "Split: train=%d (%.1f%%)  eval=%d (%.1f%%)",
            self.n_train, 100 * self.n_train / self.n_total,
            self.n_eval,  100 * self.n_eval  / self.n_total,
        )

        # --- Build GoalSample lists ---
        self._train_goals = self._build_goals(images, eef_pos, eef_quat, obs_dicts, train_idx, "train")
        self._eval_goals  = self._build_goals(images, eef_pos, eef_quat, obs_dicts, eval_idx,  "eval")

        logger.info("Dataset loaded. train=%d  eval=%d", self.n_train, self.n_eval)

    def _build_goals(
        self,
        images:    np.ndarray,
        eef_pos:   np.ndarray,
        eef_quat:  np.ndarray,
        obs_dicts: np.ndarray,
        indices:   np.ndarray,
        split_name: str,
    ) -> List[GoalSample]:
        goals = []
        for idx in tqdm(indices, desc=f"Loading {split_name} goals", unit="sample"):
            goal = GoalSample(
                obs=obs_dicts[idx],          # dict[str, np.ndarray]
                image=images[idx],           # uint8 (H, W, 3)
                position=eef_pos[idx].astype(np.float32),
                quat=eef_quat[idx].astype(np.float32),
            )
            goals.append(goal)
        return goals

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def sample_goal(self, split: str = "train") -> GoalSample:
        """
        Sample a random goal from the requested split.

        Args:
            split: "train" or "eval"

        Returns:
            GoalSample

        Raises:
            ValueError: if split is not "train" or "eval"
            RuntimeError: if load() has not been called yet
        """
        if split not in self.VALID_SPLITS:
            raise ValueError(f"split must be one of {self.VALID_SPLITS}, got '{split}'")

        goals = self._train_goals if split == "train" else self._eval_goals

        if not goals:
            raise RuntimeError(
                f"The {split} subset is empty. "
                "Make sure load() was called and the dataset file is non-empty."
            )

        idx = int(self._sample_rngs[split].integers(0, len(goals)))

        #if split == "eval":
        #    idx = np.random.randint(0, len(goals))

        #else:  # train split: ε-greedy over LP
        #    N = len(goals)
        #    lp_values = np.abs(np.array(self._lps))  # |LP_i| for all goals

            # ε-greedy proportional probability matching
            #eps = self.cfg.goals.epsilon
            #uniform = np.ones(N) / N
            #lp_sum = lp_values.sum()

            #if lp_sum == 0:
            #    probs = uniform  # fallback: all LPs are 0 at the start
            #else:
            #    probs = eps * uniform + (1 - eps) * (lp_values / lp_sum)

            #idx = np.random.choice(N, p=probs)

        return goals[idx], idx

    def _update_intrinsic_motivation(self, goal_idx: int, result: float):
        results = self._results_queues[goal_idx]
        results.append(result)
        n_eval = len(results)
        self._lps[goal_idx] = compute_lp(results, n_eval)

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        """Total number of samples across both splits."""
        return self.n_total

    def get_split(self, split: str) -> List[GoalSample]:
        """Return the full list for a given split (for iteration)."""
        if split not in self.VALID_SPLITS:
            raise ValueError(f"split must be one of {self.VALID_SPLITS}, got '{split}'")
        return self._train_goals if split == "train" else self._eval_goals
