"""
GoalsDataset: fixed datasets of goal pairs for two task types.

Class hierarchy:
    GoalsDataset              ← base: generate(), sample_goal(), __len__, __getitem__
    ├── ObjectGoalsDataset    ← captures (image, object_position) by teleporting object
    └── ArmGoalsDataset       ← captures (image, eef_position) by sampling joint angles

Usage:
    # Object task
    dataset = ObjectGoalsDataset(cfg, env)
    dataset.generate()
    goal = dataset.sample_goal()
    goal.image       # uint8 (H, W, 3)
    goal.position    # float32 (3,)  world-frame xyz of object

    # Arm task
    dataset = ArmGoalsDataset(cfg, env)
    dataset.generate()
    goal = dataset.sample_goal()
    goal.image       # uint8 (H, W, 3)
    goal.position    # float32 (3,)  world-frame xyz of EEF
    goal.quat        # float32 (4,)  EEF quaternion (w, x, y, z)
"""
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional
from tqdm import tqdm
from PIL import Image
import os

import numpy as np
from omegaconf import DictConfig

from envs.libero_env import (
    LiberoEnv,
    LiberoObjectEnv,
    LiberoArmEnv,
    PARK_QPOS,
    NEUTRAL_QPOS,
    set_arm_qpos,
    get_object_pos,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# GoalSample
# ---------------------------------------------------------------------------

@dataclass
class GoalSample:
    """
    A single goal: the image the agent should reach and the target pose.

    Fields:
        obs       full obs dict (image + proprio + positions)
        image     uint8 (H, W, 3)
        position  float32 (3,)   world-frame xyz
                    - object task: object position
                    - arm task:    EEF position
        quat      float32 (4,) or None
                    - object task: None  (object orientation not used)
                    - arm task:    EEF quaternion (w, x, y, z)
    """
    obs: Dict[str, np.ndarray]
    image: np.ndarray
    position: np.ndarray                  # float32 (3,)
    quat: Optional[np.ndarray] = None     # float32 (4,) or None

    def save_image_to_file(self, image_path: str, reverse = False) -> None:
        if reverse:
            Image.fromarray(self.image[::-1]).save(image_path)
        else:
            Image.fromarray(self.image).save(image_path)


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class GoalsDataset:
    """
    Base class for goal datasets.

    Handles generate(), sample_goal(), __len__, __getitem__.
    Subclasses must implement _capture_goal().
    """

    def __init__(self, cfg: DictConfig, env: LiberoEnv):
        self.cfg = cfg
        self.env = env
        self._goals: List[GoalSample] = []
        self._camera_key: str = cfg.goals.camera_key

    def generate(self) -> None:
        raise NotImplementedError

    def _capture_goal(self, *args, **kwargs) -> GoalSample:
        raise NotImplementedError

    def sample_goal(self) -> GoalSample:
        if not self._goals:
            raise RuntimeError(
                "GoalsDataset is empty. Call generate() before sample_goal()."
            )
        idx = np.random.randint(0, len(self._goals))
        return self._goals[idx]

    def __len__(self) -> int:
        return len(self._goals)

    def __getitem__(self, idx: int) -> GoalSample:
        return self._goals[idx]


# ---------------------------------------------------------------------------
# Object goal dataset
# ---------------------------------------------------------------------------

class ObjectGoalsDataset(GoalsDataset):
    """
    Goals defined by a random object position on the table.

    Generation procedure per goal:
        1. Reset the environment.
        2. Sample a random (x, y) within the configured workspace limits.
        3. Teleport the object via LiberoObjectEnv.set_object_position().
        4. Capture agentview image + read object position from obs.

    Config keys used (cfg.goals):
        n_goals, n_settle, camera_key, object_pos_key,
        x_min, x_max, y_min, y_max
    """

    def __init__(self, cfg: DictConfig, env: LiberoObjectEnv):
        super().__init__(cfg, env)

        self._object_pos_key: str = (
            cfg.goals.object_pos_key
            if cfg.goals.object_pos_key is not None
            else f"{cfg.env.object_name}_1_pos"
        )

        logger.info(
            "ObjectGoalsDataset: n_goals=%d, workspace x=[%.3f, %.3f] y=[%.3f, %.3f]",
            cfg.goals.n_goals,
            cfg.goals.x_min, cfg.goals.x_max,
            cfg.goals.y_min, cfg.goals.y_max,
        )

    def generate(self) -> None:
        n = self.cfg.goals.n_goals
        logger.info("Generating %d object goals...", n)

        self._goals = []
        rng = np.random.default_rng(self.cfg.seed)

        xs = rng.uniform(self.cfg.goals.x_min, self.cfg.goals.x_max, size=n)
        ys = rng.uniform(self.cfg.goals.y_min, self.cfg.goals.y_max, size=n)

        for i in tqdm(range(n), desc="Generating object goals"):
            goal = self._capture_goal(xs[i], ys[i])
            self._goals.append(goal)

        self.env.reset()
        logger.info("Object goal generation complete. Dataset size: %d", len(self._goals))

    def _capture_goal(self, x: float, y: float) -> GoalSample:
        self.env.reset()

        obs = self.env.set_object_position(
            x=x,
            y=y,
            n_settle=self.cfg.goals.n_settle,
        )

        if obs is None:
            raise RuntimeError("set_object_position returned None.")

        image = obs[self._camera_key].copy()

        position = get_object_pos(obs, self.cfg.env.object_name)
        if position is None:
            logger.warning(
                "Could not read object position from obs for goal (%.4f, %.4f). "
                "Using requested (x, y) with z=0.97 as fallback.",
                x, y,
            )
            position = np.array([x, y, 0.97], dtype=np.float32)
        else:
            position = position.astype(np.float32)

        logger.debug(
            "Object goal captured: requested=(%.4f, %.4f), actual=(%.4f, %.4f, %.4f)",
            x, y, position[0], position[1], position[2],
        )
        return GoalSample(obs=obs, image=image, position=position)


# ---------------------------------------------------------------------------
# Arm goal dataset
# ---------------------------------------------------------------------------

# Safe joint sampling ranges for the Sawyer arm (radians).
# These keep the arm above the table and within the camera frame.
# Override via cfg.goals.joint_limits if needed.
_DEFAULT_JOINT_LIMITS = {
    "j1": (-1.0,  1.0),
    "j2": (-1.5, -0.3),
    "j3": (-0.5,  0.5),
    "j4": ( 0.3,  1.5),
    "j5": (-0.5,  0.5),
    "j6": ( 0.2,  0.8),
    "j7": (-0.5,  0.5),
}


class ArmGoalsDataset(GoalsDataset):
    """
    Goals defined by a random arm pose (EEF position + orientation).

    Generation procedure per goal:
        1. Reset the environment.
        2. Sample random joint angles within per-joint limits.
        3. Teleport the arm via set_arm_qpos().
        4. Capture agentview image + read EEF position and quaternion from obs.

    GoalSample.position  float32 (3,)  EEF xyz
    GoalSample.quat      float32 (4,)  EEF quaternion (w, x, y, z)

    The gripper is held open (matches PARK_QPOS convention).

    Config keys used (cfg.goals):
        n_goals, n_settle, camera_key,
        joint_limits.j1 .. joint_limits.j7  (each a [min, max] list)
    """

    def __init__(self, cfg: DictConfig, env: LiberoArmEnv):
        super().__init__(cfg, env)

        # Merge defaults with anything specified in config
        limits_cfg = cfg.goals.get("joint_limits", {})
        self._joint_limits = []
        for key, default in _DEFAULT_JOINT_LIMITS.items():
            if key in limits_cfg:
                lo, hi = limits_cfg[key]
            else:
                lo, hi = default
            self._joint_limits.append((lo, hi))
        # self._joint_limits is now a list of 7 (lo, hi) tuples, one per arm joint

        logger.info(
            "ArmGoalsDataset: n_goals=%d, joint_limits=%s",
            cfg.goals.n_goals,
            self._joint_limits,
        )

    def generate(self) -> None:
        n = self.cfg.goals.n_goals
        logger.info("Generating %d arm goals...", n)

        self._goals = []
        rng = np.random.default_rng(self.cfg.seed)

        # Sample all joint configurations upfront for reproducibility
        joint_configs = np.array([
            [rng.uniform(lo, hi) for lo, hi in self._joint_limits]
            for _ in range(n)
        ])  # (n, 7)

        for i in tqdm(range(n), desc="Generating arm goals"):
            goal = self._capture_goal(joint_configs[i])
            self._goals.append(goal)
            # Temp - write each goal image to file for debugging
            goal.save_image_to_file(os.path.join(self.cfg.output_dir, f"arm_goal_{i}.png"))


        # Restore neutral pose after generation
        set_arm_qpos(self.env._env, NEUTRAL_QPOS, n_settle=self.cfg.goals.n_settle)
        self.env.reset()
        logger.info("Arm goal generation complete. Dataset size: %d", len(self._goals))

    def _capture_goal(self, joint_angles: np.ndarray) -> GoalSample:
        """
        Args:
            joint_angles: (7,) array of arm joint angles in radians
        """
        self.env.reset()

        # Build full qpos: 7 arm joints + gripper open
        qpos = np.concatenate([joint_angles, [0.020, -0.020]])

        obs = set_arm_qpos(
            self.env._env,
            qpos,
            n_settle=self.cfg.goals.n_settle,
        )

        if obs is None:
            raise RuntimeError("set_arm_qpos returned None.")

        image = obs[self._camera_key].copy()

        # EEF position
        eef_pos = obs.get("robot0_eef_pos")
        if eef_pos is None:
            raise RuntimeError(
                "'robot0_eef_pos' not found in obs. "
                f"Available keys: {sorted(obs.keys())}"
            )
        position = np.array(eef_pos, dtype=np.float32)

        # EEF orientation
        eef_quat = obs.get("robot0_eef_quat")
        if eef_quat is None:
            raise RuntimeError(
                "'robot0_eef_quat' not found in obs. "
                f"Available keys: {sorted(obs.keys())}"
            )
        quat = np.array(eef_quat, dtype=np.float32)

        logger.debug(
            "Arm goal captured: joints=%s, eef=(%.4f, %.4f, %.4f), quat=%s",
            np.round(joint_angles, 3), position[0], position[1], position[2],
            np.round(quat, 3),
        )
        return GoalSample(obs=obs, image=image, position=position, quat=quat)