"""
LIBERO environment wrapper.

Wraps OffScreenRenderEnv into a clean interface that:
  - returns obs dicts consistently
  - handles episode resets and goal obs management
  - exposes check_success()
"""
import logging
import os
import tempfile
from typing import Dict, Optional, Tuple

import numpy as np
from omegaconf import DictConfig

logger = logging.getLogger(__name__)







# ---------------------------------------------------------------------------
# BDDL template  (format reverse-engineered from real LIBERO bddl files)
# ---------------------------------------------------------------------------
# Regions use (:ranges ((xmin ymin xmax ymax))) — a rectangle on the table.
# cx/cy is the centre; hs is the half-size of the rectangle.
# The region name in (:init ...) must be prefixed with the fixture name:
#   main_table_obj_init_region
#
# Object declaration format: `name_1 - type`   (instance_name - class_name)
#
# Available object types (browse libero/libero/assets/stable_objects/):
#   akita_black_bowl, plate, wine_glass, moka_pot, porcelain_mug,
#   red_coffee_mug, chefmate_8_frypan, ketchup, ...

BDDL_TEMPLATE = """\
(define (problem {problem_name})
  (:domain robosuite)
  (:language {language})
  (:regions
    (obj_init_region
        (:target main_table)
        (:ranges (
            ({xmin:.4f} {ymin:.4f} {xmax:.4f} {ymax:.4f})
          )
        )
    )
  )
  (:fixtures
    main_table - table
  )
  (:objects
    {object_name}_1 - {object_name}
  )
  (:obj_of_interest
    {object_name}_1
  )
  (:init
    (On {object_name}_1 main_table_obj_init_region)
  )
  (:goal
    (And )
  )
)
"""


def write_bddl(
    object_name: str,
    cx: float,
    cy: float,
    half_size: float = 0.005,
    language: str = "place the object on the table",
) -> str:
    """
    Write a minimal BDDL file to a temp location and return its path.
    The problem name must match a key in TASK_MAPPING — it corresponds to
    the registered problem class in libero/libero/envs/problems/.
    """
    problem_name = "libero_custom_single_object"
    content = BDDL_TEMPLATE.format(
        problem_name=problem_name,
        language=language,
        object_name=object_name,
        xmin=cx - half_size,
        ymin=cy - half_size,
        xmax=cx + half_size,
        ymax=cy + half_size,
    )
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".bddl", delete=False, prefix=f"{problem_name}_"
    )
    tmp.write(content)
    tmp.flush()
    logger.info(f"BDDL written to: {tmp.name}")
    logger.debug("BDDL content:\n%s", content)
    return tmp.name


# ---------------------------------------------------------------------------
# Arm parking helpers
# ---------------------------------------------------------------------------
# Joint values that raise the arm straight up, clearing the workspace.
# Used to capture clean goal images without the arm in frame.

PARK_QPOS = np.array([
    0.0, -1.57, 0.0, 1.57, 0.0, 0.5, 0.0,  # arm joints q1..q7
    0.020, -0.020                             # gripper (open)
])

NEUTRAL_QPOS = np.array([
    0.0, -0.785, 0.0, 0.785, 0.0, 0.30, 0.0,
    0.020, -0.020
])

# Sawyer joint names as used in LIBERO's MuJoCo model
_JOINT_NAMES = [
    "robot0_joint1", "robot0_joint2", "robot0_joint3",
    "robot0_joint4", "robot0_joint5", "robot0_joint6", "robot0_joint7",
    "robot0_r_gripper_l_finger_joint", "robot0_r_gripper_r_finger_joint",
]


def set_arm_qpos(env, qpos: np.ndarray, n_settle: int = 50):
    """
    Directly set joint positions via MuJoCo sim, then step so physics
    settles. Returns the last observation.
    """
    sim = env.sim
    for i, name in enumerate(_JOINT_NAMES):
        try:
            jnt_id = sim.model.joint_name2id(name)
            addr   = sim.model.jnt_qposadr[jnt_id]
            sim.data.qpos[addr] = qpos[i]
        except Exception:
            logger.warning(f"Could not set joint '{name}' — skipping.")
    sim.forward()
    obs = None
    dummy = [0.0] * 7
    for _ in range(n_settle):
        obs, _, _, _ = env.step(dummy)
    return obs


# ---------------------------------------------------------------------------
# Object position helper
# ---------------------------------------------------------------------------

def get_object_pos(obs: dict, object_name: str):
    """Return the object's world-frame (x, y, z) position from obs, or None."""
    key = f"{object_name}_1_pos"
    if key in obs:
        return np.array(obs[key])
    candidates = [k for k in obs if object_name in k and "pos" in k]
    if candidates:
        logger.warning(f"Key '{key}' not found; using '{candidates[0]}' instead.")
        return np.array(obs[candidates[0]])
    logger.warning(f"No position key found for object '{object_name}'. "
                   f"Available keys: {sorted(obs.keys())}")
    return None




class LiberoEnv:
    """
    Thin wrapper around LIBERO's OffScreenRenderEnv.

    Attributes:
        action_dim:  dimensionality of the action space (7)
        obs:         current observation dict (set after reset/step)
    """

    def __init__(self, cfg: DictConfig):
        from libero.libero.envs import OffScreenRenderEnv

        self.cfg = cfg
        self.episode_step = 0
        self.action_dim = cfg.env.action_dim

        self.object_name = "milk"

        # ------------------------------------------------------------------
        # 1. Write BDDL and create environment
        # ------------------------------------------------------------------
        bddl_path = write_bddl(
            object_name=self.object_name,
            cx=0.0,
            cy=0.0,
            half_size=0.005, # Placement rectangle half-size (m); smaller = more deterministic
        )

        # ---- Build environment ---------------------------------------
        env_args = {
            "bddl_file_name": bddl_path,
            "hard_reset": False,
            "camera_heights": cfg.env.camera_heights,
            "camera_widths": cfg.env.camera_widths,
        }
        self._env = OffScreenRenderEnv(**env_args)
        self._env.seed(cfg.seed)
        self._env.reset()

        # Store initial state for reproducible resets
        self._init_state = self._env.sim.get_state().flatten()
        self.obs: Optional[Dict[str, np.ndarray]] = None

        self.target_pos = np.array([6, 7, 1])  #  z apres le reset est de 0.96967218

        self.success_threshold = 0.1

        logger.info("LiberoEnv ready.")

    def reset(self, goal_coordinates: np.ndarray = None) -> Dict[str, np.ndarray]:
        """
        Reset the environment.

        Args:
            init_state_idx: which initial state to use. If None, cycles
                            through states sequentially.

        Returns:
            obs dict
        """
        self._env.reset()
        self._env.set_init_state(self._init_state)
        self.episode_step = 0
        if goal_coordinates is not None:
            self.target_pos = goal_coordinates

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

        done = self.check_custom_success(obs)

        truncated = self.episode_step >= self.cfg.env.episode_length
        done = bool(done) or truncated

        self.obs = obs
        return obs, float(reward), done, info

    def check_custom_success(self, obs):
        obj_pos = get_object_pos(obs, self.object_name)
        # print('obj_pos', obj_pos)
        if obj_pos is None:
            return False

        # dist = np.linalg.norm(obj_pos[:2] - self.target_pos[:2]) # on calcule la distance sur la base des coordonées (x,y)

        dist = np.linalg.norm(obj_pos - self.target_pos)

        return dist < self.success_threshold

    def check_success(self) -> bool:
        """Query LIBERO's built-in task success condition."""
        return bool(self._env.check_success())

    def close(self):
        self._env.close()
        logger.debug("LiberoEnv closed.")
