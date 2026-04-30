"""
LIBERO environment wrapper.

Class hierarchy:
    LiberoEnv                  ← base: env creation, step loop, episode truncation
    ├── LiberoObjectEnv        ← object task: BDDL with object, success on object pos
    └── LiberoArmEnv           ← arm task:    BDDL with no object, success on EEF pos
"""
import logging
import tempfile
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import torch

import numpy as np
from omegaconf import DictConfig
from encoders.base import BaseEncoder


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# BDDL templates
# ---------------------------------------------------------------------------

BDDL_OBJECT_TEMPLATE = """\
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

# No object — just the robot arm on a table.
BDDL_ARM_TEMPLATE = """\
(define (problem {problem_name})
  (:domain robosuite)
  (:language {language})
  (:regions
    (table_region
        (:target main_table)
        (:ranges (
            (-0.3 -0.3 0.3 0.3)
          )
        )
    )
  )
  (:fixtures
    main_table - table
  )
  (:objects
  )
  (:init
  )
  (:goal
    (And )
  )
)
"""


def write_object_bddl(
    object_name: str,
    cx: float,
    cy: float,
    half_size: float = 0.005,
    language: str = "place the object on the table",
) -> str:
    """Write a single-object BDDL to a temp file, return its path."""
    problem_name = "libero_custom_single_object"
    content = BDDL_OBJECT_TEMPLATE.format(
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
    logger.info("Object BDDL written to: %s", tmp.name)
    logger.debug("BDDL content:\n%s", content)
    return tmp.name


def write_arm_bddl(language: str = "move the arm to the target position") -> str:
    """Write a no-object BDDL to a temp file, return its path."""
    problem_name = "libero_custom_single_object"
    content = BDDL_ARM_TEMPLATE.format(
        problem_name=problem_name,
        language=language,
    )
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".bddl", delete=False, prefix=f"{problem_name}_arm_"
    )
    tmp.write(content)
    tmp.flush()
    logger.info("Arm BDDL written to: %s", tmp.name)
    logger.debug("BDDL content:\n%s", content)
    return tmp.name


# ---------------------------------------------------------------------------
# Arm helpers (shared)
# ---------------------------------------------------------------------------

PARK_QPOS = np.array([
    0.0, -1.57, 0.0, 1.57, 0.0, 0.5, 0.0,   # arm joints q1..q7
    0.020, -0.020                              # gripper (open)
])

NEUTRAL_QPOS = np.array([
    0.0, -0.785, 0.0, 0.785, 0.0, 0.30, 0.0,
    0.020, -0.020
])

_JOINT_NAMES = [
    "robot0_joint1", "robot0_joint2", "robot0_joint3",
    "robot0_joint4", "robot0_joint5", "robot0_joint6", "robot0_joint7",
    "gripper0_finger_joint1", "gripper0_finger_joint2",
]


def set_arm_qpos(env, qpos: np.ndarray, n_settle: int = 50):
    """Directly set joint positions via MuJoCo sim, settle, return last obs."""
    sim = env.sim
    for i, name in enumerate(_JOINT_NAMES):
        try:
            jnt_id = sim.model.joint_name2id(name)
            addr = sim.model.jnt_qposadr[jnt_id]
            sim.data.qpos[addr] = qpos[i]
        except Exception:
            logger.warning("Could not set joint '%s' — skipping.", name)
    sim.forward()
    obs = None
    dummy = [0.0] * 7
    for _ in range(n_settle):
        obs, _, _, _ = env.step(dummy)
    return obs


# ---------------------------------------------------------------------------
# SuccessInfo
# ---------------------------------------------------------------------------

@dataclass
class SuccessInfo:
    """
    Returned by check_custom_success() in all LiberoEnv subclasses.

    Fields:
        success   whether the success condition is met
        pos_err   L2 position error in metres (object xyz or EEF xyz)
        ori_err   angular orientation error in radians
                    - LiberoObjectEnv: always 0.0 (orientation not checked)
                    - LiberoArmEnv:    EEF quaternion error
    """
    success: bool
    pos_err: float
    ori_err: float = 0.0


# ---------------------------------------------------------------------------
# Object position helper
# ---------------------------------------------------------------------------

def get_object_pos(obs: dict, object_name: str) -> Optional[np.ndarray]:
    """Return the object's world-frame (x, y, z) from obs, or None."""
    key = f"{object_name}_1_pos"
    if key in obs:
        return np.array(obs[key])
    candidates = [k for k in obs if object_name in k and "pos" in k]
    if candidates:
        logger.warning("Key '%s' not found; using '%s' instead.", key, candidates[0])
        return np.array(obs[candidates[0]])
    logger.warning(
        "No position key found for object '%s'. Available keys: %s",
        object_name, sorted(obs.keys()),
    )
    return None


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class LiberoEnv:
    """
    Base wrapper around LIBERO's OffScreenRenderEnv.

    Handles everything that is common to all task types:
      - OffScreenRenderEnv creation from a BDDL file
      - step() loop with episode truncation
      - close()

    Subclasses must implement:
      - _make_bddl() -> str          path to a written BDDL file
      - reset() -> obs dict
      - check_custom_success(obs) -> SuccessInfo
    """

    def __init__(self, cfg: DictConfig, encoder=None):
        from libero.libero.envs import OffScreenRenderEnv

        self.cfg = cfg
        self.episode_step = 0
        self.action_dim = cfg.env.action_dim
        self.obs: Optional[Dict[str, np.ndarray]] = None
        self.encoder = encoder

        bddl_path = self._make_bddl()

        env_args = {
            "bddl_file_name": bddl_path,
            "hard_reset": False,
            "camera_heights": cfg.env.camera_heights,
            "camera_widths": cfg.env.camera_widths,
        }
        self._env = OffScreenRenderEnv(**env_args)
        self._env.seed(cfg.seed)
        self._env.reset()

        self._init_state = self._env.sim.get_state().flatten()
        logger.info("%s ready.", self.__class__.__name__)

    def _make_bddl(self) -> str:
        raise NotImplementedError

    def reset(self, goal_coordinates: np.ndarray = None) -> Dict[str, np.ndarray]:
        raise NotImplementedError

    def check_custom_success(self, obs: dict, z: np.ndarray) -> SuccessInfo:
        raise NotImplementedError

    def step(
        self, action: np.ndarray
    ) -> Tuple[Dict[str, np.ndarray], float, bool, dict]:
        """
        Step the environment.

        Args:
            action: (action_dim,) array in [-1, 1]

        Returns:
            (next_obs, reward, done, info)
        """
        obs, reward, done, info = self._env.step(action.tolist())
        z_next = self.encoder.encode(obs[self.cfg.encoder.camera_key])
        self.episode_step += 1

        success_info = self.check_custom_success(obs, z_next)

        truncated = self.episode_step >= self.cfg.env.episode_length
        if truncated:
            logger.info("TRUNCATED")

        if success_info.success:
            logger.info(
                "SUCCESS pos_err=%.4f m  ori_err=%.4f rad",
                success_info.pos_err, success_info.ori_err,
            )

        done = success_info.success or truncated
        self.obs = obs
        info["success_info"] = success_info
        return obs, z_next, float(reward), done, info

    def check_success(self) -> bool:
        """Query LIBERO's built-in task success condition."""
        return bool(self._env.check_success())

    def close(self):
        self._env.close()
        logger.debug("%s closed.", self.__class__.__name__)


# ---------------------------------------------------------------------------
# Object task
# ---------------------------------------------------------------------------

class LiberoObjectEnv(LiberoEnv):
    """
    Single-object tabletop task.
    The robot must move an object to a target (x, y, z) position.

    Config keys used (cfg.env):
        object_name        MuJoCo asset name (e.g. "milk")
        success_threshold  distance threshold for success (metres)
    """

    def __init__(self, cfg: DictConfig):
        self.object_name = cfg.env.object_name
        self.target_pos = np.zeros(3)
        self.success_threshold = cfg.env.success_threshold
        super().__init__(cfg)

    def _make_bddl(self) -> str:
        return write_object_bddl(
            object_name=self.object_name,
            cx=0.0,
            cy=0.0,
            half_size=0.005,
        )

    def reset(self, goal_coordinates: np.ndarray = None) -> Dict[str, np.ndarray]:
        self._env.reset()
        self._env.set_init_state(self._init_state)
        self.episode_step = 0
        if goal_coordinates is not None:
            self.target_pos = goal_coordinates

        logger.info("RESETTING LiberoObjectEnv")
        logger.info("--> Goal coordinates: %s", self.target_pos)

        obs, _, _, _ = self._env.step([0.0] * self.action_dim)
        self.obs = obs
        logger.info("--> Object position: %s", get_object_pos(obs, self.object_name))
        return obs

    def check_custom_success(self, obs: dict, z: np.ndarray) -> SuccessInfo:
        obj_pos = get_object_pos(obs, self.object_name)
        if obj_pos is None:
            return SuccessInfo(success=False, pos_err=float("inf"))
        pos_err = float(np.linalg.norm(obj_pos - self.target_pos))
        success = pos_err < self.cfg.env.success_threshold
        return SuccessInfo(success=success, pos_err=pos_err)

    def set_object_position(
        self,
        x: float,
        y: float,
        object_name: Optional[str] = None,
        n_settle: int = 50,
    ) -> Dict[str, np.ndarray]:
        """
        Teleport the object to (x, y) on the table surface and let physics
        settle. z and orientation are preserved.
        """
        if object_name is None:
            object_name = f"{self.object_name}_1"

        sim = self._env.sim

        try:
            joint_name = f"{object_name}_joint0"
            jnt_id = sim.model.joint_name2id(joint_name)
            addr = sim.model.jnt_qposadr[jnt_id]
        except Exception:
            try:
                body_id = sim.model.body_name2id(object_name)
                jnt_id = sim.model.body_jntadr[body_id]
                addr = sim.model.jnt_qposadr[jnt_id]
            except Exception as e:
                raise RuntimeError(
                    f"Could not locate free joint for object '{object_name}'. "
                    f"Original error: {e}"
                )

        current_z = float(sim.data.qpos[addr + 2])
        current_quat = sim.data.qpos[addr + 3: addr + 7].copy()

        sim.data.qpos[addr + 0] = x
        sim.data.qpos[addr + 1] = y
        sim.data.qpos[addr + 2] = current_z
        sim.data.qpos[addr + 3: addr + 7] = current_quat

        try:
            jnt_vel_addr = sim.model.jnt_dofadr[jnt_id]
            sim.data.qvel[jnt_vel_addr: jnt_vel_addr + 6] = 0.0
        except Exception:
            logger.error("Could not zero object velocity — skipping.")

        sim.forward()

        obs = None
        dummy = [0.0] * self.action_dim
        for _ in range(n_settle):
            obs, _, _, _ = self._env.step(dummy)

        self.obs = obs
        logger.debug("Object '%s' moved to (%.4f, %.4f).", object_name, x, y)
        return obs


# ---------------------------------------------------------------------------
# Arm task
# ---------------------------------------------------------------------------

class LiberoArmEnv(LiberoEnv):
    """
    No-object arm reaching task.
    The robot must move its end-effector to a target pose (position + orientation).

    Success requires both conditions to be true simultaneously:
        - L2 position error  < cfg.env.success_pos_threshold  (metres)
        - Angular orientation error < cfg.env.success_ori_threshold  (radians)

    Orientation error is computed from the quaternion dot product, with the
    abs() handling the quaternion double-cover (q and -q are the same rotation):
        angular_error = arccos(clip(|q_current · q_goal|, 0, 1))

    Config keys used (cfg.env):
        success_pos_threshold   metres  (e.g. 0.05)
        success_ori_threshold   radians (e.g. 0.2  ≈ 11 degrees)
    """

    def __init__(self, cfg: DictConfig, encoder=None):
        self.target_pos  = np.zeros(3)   # EEF xyz
        self.target_quat = np.array([1.0, 0.0, 0.0, 0.0])  # EEF quaternion (w,x,y,z)
        self.target_latent = None  # for level 3
        super().__init__(cfg, encoder=encoder)

    def _make_bddl(self) -> str:
        return write_arm_bddl()

    def reset(
        self,
        goal_coordinates: np.ndarray = None,   # (3,)  EEF xyz
        goal_quat: np.ndarray = None,           # (4,)  EEF quaternion
        latent_goal = None,                            # for level 3
    ) -> Dict[str, np.ndarray]:
        self._env.reset()
        self._env.set_init_state(self._init_state)
        self.episode_step = 0

        if goal_coordinates is not None:
            self.target_pos = np.array(goal_coordinates)
        if goal_quat is not None:
            self.target_quat = np.array(goal_quat)
        if latent_goal is not None:
            self.target_latent = latent_goal


        logger.info("RESETTING LiberoArmEnv")
        logger.info("--> Goal EEF position:    %s", self.target_pos)
        logger.info("--> Goal EEF quaternion:  %s", self.target_quat)
        logger.info("--> Goal latent:          %s", self.target_latent)

        obs, _, _, _ = self._env.step([0.0] * self.action_dim)
        self.obs = obs
        logger.info("--> Current EEF position: %s", obs.get("robot0_eef_pos"))
        logger.info("--> Current EEF quat:     %s", obs.get("robot0_eef_quat"))
        logger.info("--> Current latent:       %s", obs.get("latent_obs"))

        return obs

    def check_custom_success(self, obs, z) -> SuccessInfo:
        if self.cfg.level in (1,2):
            # --- position ---
            eef_pos = obs.get("robot0_eef_pos")
            if eef_pos is None:
                logger.warning("'robot0_eef_pos' not in obs — cannot check success.")
                return SuccessInfo(success=False, pos_err=float("inf"), ori_err=float("inf"))
            pos_err = float(np.linalg.norm(np.array(eef_pos) - self.target_pos))

            # --- orientation ---
            eef_quat = obs.get("robot0_eef_quat")
            if eef_quat is None:
                logger.warning("'robot0_eef_quat' not in obs — cannot check success.")
                return SuccessInfo(success=False, pos_err=pos_err, ori_err=float("inf"))
            dot = np.abs(np.dot(np.array(eef_quat), self.target_quat))
            ori_err = float(np.arccos(np.clip(dot, 0.0, 1.0)))

            success = (
                pos_err < self.cfg.env.success_pos_threshold
                and ori_err < self.cfg.env.success_ori_threshold
            )
        else:
            # For level 3, success is determined by proximity in latent space.
            if self.target_latent is None:
                return SuccessInfo(success=False, pos_err=float("inf"), ori_err=float("inf"))
            logger.info("z: %s", z)
            logger.info("target_latent: %s", self.target_latent)
            pos_err = torch.linalg.norm(z - self.target_latent).cpu().item()
            ori_err = 0.0  # orientation not checked for level 3
            success = pos_err < self.cfg.env.success_latent_threshold 

        return SuccessInfo(success=success, pos_err=pos_err, ori_err=ori_err)