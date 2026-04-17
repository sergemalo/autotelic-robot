"""
LIBERO environment wrapper.

Wraps OffScreenRenderEnv into a clean interface that:
  - returns obs dicts consistently
  - handles episode resets and goal obs management
  - exposes check_success()
  - exposes set_object_position() for goal generation
"""
import logging
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
    logger.info("BDDL written to: %s", tmp.name)
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
    "gripper0_finger_joint1", "gripper0_finger_joint2",
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
# Object position helper
# ---------------------------------------------------------------------------

def get_object_pos(obs: dict, object_name: str) -> Optional[np.ndarray]:
    """Return the object's world-frame (x, y, z) position from obs, or None."""
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
# LiberoEnv
# ---------------------------------------------------------------------------

class LiberoEnv:
    """
    Thin wrapper around LIBERO's OffScreenRenderEnv for a single-object
    custom scene defined by a generated BDDL file.

    Attributes:
        action_dim:  dimensionality of the action space (7)
        obs:         current observation dict (set after reset/step)
    """

    def __init__(self, cfg: DictConfig):
        from libero.libero.envs import OffScreenRenderEnv

        self.cfg = cfg
        self.episode_step = 0
        self.action_dim = cfg.env.action_dim

        self.object_name = cfg.env.object_name

        # ------------------------------------------------------------------
        # Write BDDL and create environment
        # ------------------------------------------------------------------
        bddl_path = write_bddl(
            object_name=self.object_name,
            cx=0.0,
            cy=0.0,
            half_size=0.005,  # Placement rectangle half-size (m)
        )

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


    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------

    def reset(self, goal_coordinates: np.ndarray = None) -> Dict[str, np.ndarray]:
        """
        Reset the environment to the stored initial state.

        Returns:
            obs dict
        """
        self._env.reset()
        self._env.set_init_state(self._init_state)
        self.episode_step = 0
        if goal_coordinates is not None:
            self.target_pos = goal_coordinates

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

    # ------------------------------------------------------------------
    # Goal generation support
    # ------------------------------------------------------------------

    def set_object_position(
        self,
        x: float,
        y: float,
        object_name: Optional[str] = None,
        n_settle: int = 50,
    ) -> Dict[str, np.ndarray]:
        """
        Teleport the object to (x, y) on the table surface and let physics
        settle.

        The z coordinate is kept at its current value (resting on the table),
        so only x and y are controlled. The object's quaternion is preserved.

        Args:
            x:           target x position in world frame (metres)
            y:           target y position in world frame (metres)
            object_name: name of the object body in the MuJoCo model.
                         Defaults to cfg.env.object_name + "_1".
            n_settle:    number of no-op steps to let physics settle after
                         the teleport.

        Returns:
            obs dict after settling
        """
        if object_name is None:
            object_name = f"{self.cfg.env.object_name}_1"

        sim = self._env.sim

        # Locate the free joint that controls this object.
        # Free joints have 7 DOF in qpos: [x, y, z, qw, qx, qy, qz].
        try:
            joint_name = f"{object_name}_joint0"
            logger.debug("Locating object joint by name: '%s'", joint_name)
            jnt_id = sim.model.joint_name2id(joint_name)
            logger.debug("Located object joint: id=%d", jnt_id)
            addr = sim.model.jnt_qposadr[jnt_id]
        except Exception:
            # Fallback: try to find the joint via body name
            try:
                body_id = sim.model.body_name2id(object_name)
                logger.debug("Locating object joint via body name: '%s'", object_name)
                jnt_id = sim.model.body_jntadr[body_id]
                logger.debug("Located object joint via body: id=%d", jnt_id)
                addr = sim.model.jnt_qposadr[jnt_id]
                logger.debug(
                    "Located object joint via body fallback: body=%s, jnt_id=%d",
                    object_name, jnt_id,
                )
            except Exception as e:
                raise RuntimeError(
                    f"Could not locate free joint for object '{object_name}'. "
                    f"Check that cfg.env.object_name matches the MuJoCo body name. "
                    f"Original error: {e}"
                )

        # Read current z and quaternion — preserve them
        current_z = float(sim.data.qpos[addr + 2])
        current_quat = sim.data.qpos[addr + 3: addr + 7].copy()

        # Write new x, y; keep z and orientation
        sim.data.qpos[addr + 0] = x
        sim.data.qpos[addr + 1] = y
        sim.data.qpos[addr + 2] = current_z
        sim.data.qpos[addr + 3: addr + 7] = current_quat

        # Zero out object velocity to prevent sliding after teleport
        try:
            jnt_vel_addr = sim.model.jnt_dofadr[jnt_id]
            sim.data.qvel[jnt_vel_addr: jnt_vel_addr + 6] = 0.0
        except Exception:
            logger.error("Could not zero object velocity — skipping.")

        sim.forward()

        # Settle physics with no-op steps
        obs = None
        dummy = [0.0] * self.action_dim
        for _ in range(n_settle):
            obs, _, _, _ = self._env.step(dummy)

        self.obs = obs
        logger.debug(
            "Object '%s' moved to (%.4f, %.4f). Settled over %d steps.",
            object_name, x, y, n_settle,
        )
        return obs
