"""
custom_libero_scene.py

Sets up a minimal LIBERO environment with a single object placed at a
specific position on the table — no benchmark, no task suite, no init_states.

Object placement is controlled by --cx / --cy (centre of the placement
rectangle, in metres, relative to the table workspace centre):
  - x : forward/backward  (positive = away from robot)
  - y : left/right        (positive = left from robot's POV)

A small half-size (default 0.005 m) makes placement near-deterministic.

Outputs (saved under output/<timestamp>_custom_scene/):
  - goal_image.png      : captured with arm parked out of the way
  - scene_recording.mp4 : short idle rollout with arm at neutral pose
"""

import argparse
import os
import tempfile
import logging
from datetime import datetime

import numpy as np
import imageio

from libero.libero.envs import OffScreenRenderEnv

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
date_prefix = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
output_dir  = f"output/{date_prefix}_custom_scene"
os.makedirs(output_dir, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f"{output_dir}/run.log"),
    ],
)
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Custom LIBERO scene — single object at a fixed position"
    )
    parser.add_argument("--object",    default="akita_black_bowl",
                        help="Object type name (must exist in LIBERO assets)")
    parser.add_argument("--cx",        type=float, default=0.0,
                        help="Object centre x on table (m, relative to workspace centre)")
    parser.add_argument("--cy",        type=float, default=0.0,
                        help="Object centre y on table (m, relative to workspace centre)")
    parser.add_argument("--half_size", type=float, default=0.005,
                        help="Placement rectangle half-size (m); smaller = more deterministic")
    parser.add_argument("--img_size",  type=int,   default=224)
    parser.add_argument("--steps",     type=int,   default=50,
                        help="Number of idle steps to record after setup")
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # 1. Write BDDL and create environment
    # ------------------------------------------------------------------
    bddl_path = write_bddl(
        object_name=args.object,
        cx=args.cx,
        cy=args.cy,
        half_size=args.half_size,
    )

    logger.info("Creating OffScreenRenderEnv...")
    env = OffScreenRenderEnv(
        bddl_file_name=bddl_path,
        camera_heights=args.img_size,
        camera_widths=args.img_size,
    )
    env.seed(42)
    obs = env.reset()
    logger.info("Environment ready.")
    logger.info("Observation keys: %s", sorted(obs.keys()))

    # ------------------------------------------------------------------
    # 2. Log object position at reset
    # ------------------------------------------------------------------
    obj_pos = get_object_pos(obs, args.object)
    if obj_pos is not None:
        logger.info(f"Object '{args.object}' position at reset: {obj_pos}")

    # ------------------------------------------------------------------
    # 3. Capture goal image with arm parked out of the way
    # ------------------------------------------------------------------
    logger.info("Parking arm for goal-image capture...")
    obs_parked = set_arm_qpos(env, PARK_QPOS, n_settle=50)
    goal_image = obs_parked["agentview_image"][::-1]   # flip y-axis
    goal_path  = os.path.join(output_dir, "goal_image.png")
    imageio.imwrite(goal_path, goal_image)
    logger.info(f"Goal image saved: {goal_path}")

    obj_pos_parked = get_object_pos(obs_parked, args.object)
    if obj_pos_parked is not None:
        logger.info(f"Object position while parked: {obj_pos_parked}")

    # ------------------------------------------------------------------
    # 4. Restore arm to neutral and record a short idle video
    # ------------------------------------------------------------------
    logger.info("Restoring arm to neutral pose...")
    obs = set_arm_qpos(env, NEUTRAL_QPOS, n_settle=50)

    frames = [obs["agentview_image"][::-1]]
    dummy_action = [0.0] * 7
    logger.info(f"Recording {args.steps} idle steps...")
    for _ in range(args.steps):
        obs, _, _, _ = env.step(dummy_action)
        frames.append(obs["agentview_image"][::-1])

    env.close()
    os.unlink(bddl_path)

    # ------------------------------------------------------------------
    # 5. Save video
    # ------------------------------------------------------------------
    video_path = os.path.join(output_dir, "scene_recording.mp4")
    imageio.mimsave(video_path, frames, fps=20)
    logger.info(f"Video saved: {video_path}")
    logger.info("DONE.")


if __name__ == "__main__":
    main()
