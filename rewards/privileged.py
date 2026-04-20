"""
Privileged staged reward for pick-and-place.

Five components, activated progressively.  All components are non-negative
and normalised to [0, 1] so that weights have consistent, interpretable
meaning across components and scenes.

  r1  reach   : 1 - d_eef/max_reach_dist         always active,    ∈ [0, 1]
  r2  grasp   : 1.0 (discrete bonus)              gated on grasp,   ∈ {0, 1}
  r3  lift    : (obj_z - rest_z)/max_lift_dist    gated on grasp,   ∈ [0, 1]
  r4  place   : 1 - d_place/max_place_dist        gated on grasp,   ∈ [0, 1]
  r5  success : 1.0 (discrete bonus)              gated on grasp
                                                  + d_place < success_threshold

Grasp is detected heuristically (no contact sensor needed):
  - EEF is within `grasp_radius` of the object
  - Gripper is sufficiently closed  (gripper_qpos[0] < gripper_close_threshold)
  - Object has risen above its resting z by at least `lift_threshold`

Calibration note for gripper_close_threshold:
  For this robot: open ≈ 0.040, closed ≈ 0.001  → default threshold = 0.020

All thresholds and weights are Hydra-configurable so you can tune without
touching this file.  Set log level to DEBUG during early training to see
all five components printed at every step.
"""
import logging
from typing import Dict, Tuple

import numpy as np

from rewards.base import BaseReward

logger = logging.getLogger(__name__)


class PrivilegedReward(BaseReward):
    """
    Staged dense reward for pick-and-place using ground-truth state signals.

    All reward components are non-negative and normalised to [0, 1],
    so the total reward lives in [0, w_reach + w_grasp + w_lift + w_place + w_success].

    Args:
        object_pos_key:          obs-dict key for the target object position,
                                 e.g. "akita_black_bowl_1_pos"
        eef_pos_key:             obs-dict key for end-effector position
        gripper_qpos_key:        obs-dict key for gripper joint positions

        -- Normalisation distances --
        max_reach_dist:          expected max EEF-to-object distance in the scene (m).
                                 r1 = 0 when d_eef >= this value.
        max_place_dist:          expected max object-to-goal distance (m).
                                 r4 = 0 when d_place >= this value.
        max_lift_dist:           height (m) at which r3 saturates to 1.
                                 Typically the clearance height you want.

        -- Grasp detection --
        grasp_radius:            max EEF-to-object distance to consider grasping (m)
        gripper_close_threshold: gripper_qpos[0] value BELOW which the gripper
                                 is considered closed.  Default 0.020 (halfway
                                 between open≈0.040 and closed≈0.001).
        lift_threshold:          minimum height above rest_z to count as lifted (m)

        -- Weights --
        w_reach, w_grasp, w_lift, w_place, w_success

        -- Success --
        success_threshold:       object-to-goal distance (m) for the success bonus
    """

    def __init__(
        self,
        object_pos_key: str,
        eef_pos_key: str = "robot0_eef_pos",
        gripper_qpos_key: str = "robot0_gripper_qpos",
        # normalisation distances
        max_reach_dist: float = 0.6,
        max_place_dist: float = 0.6,
        max_lift_dist: float = 0.15,
        # grasp detection
        grasp_radius: float = 0.05,
        gripper_close_threshold: float = 0.020,
        lift_threshold: float = 0.02,
        # component weights
        w_reach: float = 1.0,
        w_grasp: float = 5.0,
        w_lift: float = 2.0,
        w_place: float = 3.0,
        w_success: float = 10.0,
        # success criterion
        success_threshold: float = 0.05,
    ):
        if object_pos_key is None:
            raise ValueError(
                "reward.object_pos_key must be set in config. "
                "Example: reward.object_pos_key=akita_black_bowl_1_pos"
            )

        self.object_pos_key = object_pos_key
        self.eef_pos_key = eef_pos_key
        self.gripper_qpos_key = gripper_qpos_key

        self.max_reach_dist = max_reach_dist
        self.max_place_dist = max_place_dist
        self.max_lift_dist = max_lift_dist

        self.grasp_radius = grasp_radius
        self.gripper_close_threshold = gripper_close_threshold
        self.lift_threshold = lift_threshold

        self.w_reach = w_reach
        self.w_grasp = w_grasp
        self.w_lift = w_lift
        self.w_place = w_place
        self.w_success = w_success

        self.success_threshold = success_threshold

        # rest_z is set lazily on the first compute() call each episode.
        # Avoids hardcoding a table height that differs across LIBERO scenes.
        self._rest_z: float | None = None

        self._max_total = w_reach + w_grasp + w_lift + w_place + w_success
        logger.info(
            "PrivilegedReward (staged, non-negative): key=%s  "
            "max_reach=%.2f  max_place=%.2f  max_lift=%.2f  "
            "grasp_radius=%.3f  gripper_close_threshold=%.3f  "
            "max_total_reward=%.1f",
            object_pos_key,
            max_reach_dist, max_place_dist, max_lift_dist,
            grasp_radius, gripper_close_threshold,
            self._max_total,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _detect_grasp(
        self,
        eef_pos: np.ndarray,
        obj_pos: np.ndarray,
        gripper_qpos: np.ndarray,
        obj_z: float,
        rest_z: float,
    ) -> Tuple[bool, float]:
        """
        Returns (grasped: bool, eef_to_obj_dist: float).

        Grasp heuristic (all three must hold):
          1. EEF is within grasp_radius of the object centre
          2. Gripper is sufficiently closed
          3. Object has risen above rest_z by at least lift_threshold

        rest_z is passed explicitly so this helper works for both the
        single-step path (uses self._rest_z) and the batch path (uses
        per-sample rest_z derived from obs).
        """
        d_eef = float(np.linalg.norm(eef_pos - obj_pos))
        gripper_closed = float(gripper_qpos[0]) < self.gripper_close_threshold
        lifted = (obj_z - rest_z) > self.lift_threshold
        grasped = (d_eef < self.grasp_radius) and gripper_closed and lifted
        return grasped, d_eef

    def _compute_from_arrays(
        self,
        eef_pos:      np.ndarray,  # (3,)
        obj_pos:      np.ndarray,  # (3,)
        goal_pos:     np.ndarray,  # (3,)
        gripper_qpos: np.ndarray,  # (2,)
        rest_z:       float,
    ) -> float:
        """
        Core reward computation given pre-extracted arrays and a known rest_z.
        Shared by compute() (single-step) and compute_batch() (HER).
        """
        obj_z = float(obj_pos[2])
        grasped, d_eef = self._detect_grasp(eef_pos, obj_pos, gripper_qpos, obj_z, rest_z)

        r1 = max(0.0, 1.0 - d_eef / self.max_reach_dist)
        r2 = 1.0 if grasped else 0.0
        r3 = min(1.0, max(0.0, (obj_z - rest_z) / self.max_lift_dist)) if grasped else 0.0
        d_place = float(np.linalg.norm(obj_pos - goal_pos))
        r4 = max(0.0, 1.0 - d_place / self.max_place_dist) if grasped else 0.0
        r5 = 1.0 if (grasped and d_place < self.success_threshold) else 0.0

        reward = (
              self.w_reach   * r1
            + self.w_grasp   * r2
            + self.w_lift    * r3
            + self.w_place   * r4
            + self.w_success * r5
        )

        logger.debug(
            "reward=%.4f/%.1f | grasped=%s | "
            "r1(reach)=%.3f  r2(grasp)=%.3f  r3(lift)=%.3f  "
            "r4(place)=%.3f  r5(success)=%.3f | "
            "d_eef=%.3f  d_place=%.3f  obj_z=%.3f  rest_z=%.3f  gripper=%.4f",
            reward, self._max_total, grasped,
            self.w_reach * r1, self.w_grasp * r2, self.w_lift * r3,
            self.w_place * r4, self.w_success * r5,
            d_eef, d_place, obj_z, rest_z, float(gripper_qpos[0]),
        )

        return float(reward)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute(
        self,
        obs: Dict[str, np.ndarray],
        next_obs: Dict[str, np.ndarray],
        goal_obs: Dict[str, np.ndarray],
        **kwargs,  # absorbs unused z / z_next / z_goal from old call sites
    ) -> float:
        """
        Compute reward for a single transition during rollout collection.

        rest_z is lazily initialised from obs on the first call of each
        episode — call reset() at episode start to clear it.
        """
        eef_pos      = np.asarray(next_obs[self.eef_pos_key],      dtype=np.float64)
        obj_pos      = np.asarray(next_obs[self.object_pos_key],   dtype=np.float64)
        goal_pos     = np.asarray(goal_obs[self.object_pos_key],   dtype=np.float64)
        gripper_qpos = np.asarray(next_obs[self.gripper_qpos_key], dtype=np.float64)

        if self._rest_z is None:
            self._rest_z = float(np.asarray(obs[self.object_pos_key], dtype=np.float64)[2])
            logger.debug("PrivilegedReward: rest_z initialised to %.4f", self._rest_z)

        return self._compute_from_arrays(
            eef_pos, obj_pos, goal_pos, gripper_qpos, self._rest_z
        )

    def compute_batch(
        self,
        obs:      Dict[str, np.ndarray],
        next_obs: Dict[str, np.ndarray],
        goal_obs: Dict[str, np.ndarray],
    ) -> np.ndarray:
        """
        Compute rewards for a batch of transitions — used by HER at sample time.

        Args:
            obs:      Dict[str, np.ndarray] where each value has shape (B, *feature_shape)
            next_obs: same structure as obs (post-transition)
            goal_obs: same structure, with relabelled goals

        Returns:
            rewards: (B,) float32 array

        All computation is vectorized over the batch dimension.
        rest_z is derived per-sample from obs[object_pos_key][:, 2], which is
        the object z-height at the start of each transition — the correct resting
        height regardless of where in the episode the transition occurred.
        """
        # --- Extract batched arrays (B, dim) --------------------------
        eef_pos      = next_obs[self.eef_pos_key].astype(np.float64)       # (B, 3)
        obj_pos      = next_obs[self.object_pos_key].astype(np.float64)    # (B, 3)
        goal_pos     = goal_obs[self.object_pos_key].astype(np.float64)    # (B, 3)
        gripper_qpos = next_obs[self.gripper_qpos_key].astype(np.float64)  # (B, 2)
        rest_z       = obs[self.object_pos_key][:, 2].astype(np.float64)   # (B,)

        # --- Vectorized grasp detection -------------------------------
        # condition 1: EEF within grasp_radius of object
        d_eef = np.linalg.norm(eef_pos - obj_pos, axis=1)                 # (B,)
        close_enough = d_eef < self.grasp_radius

        # condition 2: gripper sufficiently closed
        gripper_closed = gripper_qpos[:, 0] < self.gripper_close_threshold # (B,)

        # condition 3: object lifted above rest_z
        obj_z  = obj_pos[:, 2]                                             # (B,)
        lifted = (obj_z - rest_z) > self.lift_threshold                    # (B,)

        grasped = close_enough & gripper_closed & lifted                   # (B,) bool

        # --- Vectorized reward components (all in [0, 1]) -------------
        r1 = np.clip(1.0 - d_eef / self.max_reach_dist, 0.0, 1.0)

        r2 = grasped.astype(np.float64)

        height_gain = np.clip((obj_z - rest_z) / self.max_lift_dist, 0.0, 1.0)
        r3 = np.where(grasped, height_gain, 0.0)

        d_place = np.linalg.norm(obj_pos - goal_pos, axis=1)              # (B,)
        r4 = np.where(grasped, np.clip(1.0 - d_place / self.max_place_dist, 0.0, 1.0), 0.0)

        r5 = np.where(grasped & (d_place < self.success_threshold), 1.0, 0.0)

        # --- Weighted sum ---------------------------------------------
        rewards = (
              self.w_reach   * r1
            + self.w_grasp   * r2
            + self.w_lift    * r3
            + self.w_place   * r4
            + self.w_success * r5
        )

        #logger.info(f"Reward type '{type(rewards)}': reward dim={rewards.shape} ")

        return rewards.astype(np.float32).reshape(-1, 1)

    def reset(self) -> None:
        """
        Call at the start of each episode so rest_z re-initialises
        correctly for the new scene / object placement.
        """
        self._rest_z = None