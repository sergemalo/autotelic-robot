"""
Privileged staged reward for pick-and-place.

Six components, activated progressively.  All components are non-negative
and normalised to [0, 1] so that weights have consistent, interpretable
meaning across components and scenes.

  r1       reach           : 1 - d_eef/max_reach_dist        always active,    ∈ [0, 1]
  r1_delta reach_delta     : (d_eef_prev - d_eef) / max_reach_dist
                                                              always active,    ∈ (-1, 1]
                             potential-based shaping (Ng 1999); rewards moving
                             toward the object, penalises moving away.
  r1_5     grip_near       : proximity(eef,obj) × gripper_closure
                                                              always active,    ∈ [0, 1]
                             bridges the hover→grasp gap: the agent must be
                             close AND closing the gripper to score.
  r_premature_close        : -(far_from_obj × closure)       always active,    ∈ [-1, 0]
                             penalises closing the gripper before reaching
                             grasp_radius — prevents "arrive with closed fist".
  r_above  approach_above  : proximity × (eef_z - obj_z) / approach_height
                                                              always active,    ∈ [0, 1]
                             rewards approaching from above the object to
                             reduce lateral knock-away on contact.
  r2       grasp           : grasp_confidence (SOFT)         ∈ [0, 1]
                             = proximity × closure × lift_conf
                             smooth surrogate for the hard grasp gate;
                             avoids reward cliff at the grasp boundary.
  r3       lift            : grasp_confidence × lift_height/max_lift_dist
                                                              ∈ [0, 1]
  r4       place           : 1 - d_place/max_place_dist      always active,    ∈ [0, 1]
  r5       success         : 1.0 (discrete bonus)            gated on HARD grasped
                                                              + d_place < success_threshold
                             hard threshold is intentional for success detection.
  r_slow   slow_near       : -(proximity × action_speed)     always active,    ∈ [-1, 0]
                             penalises large EEF delta actions when near the object.

Grasp detection heuristic (no contact sensor needed):
  - EEF is within `grasp_radius` of the object
  - Gripper is sufficiently closed  (gripper_qpos[0] < gripper_close_threshold)
  - Object has risen above its resting z by at least `lift_threshold`

Gripper calibration note:
  For the default LIBERO robot:  open ≈ 0.040 m,  closed ≈ 0.001 m
  gripper_open_width default = 0.040
  gripper_close_threshold default = 0.020  (halfway)

All thresholds and weights are Hydra-configurable — no need to touch this file.
Set log level to DEBUG during early training to see all components at every step.
"""
import logging
from typing import Dict, Optional, Tuple

import numpy as np

from rewards.base import BaseReward

logger = logging.getLogger(__name__)


class PrivilegedReward(BaseReward):
    """
    Staged dense reward for pick-and-place using ground-truth state signals.

    All reward components are non-negative (except r1_delta which can be
    negative as a penalty), and normalised so the weights have consistent
    meaning.

    Args:
        object_pos_key:           obs-dict key for target object position
                                  e.g. "akita_black_bowl_1_pos"
        eef_pos_key:              obs-dict key for end-effector position
        gripper_qpos_key:         obs-dict key for gripper joint positions

        -- Normalisation --
        max_reach_dist:           expected max EEF-to-object distance (m);
                                  r1 = 0 when d_eef >= this value
        max_place_dist:           expected max object-to-goal distance (m)
        max_lift_dist:            height (m) at which r3 saturates to 1

        -- Gripper --
        gripper_open_width:       qpos[0] when gripper is fully open (m)
        grasp_radius:             max EEF-to-object distance to consider grasping (m)
        gripper_close_threshold:  qpos[0] below which gripper is "closed"

        -- Grasp / lift --
        lift_threshold:           min rise in obj_z (m) to confirm grasp
        success_threshold:        d_place below which episode is a success (m)

        -- Weights --
        w_reach:            weight for r1 (distance to object)
        w_reach_delta:      weight for r1_delta (potential-based shaping)
        w_grip_near:        weight for r1_5 (gripper closure near object)
        w_premature_close:  weight for r_premature_close (penalty for closing
                            gripper before reaching grasp_radius); set 0 to disable
        w_above:            weight for r_above (bonus for approaching from above);
                            set 0 to disable
        w_grasp:            weight for r2 (grasp bonus, now soft)
        w_lift:             weight for r3 (lift height, now soft)
        w_place:            weight for r4 (distance to goal)
        w_success:          weight for r5 (success bonus, hard threshold)
        w_slow:             weight for r_slow (speed penalty near object)
    """

    def __init__(
        self,
        object_pos_key: str,
        eef_pos_key: str = "robot0_eef_pos",
        gripper_qpos_key: str = "robot0_gripper_qpos",
        # normalisation
        max_reach_dist: float = 0.6,
        max_place_dist: float = 0.6,
        max_lift_dist: float = 0.15,
        # gripper
        gripper_open_width: float = 0.040,
        grasp_radius: float = 0.06,
        gripper_close_threshold: float = 0.020,
        # grasp / lift
        lift_threshold: float = 0.02,
        success_threshold: float = 0.05,
        # approach
        approach_height: float = 0.05,
        # weights
        w_reach: float = 1.0,
        w_reach_delta: float = 0.5,
        w_grip_near: float = 1.0,
        w_premature_close: float = 0.5,
        w_above: float = 0.5,
        w_grasp: float = 2.0,
        w_lift: float = 1.0,
        w_place: float = 1.0,
        w_success: float = 5.0,
        w_slow: float = 0.5,
    ):
        if object_pos_key is None:
            raise ValueError(
                "reward.object_pos_key must be set. "
                "Example: reward.object_pos_key=akita_black_bowl_1_pos"
            )
        for name, val in [
            ("max_reach_dist", max_reach_dist),
            ("max_place_dist", max_place_dist),
            ("max_lift_dist", max_lift_dist),
            ("gripper_open_width", gripper_open_width),
        ]:
            if val <= 0:
                raise ValueError(f"{name} must be > 0, got {val}")

        self.object_pos_key = object_pos_key
        self.eef_pos_key = eef_pos_key
        self.gripper_qpos_key = gripper_qpos_key

        self.max_reach_dist = max_reach_dist
        self.max_place_dist = max_place_dist
        self.max_lift_dist = max_lift_dist

        self.gripper_open_width = gripper_open_width
        self.grasp_radius = grasp_radius
        self.gripper_close_threshold = gripper_close_threshold

        self.lift_threshold = lift_threshold
        self.success_threshold = success_threshold
        self.approach_height = approach_height

        self.w_reach = w_reach
        self.w_reach_delta = w_reach_delta
        self.w_grip_near = w_grip_near
        self.w_premature_close = w_premature_close
        self.w_above = w_above
        self.w_grasp = w_grasp
        self.w_lift = w_lift
        self.w_place = w_place
        self.w_success = w_success
        self.w_slow = w_slow

        # episode state for compute() (scalar path only)
        self._rest_z: Optional[float] = None

        logger.info(
            "PrivilegedReward initialised | object_key=%s | "
            "weights: reach=%.1f delta=%.1f grip_near=%.1f "
            "premature_close=%.1f above=%.1f "
            "grasp=%.1f lift=%.1f place=%.1f success=%.1f slow=%.1f",
            object_pos_key,
            w_reach, w_reach_delta, w_grip_near,
            w_premature_close, w_above,
            w_grasp, w_lift, w_place, w_success, w_slow,
        )

    # ------------------------------------------------------------------
    # Vectorized core
    # ------------------------------------------------------------------

    def _compute_vectorized(
        self,
        eef_pos:       np.ndarray,   # (B, 3)
        eef_pos_prev:  np.ndarray,   # (B, 3)  — position BEFORE the step
        obj_pos:       np.ndarray,   # (B, 3)  — object position after step
        goal_pos:      np.ndarray,   # (B, 3)
        gripper_qpos:  np.ndarray,   # (B, 2)
        rest_z:        np.ndarray,   # (B,)
        action:        np.ndarray,   # (B, action_dim) — only first 3 dims (EEF delta) used
    ) -> Tuple[np.ndarray, dict]:
        """
        Returns
        -------
        rewards : (B, 1) float32
        info    : dict of (B,) arrays, one per component — useful for logging
        """
        B = eef_pos.shape[0]

        # ── distances ────────────────────────────────────────────────────
        d_eef      = np.linalg.norm(eef_pos - obj_pos, axis=1)          # (B,)
        d_eef_prev = np.linalg.norm(eef_pos_prev - obj_pos, axis=1)     # (B,)
        d_place    = np.linalg.norm(obj_pos - goal_pos, axis=1)         # (B,)
        obj_z      = obj_pos[:, 2]                                       # (B,)
        eef_z      = eef_pos[:, 2]                                       # (B,)

        # ── gripper closure ───────────────────────────────────────────────
        # gripper_qpos[:, 0]: 0 = closed, gripper_open_width = fully open
        closure = np.clip(
            1.0 - gripper_qpos[:, 0] / self.gripper_open_width,
            0.0, 1.0,
        )  # (B,)  1 = fully closed, 0 = fully open

        # ── proximity kernel ──────────────────────────────────────────────
        # Shared by r1_5, r_above, r_premature_close, r_slow.
        # peaks at 1 when d_eef = 0, decays linearly to 0 at grasp_radius.
        proximity = np.clip(1.0 - d_eef / self.grasp_radius, 0.0, 1.0)  # (B,)
        far_from_obj = 1.0 - proximity                                   # (B,)

        # ── r1  reach ────────────────────────────────────────────────────
        r1 = np.clip(1.0 - d_eef / self.max_reach_dist, 0.0, 1.0)

        # ── r1_delta  potential-based shaping ────────────────────────────
        # Positive when moving toward the object, negative when moving away.
        # Guaranteed not to change the optimal policy (Ng et al. 1999).
        r1_delta = np.clip(
            (d_eef_prev - d_eef) / self.max_reach_dist,
            -1.0, 1.0,
        )

        # ── r1_5  gripper closure near object ────────────────────────────
        # The product forces the agent to be BOTH close AND closing the gripper.
        r1_5 = proximity * closure

        # ── r_premature_close  penalty for closing gripper too early ──────
        # Fires when gripper is closed but EEF is still outside grasp_radius.
        # Without this, the agent can score r1_5 by arriving with a closed
        # gripper and dragging it toward the object — never actually grasping.
        r_premature_close = -(far_from_obj * closure)                    # ∈ [-1, 0]

        # ── r_above  approach-from-above bonus ───────────────────────────
        # Rewards approaching from above the object to reduce lateral
        # knock-away on contact. Only active within grasp_radius (via proximity).
        # eef_z - obj_z > 0 means EEF is above the object.
        height_above = np.clip(eef_z - obj_z, 0.0, self.approach_height)
        r_above = proximity * (height_above / self.approach_height)      # ∈ [0, 1]

        # ── grasp detection — SOFT ────────────────────────────────────────
        # grasp_confidence ∈ [0, 1]: smooth surrogate for the hard gate.
        # Avoids reward cliff at the grasp boundary for r2/r3.
        # Hard `grasped` flag is retained ONLY for r5 (success is binary).
        lift_height      = np.maximum(obj_z - rest_z, 0.0)              # (B,)
        lift_conf        = np.clip(lift_height / self.lift_threshold, 0.0, 1.0)
        grasp_confidence = proximity * closure * lift_conf               # (B,) ∈ [0, 1]

        grasped = (
            (d_eef              <  self.grasp_radius)               &
            (gripper_qpos[:, 0] <  self.gripper_close_threshold)    &
            (lift_height        >  self.lift_threshold)
        ).astype(np.float32)                                             # (B,)

        # ── r2  grasp bonus (soft) ────────────────────────────────────────
        r2 = grasp_confidence

        # ── r3  lift height (soft) ────────────────────────────────────────
        r3 = grasp_confidence * np.clip(lift_height / self.max_lift_dist, 0.0, 1.0)

        # ── r4  place distance ────────────────────────────────────────────
        r4 = np.clip(1.0 - d_place / self.max_place_dist, 0.0, 1.0)

        # ── r5  success ───────────────────────────────────────────────────
        # Hard threshold is intentional: success is binary.
        r5 = (grasped * (d_place < self.success_threshold)).astype(np.float32)

        # ── r_slow  speed penalty near object ────────────────────────────
        # action_speed normalised by sqrt(3) so max-norm action gives 1.0.
        action_speed = np.linalg.norm(action[:, :3], axis=1) / np.sqrt(3)  # (B,)
        r_slow = -(proximity * action_speed)                                # ∈ [-1, 0]

        # ── total ─────────────────────────────────────────────────────────
        total = (
            self.w_reach            * r1                +
            self.w_reach_delta      * r1_delta          +
            self.w_grip_near        * r1_5              +
            self.w_premature_close  * r_premature_close +
            self.w_above            * r_above           +
            self.w_grasp            * r2                +
            self.w_lift             * r3                +
            self.w_place            * r4                +
            self.w_success          * r5                +
            self.w_slow             * r_slow
        ).astype(np.float32)  # (B,)

        info = dict(
            r1=r1, r1_delta=r1_delta, r1_5=r1_5,
            r_premature_close=r_premature_close,
            r_above=r_above,
            r2=r2, r3=r3, r4=r4, r5=r5, r_slow=r_slow,
            grasped=grasped, grasp_confidence=grasp_confidence,
            d_eef=d_eef, d_place=d_place,
        )

        return total.reshape(B, 1), info
    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Call at the start of every episode (scalar path only)."""
        self._rest_z = None

    def compute(
        self,
        obs: Dict[str, np.ndarray],
        next_obs: Dict[str, np.ndarray],
        goal_obs: Dict[str, np.ndarray],
        action: np.ndarray,
        z=None, z_next=None, z_goal=None,
    ) -> float:
        """
        Scalar reward for a single transition during environment interaction.

        Maintains `_rest_z` as episode state: call `reset()` at episode start.

        Args:
            action: the action taken at this step, shape (action_dim,)
        """
        eef_prev   = obs[self.eef_pos_key].reshape(1, 3)
        eef        = next_obs[self.eef_pos_key].reshape(1, 3)
        obj        = next_obs[self.object_pos_key].reshape(1, 3)
        goal       = goal_obs[self.object_pos_key].reshape(1, 3)
        gripper    = next_obs[self.gripper_qpos_key].reshape(1, -1)
        action_b   = np.asarray(action, dtype=np.float32).reshape(1, -1)

        if self._rest_z is None:
            self._rest_z = float(obs[self.object_pos_key][2])
        rest_z = np.array([self._rest_z], dtype=np.float32)

        reward, info = self._compute_vectorized(
            eef, eef_prev, obj, goal, gripper, rest_z, action_b
        )

        scalar = float(reward.squeeze())
        info_s = {k: float(v.squeeze()) for k, v in info.items()}
        logger.debug(
            "reward=%.3f | r1=%.3f Δ=%.3f grip_near=%.3f "
            "pre_close=%.3f above=%.3f slow=%.3f | "
            "r2(soft)=%.3f r3=%.3f r4=%.3f r5=%.0f | "
            "grasp_conf=%.3f grasped=%d",
            scalar,
            info_s["r1"], info_s["r1_delta"], info_s["r1_5"],
            info_s["r_premature_close"], info_s["r_above"], info_s["r_slow"],
            info_s["r2"], info_s["r3"], info_s["r4"], info_s["r5"],
            info_s["grasp_confidence"], int(info_s["grasped"]),
        )
        return scalar

    def compute_batch(
        self,
        obs: Dict[str, np.ndarray],
        next_obs: Dict[str, np.ndarray],
        goal_obs: Dict[str, np.ndarray],
        action: np.ndarray,
        rest_z: np.ndarray,
    ) -> np.ndarray:
        """
        Vectorized reward for a batch of transitions from the replay buffer.

        Args:
            obs:      Dict where each value has shape (B, *feature_shape)
            next_obs: same structure (post-transition)
            goal_obs: same structure (relabelled goals)
            action:   actions taken, shape (B, action_dim)
            rest_z:   object z-height at episode start, shape (B,)

        Returns:
            rewards: (B, 1) float32
        """
        eef_prev  = obs[self.eef_pos_key]                    # (B, 3)
        eef       = next_obs[self.eef_pos_key]               # (B, 3)
        obj       = next_obs[self.object_pos_key]            # (B, 3)
        goal      = goal_obs[self.object_pos_key]            # (B, 3)
        gripper   = next_obs[self.gripper_qpos_key]          # (B, 2)

        rewards, info = self._compute_vectorized(
            eef, eef_prev, obj, goal, gripper, rest_z, action
        )

        logger.debug(
            "compute_batch B=%d | mean_reward=%.3f | grasp_conf=%.3f grasp_rate=%.2f | "
            "r1=%.3f Δ=%.3f grip_near=%.3f pre_close=%.3f above=%.3f "
            "r4=%.3f slow=%.3f | d_eef=%.3f d_place=%.3f",
            rewards.shape[0],
            float(rewards.mean()),
            float(info["grasp_confidence"].mean()),
            float(info["grasped"].mean()),
            float(info["r1"].mean()),
            float(info["r1_delta"].mean()),
            float(info["r1_5"].mean()),
            float(info["r_premature_close"].mean()),
            float(info["r_above"].mean()),
            float(info["r4"].mean()),
            float(info["r_slow"].mean()),
            float(info["d_eef"].mean()),
            float(info["d_place"].mean()),
        )

        return rewards  # (B, 1) float32