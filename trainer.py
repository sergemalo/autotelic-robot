"""
Main training loop for RIG with SAC.

Entry point: train.py calls trainer.train(cfg)
"""
import logging
import os
from typing import List, Optional
from tqdm import tqdm
import wandb

import imageio
import numpy as np
import torch
from omegaconf import DictConfig
from PIL import Image, ImageDraw, ImageFont

from agents import make_agent
from agents.sac import SACAgent
from encoders import make_encoder
from encoders.base import BaseEncoder
from envs.libero_env import LiberoObjectEnv, LiberoArmEnv, get_object_pos
from envs.goals_dataset import GoalSample, ObjectGoalsDataset, ArmGoalsDataset
from envs.static_arm_goals_dataset import StaticArmGoalsDataset
from replay_buffer import ReplayBuffer
from rewards.factory import make_reward
from rewards.base import BaseReward
from utils.logging_utils import WandBLogger
from utils.seed_ctrl import set_global_seed

logger = logging.getLogger(__name__)


_ENV_CLASSES = {
    "libero_object": LiberoObjectEnv,
    "libero_arm": LiberoArmEnv,
}
_GOALS_CLASSES = {
    "libero_object": ObjectGoalsDataset,
    "libero_arm":    ArmGoalsDataset,
}


class Trainer:
    """
    Orchestrates:
      - environment interaction
      - encoding observations
      - reward computation
      - replay buffer management + goal relabeling
      - SAC updates
      - checkpointing
      - WandB logging
    """

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
        logger.info("Using device: %s", self.device)

        logger.info("Setting global seed: %d", cfg.seed)
        set_global_seed(cfg.seed)

        # ---- Components ---------------------------------------------
        self.encoder: BaseEncoder = make_encoder(cfg, self.device)

        self.env = _ENV_CLASSES[cfg.env.name](cfg, encoder=self.encoder)

        #if(cfg.encoder.name == "vae"):
        #    cfg.reward = 'latent'

        self.reward_fn: BaseReward = make_reward(cfg)
        self.buffer = ReplayBuffer(
            capacity=cfg.replay_buffer.capacity,
            goal_sampling_strategy=cfg.replay_buffer.goal_sampling_strategy,
            future_fraction=cfg.replay_buffer.future_fraction,
        )
        self.agent: SACAgent = make_agent(cfg, self.encoder.latent_dim, self.device)
        self.wandb = WandBLogger(cfg)

        # ---- State --------------------------------------------------
        self.total_steps = 0
        self.episode_num = 0

        # ---- Goal management -----------------------------------------
        # Generate Arm or Object goal dataset
        #self.goal_ds = GoalsDataset(cfg, self.env)
        #self.goal_ds = _GOALS_CLASSES[cfg.env.name](cfg, self.env)
        #self.goal_ds.generate()

         # Load static goal dataset
        self.goal_ds = StaticArmGoalsDataset(cfg)
        self.goal_ds.load()


        # Current goal obs — sampled from buffer or set at episode start
        self._goal_obs: Optional[dict] = None
        self._goal: Optional[GoalSample] = None

        logger.info("Trainer initialised.")

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def train(self):
        logger.info("Starting training for %d env steps.", self.cfg.training.total_env_steps)


        # ------------------------------- Warm-up phase --------------------------------------
        # Initial obs and encoding
        obs = self.env.reset()

         # Warm-up: collect random transitions before training starts
        logger.info("Warm-up phase: %d random steps.", self.cfg.agent.warmup_steps)
        self._warmup(obs) 
    
        
        # -------------------------- Main training loop --------------------------------------
        # Set a goal for the first episode
        self._goal, self._module_idx  = self.goal_ds.sample_goal(split="train")  
        
        
        self._goal_obs = self._goal.obs # None if level 2 or 3

        
        z_goal = self._get_z_goal(self._goal)

        obs = self.reset_env(goal = self._goal, z_goal = z_goal)  # reset environment with the first goal


        # Encode obs
        #z = self.encoder.encode(self._single_obs(obs))
        z = self.encoder.encode(self._single_obs(obs)[self.cfg.encoder.camera_key])
        

        episode_return = 0.0
        episode_steps = 0

        pbar = tqdm(total=self.cfg.training.total_env_steps, desc="Training")
        
        # Main training loop
        while self.total_steps < self.cfg.training.total_env_steps: 
            
            #logger.info(f"Step {self.total_steps} | Episode {self.episode_num} | Episode steps {episode_steps} | Return so far {episode_return:.3f}")

            # ---- Select action ---------------------------------------
            action = self.agent.select_action(z, z_goal, deterministic=False)

            # ---- Step environment ------------------------------------
            next_obs, z_next, _libero_reward, done, info = self.env.step(action)
            success = info['success_info'].success
            pos_err = info['success_info'].pos_err
            ori_err = info['success_info'].ori_err

            # ---- Compute reward --------------------------------------
            reward = self.reward_fn.compute(
                obs=obs,
                next_obs=next_obs,
                goal_obs=self._goal_obs,
                z=z,
                z_next=z_next,
                z_goal=z_goal,
            )

            # ---- Store transition -----------------------------------
            self.buffer.push(
                obs=obs,
                z=z,
                action=action,
                next_obs=next_obs,
                next_z=z_next,
                reward=reward,
                done=done,
                step_in_ep=episode_steps,
                z_goal=z_goal,  # Store original goal
                goal_obs=self._goal_obs,  # Store original goal obs
                is_warmup=False,  # Not warmup

            )

            # Update episode return and steps
            episode_return += reward
            episode_steps += 1
            self.total_steps += 1
            pbar.update(1)

            # update obs and z for next step
            obs = next_obs
            z = z_next

            # ---- SAC update -----------------------------------------
            
            #logger.info("SAC update")

            for _ in range(self.cfg.agent.updates_per_step):
                if len(self.buffer) >= self.cfg.agent.batch_size:
                    metrics = self._update()
                    self.wandb.log_scalar(metrics, self.total_steps)

            #logger.info("SAC update DONE")

            # ---- Episode end ----------------------------------------
            if done:
                if self.cfg.level == 3:
                    self.goal_ds._update_intrinsic_motivation(self._module_idx, success)
                else:
                    self.goal_ds._update_intrinsic_motivation(self._module_idx, success)
                    
                self.buffer.end_episode()

                logger.info(
                    "Episode %d | steps=%d | return=%.3f | success=%s | pos_err=%.3fm | ori_err=%.3frad",
                    self.episode_num,
                    episode_steps,
                    episode_return,
                    success,
                    pos_err,
                    ori_err
                )

                self.wandb.log_scalar(
                    {
                        "episode_return": episode_return,
                        "episode_length": episode_steps,
                        "success": (float(success)),
                        "latent_distance": torch.linalg.vector_norm(z_next - z_goal).item(),
                        "last_pos_error": pos_err,
                        "last_ori_error": ori_err
                    },
                    step=self.total_steps,
                )

                # Log goal vs achieved images periodically
                if self.cfg.level != 3:
                    self.wandb.log_images(
                        step=self.total_steps,
                        goal_image=self._goal.image[::-1],
                        achieved_image=next_obs[self.cfg.encoder.camera_key][::-1],
                    )
                # Reset for next episode
                self.episode_num += 1

                self._goal, self._module_idx  = self.goal_ds.sample_goal(split="train")
                # new goal for next episode 
                self._goal_obs = self._goal.obs
                
                z_goal = self._get_z_goal(self._goal)
                
                # reset starting obs
                obs = self.reset_env(goal = self._goal, z_goal = z_goal)
                z = self.encoder.encode(self._single_obs(obs)[self.cfg.encoder.camera_key])

                # reset episode return and steps
                episode_return = 0.0
                episode_steps = 0

            # ---- Periodic eval + checkpoint -------------------------
            if self.total_steps % self.cfg.training.valid.eval_freq == 0:
                eval_metrics = self.evaluate(execution_type="train")
                self.wandb.log_scalar(eval_metrics, self.total_steps)
                # Restore training state: eval borrows the env and leaves
                # it in an undefined state. Reset everything so the next
                # training step starts from a clean episode.
                self._goal, self._module_idx = self.goal_ds.sample_goal(split="train")
                self._goal_obs = self._goal.obs
                obs = self.reset_env(goal = self._goal, z_goal = z_goal)
                z = self.encoder.encode(self._single_obs(obs)[self.cfg.encoder.camera_key])
                episode_return = 0.0
                episode_steps = 0

            if self.total_steps % self.cfg.training.checkpoint_freq == 0:
                ckpt_path = os.path.join(
                    self.cfg.training.checkpoint_dir,
                    f"sac_step_{self.total_steps}.pt",
                )
                self.agent.save(ckpt_path)

        # Final checkpoint
        self.agent.save(os.path.join(self.cfg.training.checkpoint_dir, "sac_final.pt"))
        self.wandb.finish()
        logger.info("Training complete.")

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(self, execution_type: str = "train") -> dict:
        """
        Run deterministic episodes and return metrics.

        Args:
            execution_type: "train" or "eval".
                  - "train": called during train.py, validates using train goals
                  - "eval":  called during eval.py, evaluates ALL goals from eval split

        For each episode, saves:
          - <eval_dir>/step_<N>/ep_<E>_goal.png   — the goal image
          - <eval_dir>/step_<N>/ep_<E>_rollout.mp4 — the full episode video
        """
        if execution_type == "train":
            num_episodes = self.cfg.training.valid.eval_episodes
            goals_to_eval = None  # sample_goal will be used
            logger.info("Evaluating for %d episodes from train split...", num_episodes)
        elif execution_type == "eval":
            goals_to_eval = self.goal_ds._eval_goals
            num_episodes = len(goals_to_eval)
            logger.info("Evaluating all %d goals from eval split...", num_episodes)
        else:
            raise ValueError(
                f"execution_type must be 'train' or 'eval', got '{execution_type}'"
            )

        # Create output directory for this eval checkpoint
        if execution_type == "eval":
            eval_dir = os.path.join(self.cfg.output_dir, "eval", "full_eval_set")
        else:
            eval_dir = os.path.join(
                self.cfg.output_dir, "eval", f"step_{self.total_steps}"
            )
        os.makedirs(eval_dir, exist_ok=True)
        logger.info("Saving eval visuals to: %s", eval_dir)

        successes, returns, distances = [], [], []

        for ep in range(num_episodes):
            # ---- Get goal for this episode --------------------------
            if execution_type == "train":
                goal, _ = self.goal_ds.sample_goal(split="eval")
            else:  # execution_type == "eval"
                goal = goals_to_eval[ep]
            
            z_goal = self._get_z_goal(goal, goal_split_origin='eval')
            
            obs = self.reset_env(goal = goal, z_goal = z_goal)
            goal_obs = goal.obs
            ep_return = 0.0
            done = False
            ep_step = 0
            max_steps = self.cfg.env.episode_length
            frames: List[np.ndarray] = []

            # ---- Save goal image ------------------------------------
            goal.save_image_to_file(os.path.join(eval_dir, f"ep_{ep:03d}_goal.png"))

            # ---- Roll out episode -----------------------------------
            pbar = tqdm(total=max_steps, desc="Evaluating")
            while not done and ep_step < max_steps:
                # Capture frame before stepping (shows state at this step)

                z = self.encoder.encode(self._single_obs(obs)[self.cfg.encoder.camera_key])
                action = self.agent.select_action(z, z_goal, deterministic=True)
                next_obs, z_next, _, done, info = self.env.step(action)
                success = info['success_info'].success
                pos_err = info['success_info'].pos_err
                ori_err = info['success_info'].ori_err

                r = self.reward_fn.compute(
                    obs=obs,
                    next_obs=next_obs,
                    goal_obs=goal_obs,
                    z=z,
                    z_next=z_next,
                    z_goal=z_goal,
                )
                ep_return += r
                ep_step += 1

                # Capture frame after reward is known so overlay values are current
                raw_frame = obs[self.cfg.encoder.camera_key][::-1].copy()
                if self.cfg.env.name == "libero_object":
                    obj_pos = obs[self.cfg.reward.object_pos_key]
                else:                    
                    obj_pos = None 
                frames.append(self._annotate_frame(
                    raw_frame, r,
                    eef_pos=obs["robot0_eef_pos"],
                    obj_pos=obj_pos,
                    goal_pos=goal.position,
                    info_success=info['success_info'],
                ))

                obs = next_obs
                pbar.update(1)

            pbar.close()


            # Capture the final frame (terminal state)
            raw_frame = obs[self.cfg.encoder.camera_key][::-1].copy()
            if self.cfg.env.name == "libero_object":
                obj_pos = obs[self.cfg.reward.object_pos_key]
            else:                    
                obj_pos = None 
            last_frame = self._annotate_frame(
                raw_frame, r,
                eef_pos=obs["robot0_eef_pos"],
                obj_pos=obj_pos,
                goal_pos=goal.position,
                info_success=info['success_info'],
            )
            for _ in range(10):  # Show final frame for a few frames at the end of the video
                frames.append(last_frame)

            # ---- Save episode video ---------------------------------
            video_path = os.path.join(eval_dir, f"ep_{ep:03d}_rollout.mp4")
            self._save_video(frames, video_path)

            # ---- Metrics --------------------------------------------
            z_final = self.encoder.encode(self._single_obs(obs)[self.cfg.encoder.camera_key])
            z_goal_t = z_goal.squeeze(0)  # (latent_dim,) — remove batch dim for distance computation
            dist = float(torch.linalg.vector_norm(z_final - z_goal_t).item())

            successes.append(float(success))
            returns.append(ep_return)
            distances.append(dist)

            logger.info(
                "  ep %d/%d | steps=%d | return=%.3f | dist=%.4f | success=%s | pos_err=%.3fm | ori_err=%.3frad",
                ep + 1, num_episodes,
                ep_step, ep_return, dist, success, pos_err, ori_err,
            )

            # ---- Log to WandB ---------------------------------------
            if execution_type == "eval":
                self.wandb.log_scalar(
                    {
                        "eval/episode_return": ep_return,
                        "eval/episode_length": ep_step,
                        "eval/success": float(success),
                        "eval/latent_distance": dist,
                        "eval/last_pos_error": pos_err,
                        "eval/last_ori_error": ori_err,
                    },
                    step=ep,
                )            
            log_step = self.total_steps if execution_type == "train" else ep
            self.wandb.log_images(
                step=log_step,
                goal_image=goal.image,
                achieved_image=obs[self.cfg.encoder.camera_key][::-1],
            )

        metrics = {
            "eval/success_rate": float(np.mean(successes)),
            "eval/mean_return": float(np.mean(returns)),
            "eval/mean_latent_distance": float(np.mean(distances)),
        }
        logger.info(
            "Eval complete | success=%.2f | return=%.3f | dist=%.4f",
            metrics["eval/success_rate"],
            metrics["eval/mean_return"],
            metrics["eval/mean_latent_distance"],
        )
        if execution_type == "eval":
            wandb.run.summary.update(metrics)            
        return metrics

    # ------------------------------------------------------------------
    # Visual output helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _save_video(frames: List[np.ndarray], path: str, fps: int = 20):
        """
        Write a list of uint8 (H, W, 3) frames to an MP4 file.

        Args:
            frames: list of numpy arrays, each (H, W, 3) uint8
            path:   output file path (must end in .mp4)
            fps:    playback frame rate
        """
        if not frames:
            logger.warning("No frames to save for video: %s", path)
            return
        writer = imageio.get_writer(path, fps=fps, codec="libx264", quality=7)
        for frame in frames:
            writer.append_data(frame)
        writer.close()
        logger.debug("Video saved: %s (%d frames @ %d fps)", path, len(frames), fps)


    @staticmethod
    def _annotate_frame(
        frame: np.ndarray,
        reward: float,
        eef_pos: np.ndarray,
        obj_pos: np.ndarray,
        goal_pos: np.ndarray,
        font_size: int = 12,
        info_success = None,
    ) -> np.ndarray:
        """
        Overlay diagnostic text on a (H, W, 3) uint8 frame.

        Bottom-right corner (all frames):
          Reward:   <value>   (blue)
          EEF-OBJ:  <dist>    (green)
          OBJ-GOAL: <dist>    (red)

        Top-right corner (final frame only, when success is not None):
          "SUCCESS!" in green  or  "FAIL" in red

        Args:
            frame:     uint8 numpy array (H, W, 3)
            reward:    scalar reward for the last action
            eef_pos:   (3,) end-effector position in metres
            obj_pos:   (3,) object position in metres
            goal_pos:  (3,) goal position in metres
            font_size: approximate font size in pixels
            success:   if not None, draw outcome label in the top-right corner

        Returns:
            Annotated uint8 numpy array (H, W, 3).
        """
        if obj_pos is None:
            eef_goal_dist  = float(np.linalg.norm(eef_pos - goal_pos))
            lines = [
                (f"Reward:   {reward:+.2f}",       (100, 160, 255)),  # blue
                (f"Pos err: {info_success.pos_err:.2f}m", (255,  80,  80)),  # red
                (f"Ori err: {info_success.ori_err:.2f}rad", (255,  80,  80)),  # red
                
            ]
        else:
            eef_obj_dist  = float(np.linalg.norm(eef_pos - obj_pos))
            obj_goal_dist = float(np.linalg.norm(obj_pos - goal_pos))

            lines = [
                (f"Reward:   {reward:+.2f}",       (100, 160, 255)),  # blue
                (f"EEF-OBJ:  {eef_obj_dist:.2f}m", ( 80, 220,  80)),  # green
                (f"OBJ-GOAL: {obj_goal_dist:.2f}m", (255,  80,  80)),  # red
            ]

        img = Image.fromarray(frame)
        draw = ImageDraw.Draw(img)

        # Use a truetype font if available, otherwise fall back to default bitmap
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", font_size)
        except OSError:
            font = ImageFont.load_default()

        padding = 4
        line_spacing = 2
        w, h = img.size

        sample_bbox = font.getbbox("Ag")  # (left, top, right, bottom)
        line_h = sample_bbox[3] - sample_bbox[1] + line_spacing

        # ---- Bottom-right: metric lines --------------------------------
        total_h = line_h * len(lines) + padding
        y = h - total_h
        for text, color in lines:
            bbox = font.getbbox(text)
            text_w = bbox[2] - bbox[0]
            x = w - text_w - padding
            draw.text((x + 1, y + 1), text, font=font, fill=(0, 0, 0))
            draw.text((x, y), text, font=font, fill=color)
            y += line_h

        # ---- Top-right: success/fail label ----------
        label = "SUCCESS!" if info_success.success else "FAIL"
        color = (80, 220, 80) if info_success.success else (255, 80, 80)
        bbox = font.getbbox(label)
        text_w = bbox[2] - bbox[0]
        x = w - text_w - padding
        y = padding
        draw.text((x + 1, y + 1), label, font=font, fill=(0, 0, 0))
        draw.text((x, y), label, font=font, fill=color)

        return np.array(img)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------


    def reset_env(self, goal = None, z_goal = None):
        if self.cfg.level == 3:
            obs = self.env.reset(latent_goal = goal.latent_representation)
        else:
            obs = self.env.reset(goal_coordinates = goal.position, goal_quat=goal.quat)
        return obs



    def _get_z_goal(self, goal: GoalSample, goal_split_origin = 'train'):        
        if self.cfg.level in (1, 2) or goal_split_origin == 'eval': 
            z_goal = self.encoder.encode(goal.image)
        else:
            z_goal = goal.latent_representation.unsqueeze(0)  # add batch dim

        return z_goal

    def _update(self) -> dict:
        """Sample from buffer and perform one SAC update."""
        z, actions, next_obs, z_next, rewards, dones, z_goal, goal_obs_batch, use_relabeled = \
            self.buffer.sample(
                batch_size=self.cfg.agent.batch_size,
                device=self.device,
            )
        use_relabeled_gpu = torch.BoolTensor(use_relabeled).to(self.device)


        # Only recompute rewards for relabeled transitions
        if use_relabeled.any():
            if self.cfg.reward.name in ("privileged_obj", "privileged_arm"):
                relabeled_rewards = self.reward_fn.compute_batch(
                    next_obs={k: v[use_relabeled] for k, v in next_obs.items()},
                    goal_obs={k: v[use_relabeled] for k, v in goal_obs_batch.items()},
                    device=self.device,
                )
                rewards[use_relabeled_gpu] = relabeled_rewards
                
            elif self.cfg.reward.name == "latent":
                relabeled_rewards = self.reward_fn.compute_batch(
                    z_next[use_relabeled_gpu],
                    z_goal[use_relabeled_gpu],
                )
                rewards[use_relabeled_gpu] = relabeled_rewards


        return self.agent.update(z, actions, z_next, rewards, dones, z_goal)

    def _recompute_privileged_rewards(
        self, next_obs: dict, goal_obs_batch: dict, device: torch.device
    ) -> torch.Tensor:
        """
        Recompute privileged rewards for a batch of relabeled goals.
        Returns a (B, 1) tensor.
        """
        key = self.cfg.reward.object_pos_key

        # Extract positions from the observation dictionaries and convert to numpy for distance computation
        pos_next = next_obs[key]
        pos_goal = goal_obs_batch[key]        

        # Compute Euclidean distance with torch
        #dists = torch.norm(pos_next - pos_goal, dim=-1, keepdim=True)
        dists = np.linalg.norm(
            pos_next - pos_goal, axis=-1, keepdims=True
        ).astype(np.float32)
        rewards = -dists * self.cfg.reward.reward_scale

        return torch.FloatTensor(rewards).to(device)

    def _warmup(self, obs: dict):
        """Collect random transitions to seed the replay buffer."""
        step_in_ep = 0

        for _ in tqdm(range(self.cfg.agent.warmup_steps)):
            action = np.random.uniform(-1, 1, size=self.env.action_dim)
            next_obs, z_next, _, done, _ = self.env.step(action)

            # Dummy zero reward during warmup — buffer is just being seeded
            z = self.encoder.encode(self._single_obs(obs)[self.cfg.encoder.camera_key])
            
            self.buffer.push(
                obs=obs,
                z=z,
                action=action,
                next_obs=next_obs,
                next_z=z_next,
                reward=0.0, # 0 reward during warmup; actual rewards are computed during updates
                done=done,
                step_in_ep=step_in_ep,
                z_goal=None,  # No original goal
                goal_obs=None,
                is_warmup=True,  # Mark as warmup
            )

            step_in_ep += 1

            if done:
                self.buffer.end_episode()
                obs = self.env.reset()
                step_in_ep = 0
            else:
                obs = next_obs

        self.buffer.end_episode()
        obs = self.env.reset()
        logger.info("Warm-up complete. Buffer size: %d", len(self.buffer))

    def _sample_goal_obs(self) -> dict:
        """
        Sample a goal obs from Goal dataset
        """
        self._goal, self._goal_idx = self.goal_ds.sample_goal(split="train")
        return self._goal.obs

    def _random_goal_obs(self, obs: dict) -> dict:
        """Use current obs as a placeholder goal during warmup."""
        return obs

    @staticmethod
    def _single_obs(obs: dict) -> dict:
        """
        Ensure obs values have a batch dimension for the encoder.
        If already batched (ndim matches), leave as-is.
        """
        return {
            k: v[np.newaxis] if v.ndim == len(v.shape) else v
            for k, v in obs.items()
        }