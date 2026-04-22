"""
Main training loop for RIG with SAC.

Entry point: train.py calls trainer.train(cfg)
"""
import logging
import os
from typing import List, Optional
from tqdm import tqdm

import imageio
import numpy as np
import torch
from omegaconf import DictConfig
from PIL import Image, ImageDraw, ImageFont

from agents.factory import make_agent
from agents.sac import SACAgent
from encoders import make_encoder
from encoders.base import BaseEncoder
from envs.libero_env import LiberoEnv, get_object_pos
from envs.goals_dataset import GoalSample, GoalsDataset
from replay_buffer import ReplayBuffer
from rewards.factory import make_reward
from rewards.base import BaseReward
from utils.logging_utils import WandBLogger
from utils.seed_ctrl import set_global_seed

logger = logging.getLogger(__name__)


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
        self.env = LiberoEnv(cfg)
        self.encoder: BaseEncoder = make_encoder(cfg, self.device)
        self.reward_fn: BaseReward = make_reward(cfg)
        self.buffer = ReplayBuffer(
            capacity=cfg.replay_buffer.capacity,
            goal_sampling_strategy=cfg.replay_buffer.goal_sampling_strategy,
            future_fraction=cfg.replay_buffer.future_fraction,
        )
        self.agent: SACAgent = make_agent(cfg, self.device)
        self.wandb = WandBLogger(cfg)

        # ---- State --------------------------------------------------
        self.total_steps = 0
        self.episode_num = 0

        # ---- Goal management -----------------------------------------
        # Geneate Goal Dataset
        self.goal_ds = GoalsDataset(cfg, self.env)
        self.goal_ds.generate()

        # Current goal obs — sampled from buffer or set at episode start
        self._goal_obs: Optional[dict] = None
        self._goal: Optional[GoalSample] = None

        logger.info("Trainer initialised.")

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def train(self):
        logger.info("Starting training for %d env steps.", self.cfg.training.total_env_steps)

        # Initial obs and encoding
        obs = self.env.reset()

         # Warm-up: collect random transitions before training starts
        logger.info("Warm-up phase: %d random steps.", self.cfg.agent.warmup_steps)
        self._warmup(obs) 
    
        # Set a goal for the first episode
        self._goal = self.goal_ds.sample_goal()
        self._goal_obs = self._goal.obs
        self._goal.save_image_to_file(os.path.join(self.cfg.output_dir, "goal_image_0.png"))

        #goal_coordinates = get_object_pos(self._goal_obs, self.env.object_name)

        obs = self.env.reset(goal_coordinates = self._goal.position)
        rest_z = float(obs[self.cfg.reward.object_pos_key][2])

        z = self.encoder.encode(self._single_obs(obs))
        z_goal = self.encoder.encode(self._single_obs(self._goal_obs))
        episode_return = 0.0
        episode_steps = 0

        pbar = tqdm(total=self.cfg.training.total_env_steps, desc="Training")
        
        # Main training loop
        while self.total_steps < self.cfg.training.total_env_steps: 
            
            #logger.info(f"Step {self.total_steps} | Episode {self.episode_num} | Episode steps {episode_steps} | Return so far {episode_return:.3f}")

            # ---- Select action ---------------------------------------
            #action = self.agent.select_action(z, z_goal, deterministic=False)
            action = self.agent.select_action(obs, self._goal_obs, deterministic=False)

            # ---- Step environment ------------------------------------
            next_obs, _libero_reward, done, info = self.env.step(action)
            logger.debug(f"Env step | next_obs: {next_obs} | info: {info}")

            # ---- Encode next obs -------------------------------------
            z_next = self.encoder.encode(self._single_obs(next_obs))

            # ---- Compute reward --------------------------------------
            reward = self.reward_fn.compute(
                obs=obs,
                next_obs=next_obs,
                goal_obs=self._goal_obs,
                action=action,
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
                rest_z=rest_z,          # Store rest_z
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
            rest_z = float(obs[self.cfg.reward.object_pos_key][2])

            # ---- SAC update -----------------------------------------
            
            #logger.info("SAC update")

            for _ in range(self.cfg.agent.updates_per_step):
                if len(self.buffer) >= self.cfg.agent.batch_size:
                    metrics = self._update()
                    self.wandb.log_scalar(metrics, self.total_steps)

            #logger.info("SAC update DONE")

            # ---- Episode end ----------------------------------------
            if done:
                self.buffer.end_episode()

                logger.info(
                    "Episode %d | steps=%d | return=%.3f | success=%s",
                    self.episode_num,
                    episode_steps,
                    episode_return,
                    self.env.check_custom_success(obs),
                )

                self.wandb.log_scalar(
                    {
                        "episode_return": episode_return,
                        "episode_length": episode_steps,
                        "success": float(self.env.check_custom_success(obs)),
                        "latent_distance": torch.linalg.vector_norm(z_next - z_goal).item(),
                    },
                    step=self.total_steps,
                )

                # Log goal vs achieved images periodically
                self.wandb.log_images(
                    step=self.total_steps,
                    goal_image=self._goal.image[::-1],
                    achieved_image=next_obs.get(self.cfg.encoder.camera_key)[::-1],
                )

                # Reset for next episode
                self.episode_num += 1

                self._goal = self.goal_ds.sample_goal()
                # new goal for next episode 
                self._goal_obs = self._goal.obs
                z_goal = self.encoder.encode(self._single_obs(self._goal_obs))
                
                # reset starting obs

                obs = self.env.reset(goal_coordinates = self._goal.position)
                self.reward_fn.reset()  # Reset reward function state for new episode
                z = self.encoder.encode(self._single_obs(obs))

                # reset episode return and steps
                episode_return = 0.0
                episode_steps = 0

            # ---- Periodic eval + checkpoint -------------------------
            if self.total_steps % self.cfg.eval.eval_freq == 0:
                eval_metrics = self.evaluate()
                self.wandb.log_scalar(eval_metrics, self.total_steps)

                # Restore training state: eval borrows the env and leaves
                # it in an undefined state. Reset everything so the next
                # training step starts from a clean episode.
                self._goal = self.goal_ds.sample_goal()
                self._goal_obs = self._goal.obs
                obs = self.env.reset(goal_coordinates=self._goal.position)
                self.reward_fn.reset()  # Reset reward function state for new episode
                z = self.encoder.encode(self._single_obs(obs))
                episode_return = 0.0
                episode_steps = 0

            if self.total_steps % self.cfg.checkpoint_freq == 0:
                ckpt_path = os.path.join(
                    self.cfg.checkpoint_dir,
                    f"sac_step_{self.total_steps}.pt",
                )
                self.agent.save(ckpt_path)

        # Final checkpoint
        self.agent.save(os.path.join(self.cfg.checkpoint_dir, "sac_final.pt"))
        self.wandb.finish()
        logger.info("Training complete.")

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(self) -> dict:
        """
        Run cfg.eval.eval_episodes deterministic episodes and return metrics.

        For each episode, saves:
          - <eval_dir>/step_<N>/ep_<E>_goal.png   — the goal image
          - <eval_dir>/step_<N>/ep_<E>_rollout.mp4 — the full episode video
        """
        logger.info("Evaluating for %d episodes...", self.cfg.eval.eval_episodes)

        # Create output directory for this eval checkpoint
        eval_dir = os.path.join(
            self.cfg.output_dir, "eval", f"step_{self.total_steps}"
        )
        os.makedirs(eval_dir, exist_ok=True)
        logger.info("Saving eval visuals to: %s", eval_dir)

        successes, returns, distances = [], [], []

        for ep in range(self.cfg.eval.eval_episodes):
            goal = self.goal_ds.sample_goal()
            obs = self.env.reset(goal_coordinates=goal.position)
            self.reward_fn.reset()  # Reset reward function state for new episode
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

                z = self.encoder.encode(self._single_obs(obs))
                z_goal = self.encoder.encode(self._single_obs(goal_obs))
                #action = self.agent.select_action(z, z_goal, deterministic=True)
                action = self.agent.select_action(obs, goal_obs, deterministic=True)
                next_obs, _, done, _ = self.env.step(action)

                r = self.reward_fn.compute(
                    obs=obs,
                    next_obs=next_obs,
                    goal_obs=goal_obs,
                    action=action,
                    z=z,
                    z_next=self.encoder.encode(self._single_obs(next_obs)),
                    z_goal=z_goal,
                )
                ep_return += r
                ep_step += 1

                # Capture frame after reward is known so overlay values are current
                raw_frame = obs[self.cfg.encoder.camera_key][::-1].copy()
                frames.append(self._annotate_frame(
                    raw_frame, r,
                    eef_pos=obs["robot0_eef_pos"],
                    obj_pos=obs[self.cfg.reward.object_pos_key],
                    goal_pos=goal.position,
                ))

                obs = next_obs
                pbar.update(1)

            pbar.close()


            # Capture the final frame (terminal state — reward shown as 0.0)
            raw_frame = obs[self.cfg.encoder.camera_key][::-1].copy()
            frames.append(self._annotate_frame(
                raw_frame, 0.0,
                eef_pos=obs["robot0_eef_pos"],
                obj_pos=obs[self.cfg.reward.object_pos_key],
                goal_pos=goal.position,
            ))

            # ---- Save episode video ---------------------------------
            video_path = os.path.join(eval_dir, f"ep_{ep:03d}_rollout.mp4")
            self._save_video(frames, video_path)

            # ---- Metrics --------------------------------------------
            z_final = self.encoder.encode(self._single_obs(obs))
            z_goal_t = self.encoder.encode(self._single_obs(goal_obs))
            dist = float(torch.linalg.vector_norm(z_final - z_goal_t).item())
            success = self.env.check_custom_success(obs)

            successes.append(float(success))
            returns.append(ep_return)
            distances.append(dist)

            logger.info(
                "  ep %d/%d | steps=%d | return=%.3f | dist=%.4f | success=%s",
                ep + 1, self.cfg.eval.eval_episodes,
                ep_step, ep_return, dist, success,
            )

            # ---- Log to WandB ---------------------------------------
            self.wandb.log_images(
                step=self.total_steps,
                goal_image=goal.image[::-1],
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
    ) -> np.ndarray:
        """
        Overlay diagnostic text on a (H, W, 3) uint8 frame.

        Draws three lines in the bottom-right corner:
          Reward:   <value>   (blue)
          EEF-OBJ:  <dist>    (green)
          OBJ-GOAL: <dist>    (red)

        Args:
            frame:     uint8 numpy array (H, W, 3)
            reward:    scalar reward for the last action
            eef_pos:   (3,) end-effector position in metres
            obj_pos:   (3,) object position in metres
            goal_pos:  (3,) goal position in metres
            font_size: approximate font size in pixels

        Returns:
            Annotated uint8 numpy array (H, W, 3).
        """
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

        # Measure the widest line to anchor text to the right edge
        padding = 4
        line_spacing = 2
        w, h = img.size

        # Compute line heights via getbbox for accurate placement
        sample_bbox = font.getbbox("Ag")  # (left, top, right, bottom)
        line_h = sample_bbox[3] - sample_bbox[1] + line_spacing

        total_h = line_h * len(lines) + padding
        y = h - total_h  # start y for the first line

        for text, color in lines:
            bbox = font.getbbox(text)
            text_w = bbox[2] - bbox[0]
            x = w - text_w - padding
            # Thin dark shadow for legibility on any background
            draw.text((x + 1, y + 1), text, font=font, fill=(0, 0, 0))
            draw.text((x, y), text, font=font, fill=color)
            y += line_h

        return np.array(img)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _update(self) -> dict:
        """Sample from buffer and perform one SAC update."""
        obs, z, actions, next_obs, z_next, rewards, dones, z_goal, goal_obs_batch, use_relabeled, rest_z_batch = \
            self.buffer.sample(
                batch_size=self.cfg.agent.batch_size,
                device=self.device,
            )
        use_relabeled_gpu = torch.BoolTensor(use_relabeled).to(self.device)


         # Only recompute rewards for relabeled transitions
        if use_relabeled.any():
            if self.cfg.reward.name == "privileged":

                # Shortcut: re-compute them all.
                rewards = self.reward_fn.compute_batch(obs, next_obs, goal_obs_batch, actions, rest_z_batch)
                rewards = torch.FloatTensor(rewards).to(self.device)
                
            elif self.cfg.reward.name == "latent":
                relabeled_rewards = -torch.norm(
                    z_next[use_relabeled_gpu] - z_goal[use_relabeled_gpu], 
                    dim=-1, keepdim=True
                ) * self.cfg.reward.reward_scale
                rewards[use_relabeled_gpu] = relabeled_rewards


        actions_t = torch.FloatTensor(actions).to(self.device)
        metrics = self.agent.update(obs, actions_t, next_obs, rewards, dones, goal_obs_batch)
        metrics["rest_z_mean"] = float(rest_z_batch.mean())
        metrics["rest_z_std"] = float(rest_z_batch.std())        
        return metrics


    def _warmup(self, obs: dict):
        """Collect transitions to seed the replay buffer.

        Option A: reward is stored as 0.0 here but the training loop always
                recomputes privileged reward after sampling, so the stored
                value is never used for critic updates.

        Option B: instead of purely random actions, we bias the first 3 dims
                (EEF delta) toward the object with added noise, and randomise
                the gripper dimension independently.  This ensures a
                meaningful fraction of warmup transitions involve the arm
                near the object, giving the critic real signal on the
                approach and grip_near components from the very first update.
        """
        step_in_ep = 0

        for _ in tqdm(range(self.cfg.agent.warmup_steps)):

            eef_pos_key = "robot0_eef_pos"

            # ── Option B: biased action ───────────────────────────────────
            eef_pos = obs[eef_pos_key]          # (3,)
            obj_pos = obs[self.cfg.reward.object_pos_key]       # (3,)
            rest_z = float(obs[self.cfg.reward.object_pos_key][2])

            delta = obj_pos - eef_pos
            norm  = np.linalg.norm(delta) + 1e-6
            direction = delta / norm                            # unit vector toward object

            # Approach direction + Gaussian noise on EEF translation dims
            translation = (
                self.cfg.agent.warmup_bias_weight * direction
                + (1.0 - self.cfg.agent.warmup_bias_weight) * np.random.randn(3)
            )
            translation = np.clip(translation, -1.0, 1.0)
 
            # Gripper (dim 3) stays fully random — random open/close is intentional
            gripper = np.random.uniform(-1.0, 0.0)
 
            action = np.concatenate([translation, [gripper]])  # (4,)

            next_obs, _, done, _ = self.env.step(action)

            z      = self.encoder.encode(self._single_obs(obs))
            z_next = self.encoder.encode(self._single_obs(next_obs))

            self.buffer.push(
                obs=obs,
                z=z,
                action=action,
                next_obs=next_obs,
                next_z=z_next,
                reward=0.0,   # Option A: training loop always recomputes; this is never used
                done=done,
                step_in_ep=step_in_ep,
                rest_z=rest_z,          # Store rest_z
                z_goal=None,
                goal_obs=None,
                is_warmup=True,
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
        return self.goal_ds.sample_goal().obs

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
