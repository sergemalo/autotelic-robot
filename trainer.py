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
from PIL import Image

from agents import make_agent
from agents.sac import SACAgent
from encoders import make_encoder
from encoders.base import BaseEncoder
from envs.libero_env import LiberoEnv
from replay_buffer import ReplayBuffer
from rewards.factory import make_reward
from rewards.base import BaseReward
from utils.logging_utils import WandBLogger

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

        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)

        # ---- Components ---------------------------------------------
        self.env = LiberoEnv(cfg)
        self.encoder: BaseEncoder = make_encoder(cfg, self.device)
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

        # Current goal obs — sampled from buffer or set at episode start
        self._goal_obs: Optional[dict] = None

        logger.info("Trainer initialised.")

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def train(self):
        logger.info("Starting training for %d env steps.", self.cfg.training.total_env_steps)

        # Initial obs and encoding
        obs = self.env.reset()
        z = self.encoder.encode(self._single_obs(obs)) # used by latent reward and networks

         # Warm-up: collect random transitions before training starts
        logger.info("Warm-up phase: %d random steps.", self.cfg.agent.warmup_steps)
        self._warmup(obs) 
    
        # Set a goal for the first episode
        self._goal_obs = self._sample_goal_obs() # from a buffer of random goal object positions
        z_goal = self.encoder.encode(self._single_obs(self._goal_obs))
        
        episode_return = 0.0
        episode_steps = 0

        pbar = tqdm(total=self.cfg.training.total_env_steps, desc="Training")
        
        # Main training loop
        while self.total_steps < self.cfg.training.total_env_steps: 
            
            #logger.info(f"Step {self.total_steps} | Episode {self.episode_num} | Episode steps {episode_steps} | Return so far {episode_return:.3f}")

            # ---- Select action ---------------------------------------
            action = self.agent.select_action(z, z_goal, deterministic=False)

            # ---- Step environment ------------------------------------
            next_obs, _libero_reward, done, info = self.env.step(action)

            # ---- Encode next obs -------------------------------------
            z_next = self.encoder.encode(self._single_obs(next_obs))

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
                self.buffer.end_episode()

                logger.info(
                    "Episode %d | steps=%d | return=%.3f | success=%s",
                    self.episode_num,
                    episode_steps,
                    episode_return,
                    self.env.check_success(),
                )

                self.wandb.log_scalar(
                    {
                        "episode_return": episode_return,
                        "episode_length": episode_steps,
                        "success": float(self.env.check_success()),
                        "latent_distance": float(
                            torch.norm(z_next - z_goal).item()
                        ),
                    },
                    step=self.total_steps,
                )

                # Log goal vs achieved images periodically
                self.wandb.log_images(
                    step=self.total_steps,
                    goal_image=self._goal_obs.get(self.cfg.encoder.camera_key),
                    achieved_image=next_obs.get(self.cfg.encoder.camera_key),
                )

                # Reset for next episode
                self.episode_num += 1

                # new goal for next episode 
                self._goal_obs = self._sample_goal_obs()
                z_goal = self.encoder.encode(self._single_obs(self._goal_obs))
                
                # reset starting obs
                obs = self.env.reset()
                z = self.encoder.encode(self._single_obs(obs))

                # reset episode return and steps
                episode_return = 0.0
                episode_steps = 0

            # ---- Periodic eval + checkpoint -------------------------
            if self.total_steps % self.cfg.eval_freq == 0:
                eval_metrics = self.evaluate()
                self.wandb.log_scalar(eval_metrics, self.total_steps)

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
        Run cfg.eval_episodes deterministic episodes and return metrics.

        For each episode, saves:
          - <eval_dir>/step_<N>/ep_<E>_goal.png   — the goal image
          - <eval_dir>/step_<N>/ep_<E>_rollout.mp4 — the full episode video
        """
        logger.info("Evaluating for %d episodes...", self.cfg.eval_episodes)

        # Create output directory for this eval checkpoint
        eval_dir = os.path.join(
            self.cfg.checkpoint_dir, "eval", f"step_{self.total_steps}"
        )
        os.makedirs(eval_dir, exist_ok=True)
        logger.info("Saving eval visuals to: %s", eval_dir)

        successes, returns, distances = [], [], []

        for ep in range(self.cfg.eval_episodes):
            obs = self.env.reset()
            goal_obs = self._sample_goal_obs()
            ep_return = 0.0
            done = False
            ep_step = 0
            max_steps = self.cfg.env.episode_length
            frames: List[np.ndarray] = []

            # ---- Save goal image ------------------------------------
            goal_img = goal_obs[self.cfg.encoder.camera_key]  # (H, W, 3) uint8
            goal_path = os.path.join(eval_dir, f"ep_{ep:03d}_goal.png")
            Image.fromarray(goal_img).save(goal_path)
            logger.debug("Goal image saved: %s", goal_path)

            # ---- Roll out episode -----------------------------------
            while not done and ep_step < max_steps:
                # Capture frame before stepping (shows state at this step)
                frames.append(obs[self.cfg.encoder.camera_key].copy())

                z = self.encoder.encode(self._single_obs(obs))
                z_goal = self.encoder.encode(self._single_obs(goal_obs))
                action = self.agent.select_action(z, z_goal, deterministic=True)
                next_obs, _, done, _ = self.env.step(action)

                r = self.reward_fn.compute(
                    obs=obs,
                    next_obs=next_obs,
                    goal_obs=goal_obs,
                    z=z,
                    z_next=self.encoder.encode(self._single_obs(next_obs)),
                    z_goal=z_goal,
                )
                ep_return += r
                ep_step += 1
                obs = next_obs

            # Capture the final frame
            frames.append(obs[self.cfg.encoder.camera_key].copy())

            # ---- Save episode video ---------------------------------
            video_path = os.path.join(eval_dir, f"ep_{ep:03d}_rollout.mp4")
            self._save_video(frames, video_path)

            # ---- Metrics --------------------------------------------
            z_final = self.encoder.encode(self._single_obs(obs))
            z_goal_t = self.encoder.encode(self._single_obs(goal_obs))
            dist = float(torch.norm(z_final - z_goal_t).item())
            success = self.env.check_success()

            successes.append(float(success))
            returns.append(ep_return)
            distances.append(dist)

            logger.info(
                "  ep %d/%d | steps=%d | return=%.3f | dist=%.4f | success=%s",
                ep + 1, self.cfg.eval_episodes,
                ep_step, ep_return, dist, success,
            )

            # ---- Log to WandB ---------------------------------------
            self.wandb.log_images(
                step=self.total_steps,
                goal_image=goal_img,
                achieved_image=obs[self.cfg.encoder.camera_key],
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

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _update(self) -> dict:
        """Sample from buffer and perform one SAC update."""
        z, actions, next_obs, z_next, rewards, dones, z_goal, goal_obs_batch, relabeled_mask = \
            self.buffer.sample(
                batch_size=self.cfg.agent.batch_size,
                device=self.device,
            )

         # Only recompute rewards for relabeled transitions
        if relabeled_mask.any():
            if self.cfg.reward.name == "privileged":
                relabeled_rewards = self._recompute_privileged_rewards(
                    {k: v[relabeled_mask] for k, v in next_obs.items()},  
                    {k: v[relabeled_mask] for k, v in goal_obs_batch.items()},
                    self.device
             )
                rewards[relabeled_mask] = relabeled_rewards
                
            elif self.cfg.reward.name == "latent":
                relabeled_rewards = -torch.norm(
                    z_next[relabeled_mask] - z_goal[relabeled_mask], 
                    dim=-1, keepdim=True
                ) * self.cfg.reward.reward_scale
                rewards[relabeled_mask] = relabeled_rewards


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
        dists = torch.norm(pos_next - pos_goal, dim=-1, keepdim=True)

        rewards = -dists * self.cfg.reward.reward_scale

        return torch.FloatTensor(rewards).to(device)

    def _warmup(self, obs: dict):
        """Collect random transitions to seed the replay buffer."""
        step_in_ep = 0

        for _ in tqdm(range(self.cfg.agent.warmup_steps)):
            action = np.random.uniform(-1, 1, size=self.env.action_dim)
            next_obs, _, done, _ = self.env.step(action)

            # Dummy zero reward during warmup — buffer is just being seeded
            z = self.encoder.encode(self._single_obs(obs))
            z_next = self.encoder.encode(self._single_obs(next_obs))
            
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
        Sample a goal obs from the replay buffer.
        Falls back to a random obs if buffer is empty.
        """
        if len(self.buffer) == 0:
            return self.env.obs or self.env.reset()
        idx = np.random.randint(0, len(self.buffer))
        return {k: self.buffer._next_obs[k][idx] for k in self.buffer._next_obs}

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
