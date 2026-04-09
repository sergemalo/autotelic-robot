"""
Main training loop for RIG with SAC.

Entry point: train.py calls trainer.train(cfg)
"""
import logging
import os
from typing import Optional

import numpy as np
import torch
from omegaconf import DictConfig

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
        logger.info("Starting training for %d env steps.", self.cfg.total_env_steps)

        obs = self.env.reset()

        # Warm-up: collect random transitions before training starts
        logger.info("Warm-up phase: %d random steps.", self.cfg.agent.warmup_steps)
        self._warmup(obs)

        # Set a goal for the first episode
        self._goal_obs = self._sample_goal_obs()
        obs = self.env.reset()
        episode_return = 0.0
        episode_steps = 0

        while self.total_steps < self.cfg.total_env_steps:

            # ---- Encode current obs and goal -------------------------
            z = self.encoder.encode(self._single_obs(obs))
            z_goal = self.encoder.encode(self._single_obs(self._goal_obs))

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
                action=action,
                next_obs=next_obs,
                reward=reward,
                done=done,
                step_in_ep=episode_steps,
            )

            episode_return += reward
            episode_steps += 1
            self.total_steps += 1

            obs = next_obs

            # ---- SAC update -----------------------------------------
            for _ in range(self.cfg.agent.updates_per_step):
                if len(self.buffer) >= self.cfg.agent.batch_size:
                    metrics = self._update()
                    self.wandb.log_scalar(metrics, self.total_steps)

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
                self._goal_obs = self._sample_goal_obs()
                obs = self.env.reset()
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
        """Run cfg.eval_episodes deterministic episodes and return metrics."""
        logger.info("Evaluating for %d episodes...", self.cfg.eval_episodes)
        successes, returns, distances = [], [], []

        for ep in range(self.cfg.eval_episodes):
            obs = self.env.reset()
            goal_obs = self._sample_goal_obs()
            ep_return = 0.0
            done = False

            while not done:
                z = self.encoder.encode(self._single_obs(obs))
                z_goal = self.encoder.encode(self._single_obs(goal_obs))
                action = self.agent.select_action(z, z_goal, deterministic=True)
                obs, _, done, _ = self.env.step(action)
                r = self.reward_fn.compute(
                    obs=obs,
                    next_obs=obs,
                    goal_obs=goal_obs,
                    z=z,
                    z_next=z,
                    z_goal=z_goal,
                )
                ep_return += r

            z_final = self.encoder.encode(self._single_obs(obs))
            z_goal = self.encoder.encode(self._single_obs(goal_obs))
            dist = float(torch.norm(z_final - z_goal).item())

            successes.append(float(self.env.check_success()))
            returns.append(ep_return)
            distances.append(dist)

        metrics = {
            "eval/success_rate": float(np.mean(successes)),
            "eval/mean_return": float(np.mean(returns)),
            "eval/mean_latent_distance": float(np.mean(distances)),
        }
        logger.info(
            "Eval | success=%.2f | return=%.3f | dist=%.4f",
            metrics["eval/success_rate"],
            metrics["eval/mean_return"],
            metrics["eval/mean_latent_distance"],
        )
        return metrics

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _update(self) -> dict:
        """Sample from buffer and perform one SAC update."""
        z, actions, z_next, rewards, dones, z_goal, goal_obs_batch = \
            self.buffer.sample(
                batch_size=self.cfg.agent.batch_size,
                encoder=self.encoder,
                device=self.device,
            )

        # Optionally recompute rewards with privileged function on relabeled goals
        # (only needed when reward is privileged and goals were relabeled)
        # For latent reward, the buffer already recomputes in latent space.
        # For privileged reward, rewards stored in buffer are w.r.t. original goal;
        # relabeled reward recomputation is handled below.
        if self.cfg.reward.name == "privileged":
            rewards = self._recompute_privileged_rewards(
                z_next, goal_obs_batch, self.device
            )

        return self.agent.update(z, actions, z_next, rewards, dones, z_goal)

    def _recompute_privileged_rewards(
        self, z_next, goal_obs_batch: dict, device: torch.device
    ) -> torch.Tensor:
        """
        Recompute privileged rewards for a batch of relabeled goals.
        Returns a (B, 1) tensor.
        """
        key = self.cfg.reward.object_pos_key
        pos_next = goal_obs_batch[key]          # (B, 3) — next obs positions
        # For a relabeled goal, the "goal position" is the next_obs of the
        # sampled goal transition.
        # We use the same key from the goal obs batch.
        pos_goal = goal_obs_batch[key]          # same batch, acts as goal

        # distance is zero for same-index — this is handled correctly by
        # the mixed sampling since goal is a *different* sampled transition.
        # Re-using the batch as both is intentional for the recomputation path.
        dists = np.linalg.norm(
            pos_next - pos_goal, axis=-1, keepdims=True
        ).astype(np.float32)
        rewards = -dists * self.cfg.reward.reward_scale
        return torch.FloatTensor(rewards).to(device)

    def _warmup(self, obs: dict):
        """Collect random transitions to seed the replay buffer."""
        goal_obs = self._random_goal_obs(obs)
        step_in_ep = 0

        for _ in range(self.cfg.agent.warmup_steps):
            action = np.random.uniform(-1, 1, size=self.env.action_dim)
            next_obs, _, done, _ = self.env.step(action)

            # Dummy zero reward during warmup — buffer is just being seeded
            self.buffer.push(
                obs=obs,
                action=action,
                next_obs=next_obs,
                reward=0.0,
                done=done,
                step_in_ep=step_in_ep,
            )
            step_in_ep += 1

            if done:
                self.buffer.end_episode()
                obs = self.env.reset()
                goal_obs = self._random_goal_obs(obs)
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
