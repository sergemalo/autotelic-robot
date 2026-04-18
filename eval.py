"""
Evaluation and demo script.

Loads a checkpoint, samples a goal from the replay buffer or a
specified goal image, runs the policy, and logs a video to WandB.

Usage:

    # Evaluate a saved checkpoint
    python eval.py checkpoint=checkpoints/sac_final.pt \\
                   reward.object_pos_key=akita_black_bowl_1_pos

    # Run a single demo episode (renders to WandB)
    python eval.py checkpoint=checkpoints/sac_final.pt \\
                   eval_episodes=1 \\
                   reward.object_pos_key=akita_black_bowl_1_pos
"""
import logging
from typing import List

import hydra
import numpy as np
import torch
from omegaconf import DictConfig

from utils.logging_utils import setup_logging, WandBLogger

logger = logging.getLogger(__name__)


@hydra.main(config_path="configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    setup_logging(cfg)
    logger.info("Starting eval/demo.")

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    # ---- Load components -------------------------------------------
    from encoders import make_encoder
    from rewards.factory import make_reward
    from agents.sac import SACAgent
    from agents import make_agent
    from envs.libero_env import LiberoEnv

    encoder = make_encoder(cfg, device)
    reward_fn = make_reward(cfg)
    env = LiberoEnv(cfg)
    agent = make_agent(cfg, encoder.latent_dim, device)

    # ---- Load checkpoint -------------------------------------------
    ckpt_path = cfg.get("checkpoint", None)
    if ckpt_path is None:
        raise ValueError(
            "No checkpoint specified. "
            "Pass checkpoint=<path> on the command line."
        )
    agent.load(ckpt_path)
    logger.info("Checkpoint loaded: %s", ckpt_path)

    # ---- Optionally load replay buffer for goal sampling -----------
    # If a buffer snapshot is available, load it; otherwise use env resets
    buffer_path = cfg.get("buffer_path", None)
    goal_obs_pool = None
    if buffer_path is not None:
        import pickle
        with open(buffer_path, "rb") as f:
            saved_buffer = pickle.load(f)
        goal_obs_pool = saved_buffer  # dict of arrays
        logger.info("Goal obs pool loaded from buffer: %s", buffer_path)

    # ---- WandB -----------------------------------------------------
    wandb_logger = WandBLogger(cfg)

    # ---- Run eval episodes -----------------------------------------
    n_episodes = cfg.eval_episodes
    successes, returns, distances = [], [], []
    frames: List[np.ndarray] = []

    for ep in range(n_episodes):
        obs = env.reset()

        # Sample goal
        if goal_obs_pool is not None:
            idx = np.random.randint(0, len(next(iter(goal_obs_pool.values()))))
            goal_obs = {k: goal_obs_pool[k][idx] for k in goal_obs_pool}
        else:
            # Use the initial obs as a trivial goal (replace with your own logic)
            goal_obs = obs
            logger.warning(
                "No buffer provided for goal sampling — using initial obs as goal."
            )

        ep_return = 0.0
        done = False
        ep_frames = []
        ep_step = 0
        max_steps = cfg.env.episode_length

        while not done and ep_step < max_steps:
            z = encoder.encode(_single_obs(obs))
            z_goal = encoder.encode(_single_obs(goal_obs))
            action = agent.select_action(z, z_goal, deterministic=True)
            next_obs, _, done, _ = env.step(action)

            r = reward_fn.compute(
                obs=obs,
                next_obs=next_obs,
                goal_obs=goal_obs,
                z=z,
                z_next=encoder.encode(_single_obs(next_obs)),
                z_goal=z_goal,
            )
            ep_return += r
            ep_frames.append(obs[cfg.encoder.camera_key][::-1].copy())
            obs = next_obs
            ep_step += 1

        z_final = encoder.encode(_single_obs(obs))
        z_goal_t = encoder.encode(_single_obs(goal_obs))
        dist = float(torch.norm(z_final - z_goal_t).item())
        success = env.check_success()

        successes.append(float(success))
        returns.append(ep_return)
        distances.append(dist)
        frames.append(ep_frames[::-1].copy())

        logger.info(
            "Episode %d/%d | return=%.3f | dist=%.4f | success=%s",
            ep + 1, n_episodes, ep_return, dist, success,
        )

        # Log goal vs final frame
        wandb_logger.log_images(
            step=ep,
            goal_image=goal_obs[cfg.encoder.camera_key],
            achieved_image=obs[cfg.encoder.camera_key],
        )

    # ---- Summary ---------------------------------------------------
    logger.info(
        "Eval complete | success=%.2f | mean_return=%.3f | mean_dist=%.4f",
        np.mean(successes),
        np.mean(returns),
        np.mean(distances),
    )

    wandb_logger.log_scalar(
        {
            "eval/success_rate": float(np.mean(successes)),
            "eval/mean_return": float(np.mean(returns)),
            "eval/mean_latent_distance": float(np.mean(distances)),
        },
        step=0,
    )

    # Log video of first episode to WandB
    if cfg.wandb.enabled and frames:
        import wandb
        video = np.stack(frames[0], axis=0)        # (T, H, W, 3)
        video = video.transpose(0, 3, 1, 2)        # (T, 3, H, W)
        wandb.log({"demo_video": wandb.Video(video, fps=10, format="mp4")})

    wandb_logger.finish()
    env.close()


def _single_obs(obs: dict) -> dict:
    return {
        k: v[np.newaxis] if v.ndim == len(v.shape) else v
        for k, v in obs.items()
    }


if __name__ == "__main__":
    main()
