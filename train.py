"""
Training entry point.

Usage examples:

    # Default: R3M encoder, privileged reward, SAC
    python train.py reward.object_pos_key=akita_black_bowl_1_pos

    # Switch to VAE encoder + latent reward
    python train.py encoder=vae reward=latent

    # Load BC pretrained policy
    python train.py agent.pretrained_bc_path=checkpoints/bc_actor.pt \\
                    reward.object_pos_key=akita_black_bowl_1_pos

    # Different task
    python train.py env.task_suite=libero_object env.task_id=2 \\
                    reward.object_pos_key=<object_key>

    # Disable WandB
    python train.py wandb.enabled=false
"""
import logging

import hydra
from omegaconf import DictConfig

from utils.logging_utils import setup_logging

logger = logging.getLogger(__name__)


@hydra.main(config_path="configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    setup_logging(cfg)
    logger.info("Starting RIG training run.")
    logger.debug("Full config:\n%s", cfg)

    from trainer import Trainer
    trainer = Trainer(cfg)
    trainer.train()


if __name__ == "__main__":
    main()
