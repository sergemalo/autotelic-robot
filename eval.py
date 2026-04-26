"""
Evaluation script for full eval set.

Loads a checkpoint and evaluates the policy on all goals in the eval split.
Uses the Trainer's evaluate method to generate metrics and visualizations.

Usage:

    # Evaluate a checkpoint
    python eval.py eval.checkpoint_path=checkpoints/sac_final.pt

    # Override other config
    python eval.py eval.checkpoint_path=checkpoints/sac_final.pt \\
                   reward.object_pos_key=akita_black_bowl_1_pos
"""
import logging

import hydra
from omegaconf import DictConfig

from utils.logging_utils import setup_logging

logger = logging.getLogger(__name__)


@hydra.main(config_path="configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    setup_logging(cfg)
    logger.info("Starting evaluation on full eval set.")
    logger.debug("Full config:\n%s", cfg)

    # ---- Validate checkpoint path ----------------------------------
    checkpoint_path = cfg.eval.checkpoint_path
    if checkpoint_path is None:
        raise ValueError(
            "No checkpoint specified. "
            "Pass eval.checkpoint_path=<path> on the command line."
        )

    # ---- Create trainer --------------------------------------------
    from trainer import Trainer
    trainer = Trainer(cfg)

    # ---- Load checkpoint -------------------------------------------
    trainer.agent.load(checkpoint_path)
    logger.info("Checkpoint loaded: %s", checkpoint_path)

    # ---- Evaluate on all eval goals --------------------------------
    eval_metrics = trainer.evaluate(execution_type="eval")

    # ---- Log results -----------------------------------------------
    logger.info("Evaluation complete!")
    logger.info("Results:")
    for key, val in eval_metrics.items():
        logger.info("  %s: %.4f", key, val)

    trainer.wandb.finish()


if __name__ == "__main__":
    main()
