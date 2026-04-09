"""
Utility helpers:
  - setup_logging: configure Python logging from Hydra config
  - WandBLogger:   thin wrapper around wandb for experiment tracking
"""
import logging
import os
from typing import Any, Dict, Optional

import numpy as np
from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)


def setup_logging(cfg: DictConfig):
    """
    Configure root logger level from cfg.log_level.
    Hydra already sets up handlers; we just set the level.
    """
    level = getattr(logging, cfg.log_level.upper(), logging.INFO)
    logging.getLogger().setLevel(level)
    # Keep third-party loggers quieter
    for noisy in ("PIL", "matplotlib", "urllib3", "filelock"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    logger.debug("Logging level set to %s", cfg.log_level)


class WandBLogger:
    """
    Wrapper around wandb.

    If cfg.wandb.enabled is False, all calls are no-ops so the
    rest of the code never needs to check.
    """

    def __init__(self, cfg: DictConfig):
        self.enabled = cfg.wandb.enabled
        self.log_freq = cfg.wandb.log_freq
        self.log_image_freq = cfg.wandb.log_image_freq
        self._step = 0

        if not self.enabled:
            logger.info("WandB logging disabled.")
            return

        import wandb

        init_kwargs = dict(
            project=cfg.wandb.project,
            config=OmegaConf.to_container(cfg, resolve=True),
        )
        if cfg.wandb.entity:
            init_kwargs["entity"] = cfg.wandb.entity
        if cfg.wandb.group:
            init_kwargs["group"] = cfg.wandb.group
        if cfg.wandb.tags:
            init_kwargs["tags"] = list(cfg.wandb.tags)

        wandb.init(**init_kwargs)
        logger.info(
            "WandB initialised: project=%s, run=%s",
            cfg.wandb.project,
            wandb.run.name,
        )

    def log_scalar(self, metrics: Dict[str, Any], step: int):
        """Log a dict of scalar metrics."""
        if not self.enabled:
            return
        if step % self.log_freq != 0:
            return
        import wandb
        wandb.log(metrics, step=step)

    def log_images(
        self,
        step: int,
        goal_image: Optional[np.ndarray] = None,
        achieved_image: Optional[np.ndarray] = None,
    ):
        """
        Log goal and achieved images side by side.

        Args:
            step:           current env step
            goal_image:     uint8 (H, W, 3)
            achieved_image: uint8 (H, W, 3)
        """
        if not self.enabled:
            return
        if step % self.log_image_freq != 0:
            return
        import wandb

        log_dict = {}
        if goal_image is not None:
            log_dict["goal_image"] = wandb.Image(goal_image)
        if achieved_image is not None:
            log_dict["achieved_image"] = wandb.Image(achieved_image)
        if log_dict:
            wandb.log(log_dict, step=step)

    def finish(self):
        if not self.enabled:
            return
        import wandb
        wandb.finish()
        logger.info("WandB run finished.")