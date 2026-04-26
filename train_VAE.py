import logging
import dill
import hydra
from omegaconf import DictConfig

from utils.logging_utils import setup_logging
from rlkit.pythonplusplus import identity


import numpy as np
import torch
from rlkit.torch import pytorch_util as ptu
from rlkit.torch.vae.conv_vae import ConvVAE, imsize48_default_architecture, imsize84_default_architecture
from rlkit.torch.vae.vae_trainer import ConvVAETrainer
from utils.logging_utils import WandBLogger

import os
from sklearn.model_selection import train_test_split
from envs.static_arm_goals_dataset import StaticArmGoalsDataset

logger = logging.getLogger(__name__)


# Extraction et prétraitement des images (resize, flatten)
def preprocess_goals(goals, imsize):
    images = []
    for goal in goals:
        img = goal.image  # (H, W, 3) uint8
        # Resize si besoin
        from PIL import Image
        img_pil = Image.fromarray(img)
        img_pil = img_pil.resize((imsize, imsize), Image.BILINEAR)
        img_np = np.array(img_pil, dtype=np.uint8)
        # (H, W, C) -> (C, H, W)
        img_np = np.transpose(img_np, (2, 0, 1))
        images.append(img_np)
    images = np.stack(images, axis=0)
    images = images.reshape(images.shape[0], -1)
    return images

@hydra.main(config_path="configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    setup_logging(cfg)
    logger.info("Starting VAE training.")
    logger.debug("Full config:\n%s", cfg)

    # Utilisation de StaticArmGoalsDataset pour charger les images
    dataset = StaticArmGoalsDataset(cfg)
    dataset.load()
    train_goals = dataset.get_split("train")
    eval_goals = dataset.get_split("eval")

    train_data = preprocess_goals(train_goals, cfg.encoder.imsize)
    test_data = preprocess_goals(eval_goals, cfg.encoder.imsize)

    ptu.set_gpu_mode(True)

    if cfg.encoder.decoder_activation == 'sigmoid':
        decoder_activation = torch.nn.Sigmoid()
    else:
        decoder_activation = identity

    model = ConvVAE(
        representation_size=cfg.encoder.representation_size,
        input_channels=cfg.encoder.channels,
        architecture=imsize48_default_architecture,
        imsize=cfg.encoder.imsize,
        decoder_output_activation=decoder_activation
    )

    print(ptu.device)
    model.to(ptu.device)

    trainer = ConvVAETrainer(
        train_dataset=train_data,
        test_dataset=test_data,
        model=model,
        batch_size=cfg.encoder.batch_size,
        beta=cfg.encoder.beta,
        lr=cfg.encoder.lr,
    )

    cfg.wandb.log_freq = 1
    wandb = WandBLogger(cfg)

    #index = 0
    for epoch in range(cfg.encoder.num_epochs):
      trainer.train_epoch(epoch)
      trainer.test_epoch(epoch)
      diagnostics = trainer.get_diagnostics()
      train_loss = diagnostics.get('train/loss', None)
      test_loss = diagnostics.get('test/loss', None)
      print(f"Epoch {epoch}: Train Loss = {train_loss}, Test Loss = {test_loss}")
      # Log to wandb
      # wandb.log({"epoch": epoch, "train/loss": train_loss, "test/loss": test_loss})
      wandb.log_scalar(
                  {
                    "VAE_train_loss": train_loss,
                    "VAE_test_loss": test_loss
                  }, step=epoch
              )
      #index += 1
      if epoch % 100 == 0 or epoch == cfg.encoder.num_epochs - 1:
          os.makedirs('vae_models', exist_ok=True)
          torch.save(model, f'vae_models/model_epoch_{epoch}.pth', pickle_module=dill)


if __name__ == "__main__":
    main()