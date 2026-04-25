import logging

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
from PIL import Image
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

def load_and_preprocess_images(image_dir, imsize=48):
    image_files = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith('.png')]
    images = []
    for img_path in image_files:
        img = Image.open(img_path).convert('RGB')
        img = img.resize((imsize, imsize), Image.BILINEAR)
        img_np = np.array(img, dtype=np.uint8)
        # (H, W, C) -> (C, H, W)
        img_np = np.transpose(img_np, (2, 0, 1))
        images.append(img_np)
    images = np.stack(images, axis=0)
    # Flatten for VAE: (N, C*H*W)
    images = images.reshape(images.shape[0], -1)
    return images

@hydra.main(config_path="configs", config_name="vae", version_base="1.3")
def main(cfg: DictConfig):
    setup_logging(cfg)
    logger.info("Starting VAE training.")
    logger.debug("Full config:\n%s", cfg)



    # Load images from goals_data directory
    image_dir = os.path.abspath("goals_data")
    imsize = 84  # Target size for VAE
    channels = 3
    all_images = load_and_preprocess_images(image_dir, imsize=imsize)

    # Split into train and test sets (e.g., 80% train, 20% test)
    train_data, test_data = train_test_split(all_images, test_size=0.2, random_state=42)

    ptu.set_gpu_mode(True)

    if cfg.decoder_activation == 'sigmoid':
        decoder_activation = torch.nn.Sigmoid()
    else:
        decoder_activation = identity

    model = ConvVAE(
        representation_size=cfg.representation_size,
        input_channels=channels,
        architecture=imsize84_default_architecture,
        imsize=imsize,
        decoder_output_activation=decoder_activation
    )

    print(ptu.device)
    model.to(ptu.device)

    trainer = ConvVAETrainer(
        train_dataset=train_data,
        test_dataset=test_data,
        model=model,
        batch_size=cfg.batch_size,
        beta=cfg.beta,
        lr=cfg.lr,
    )

    cfg.wandb.log_freq = 1
    wandb = WandBLogger(cfg)

    #index = 0
    for epoch in range(cfg.num_epochs):
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
                    "train_loss": train_loss,
                    "test_loss": test_loss
                  }, step=epoch
              )
      #index += 1
      if epoch % 3 == 0:
          os.makedirs('vae_models', exist_ok=True)
          torch.save(model.state_dict(), f'vae_models/model_epoch_{epoch}.pth')


if __name__ == "__main__":
    main()