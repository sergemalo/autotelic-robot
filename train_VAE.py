import logging

import hydra
from omegaconf import DictConfig

from utils.logging_utils import setup_logging
from rlkit.pythonplusplus import identity

import numpy as np
import torch
from rlkit.torch import pytorch_util as ptu

from rlkit.torch.vae.conv_vae import ConvVAE  # Adaptez le chemin d'import selon votre projet
from rlkit.torch.vae.conv_vae import imsize48_default_architecture
from rlkit.torch.vae.vae_trainer import ConvVAETrainer

logger = logging.getLogger(__name__)

@hydra.main(config_path="configs", config_name="vae", version_base="1.3")
def main(cfg: DictConfig):
    setup_logging(cfg)
    logger.info("Starting VAE training.")
    logger.debug("Full config:\n%s", cfg)

    # GET DATA
    num_train = 1000  # Nombre d'images d'entraînement
    num_test = 200    # Nombre d'images de test
    height = 48       # Hauteur des images (doit correspondre à imsize du VAE)
    width = height        # Largeur des images (doit correspondre à imsize du VAE)
    channels = 3      # Canaux (3 pour RGB, 1 pour niveaux de gris)

    # Générer des images aléatoires (bruit uniforme entre 0 et 255)
    train_data = np.random.randint(0, 256, (num_train, channels * height * width), dtype=np.uint8)
    test_data = np.random.randint(0, 256, (num_test, channels * height * width), dtype=np.uint8)

    train_data = np.array(train_data, dtype=np.uint8)  # Shape: (num_train, height, width, channels)
    test_data = np.array(test_data, dtype=np.uint8)

    ptu.set_gpu_mode(True)  # Utilisez GPU si disponible, sinon False pour CPU

    if cfg.decoder_activation == 'sigmoid':
        decoder_activation = torch.nn.Sigmoid()
    else:
        decoder_activation = identity

    model = ConvVAE(
        representation_size=cfg.representation_size,
        input_channels=channels,  # 3 pour RGB, 1 pour gris
        architecture=imsize48_default_architecture,
        imsize=height,  # Taille des images (doit correspondre à vos données)
        decoder_output_activation=decoder_activation
    )

    print(ptu.device)

    model.to(ptu.device)  # Déplacez sur GPU/CPU

    trainer = ConvVAETrainer(
        train_dataset=train_data,
        test_dataset=test_data,
        model=model,
        batch_size=cfg.batch_size,  # Taille de batch
        beta=cfg.beta,  # Poids de la divergence KL (0 pour auto-encodeur pur)
        lr=cfg.lr,  # Taux d'apprentissage
        # Autres options : log_interval=10, etc.
    )

    for epoch in range(cfg.num_epochs):
        trainer.train_epoch(epoch)  # Entraînement sur une époque
        trainer.test_epoch(epoch)  # Évaluation sur les données de test : genere images debut et resultantes
        print(f"Epoch {epoch}: Loss = {trainer.get_diagnostics()['train/loss']}")
        # Optionnel : sauvegardez le modèle ou des reconstructions
        if epoch % 10 == 0:
            # trainer.dump_samples(epoch)  # Si vous voulez sauvegarder des échantillons
            torch.save(model.state_dict(), f'vae_models/model_epoch_{epoch}.pth')


if __name__ == "__main__":
    main()