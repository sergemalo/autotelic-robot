"""
Behavioural cloning (BC) pretraining.

Loads LIBERO demonstration data and trains the SAC actor via
supervised imitation before RL begins.

Usage:
    from agents.bc import BCTrainer
    bc = BCTrainer(actor, encoder, cfg, device)
    bc.train(demo_dataset)
    bc.save(path)
"""
import logging
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)


class DemoDataset(Dataset):
    """
    Simple dataset wrapping a list of (obs_dict, action) pairs
    pre-encoded into latent vectors.
    """

    def __init__(
        self,
        z_obs: np.ndarray,          # (N, latent_dim)
        z_goals: np.ndarray,        # (N, latent_dim)
        actions: np.ndarray,        # (N, action_dim)
    ):
        self.z_obs = torch.FloatTensor(z_obs)
        self.z_goals = torch.FloatTensor(z_goals)
        self.actions = torch.FloatTensor(actions)

    def __len__(self):
        return len(self.z_obs)

    def __getitem__(self, idx):
        return self.z_obs[idx], self.z_goals[idx], self.actions[idx]


class BCTrainer:
    """
    Trains the SAC actor network with behavioural cloning.

    The actor is trained to minimise MSE between its mean action
    (tanh(mu)) and the demonstrated action.
    """

    def __init__(
        self,
        actor,           # SACActorNetwork
        encoder,         # BaseEncoder
        lr: float = 1e-3,
        batch_size: int = 256,
        device: torch.device = torch.device("cpu"),
    ):
        self.actor = actor
        self.encoder = encoder
        self.batch_size = batch_size
        self.device = device
        self.optimizer = torch.optim.Adam(actor.parameters(), lr=lr)
        logger.info("BCTrainer initialised: lr=%.4f, batch_size=%d", lr, batch_size)

    def encode_demos(
        self,
        demo_obs: List[dict],
        demo_goal_obs: List[dict],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Encode raw obs dicts into latent vectors.

        Args:
            demo_obs:      list of obs dicts (one per timestep)
            demo_goal_obs: list of goal obs dicts (one per timestep)

        Returns:
            z_obs, z_goals as numpy arrays
        """
        logger.info("Encoding %d demonstration observations...", len(demo_obs))
        z_list, zg_list = [], []

        # Encode in batches to avoid OOM
        batch = 64
        for i in range(0, len(demo_obs), batch):
            obs_b = self._stack_obs(demo_obs[i: i + batch])
            goal_b = self._stack_obs(demo_goal_obs[i: i + batch])
            with torch.no_grad():
                z_list.append(self.encoder.encode(obs_b).cpu().numpy())
                zg_list.append(self.encoder.encode(goal_b).cpu().numpy())

        return np.concatenate(z_list, axis=0), np.concatenate(zg_list, axis=0)

    @staticmethod
    def _stack_obs(obs_list: List[dict]) -> dict:
        """Stack a list of single obs dicts into a batched obs dict."""
        keys = obs_list[0].keys()
        return {k: np.stack([o[k] for o in obs_list], axis=0) for k in keys}

    def train(
        self,
        z_obs: np.ndarray,
        z_goals: np.ndarray,
        actions: np.ndarray,
        n_epochs: int = 50,
    ) -> List[float]:
        """
        Run BC training.

        Args:
            z_obs:    (N, latent_dim)
            z_goals:  (N, latent_dim)
            actions:  (N, action_dim)
            n_epochs: number of passes over the dataset

        Returns:
            list of per-epoch mean losses
        """
        dataset = DemoDataset(z_obs, z_goals, actions)
        loader = DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True, drop_last=True
        )
        epoch_losses = []

        for epoch in range(n_epochs):
            losses = []
            for z, zg, a_demo in loader:
                z = z.to(self.device)
                zg = zg.to(self.device)
                a_demo = a_demo.to(self.device)

                # Use deterministic (mean) action for BC
                mu, _ = self.actor(z, zg)
                a_pred = torch.tanh(mu)

                loss = F.mse_loss(a_pred, a_demo)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                losses.append(loss.item())

            mean_loss = float(np.mean(losses))
            epoch_losses.append(mean_loss)
            logger.info("BC epoch %d/%d — loss: %.5f", epoch + 1, n_epochs, mean_loss)

        return epoch_losses

    def save(self, path: str):
        """Save only the actor state dict (compatible with SAC._load_bc_weights)."""
        torch.save({"actor": self.actor.state_dict()}, path)
        logger.info("BC actor saved: %s", path)
