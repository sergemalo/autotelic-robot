"""
1. Load the static pre-generated dataset : Call StaticArmGoalsDataset.load() to load the dataset and populate the train / eval subsets.

2. Encode train subset in VAE latent space : Use the VAE encoder to encode the images in the train subset into a latent space representation.

3.1 Run k-means on original train subset to extract centroids for level 2 (use goal.position and goal.quat as features)
3.2 Save the representative centroids for each module in a file

4.1 Run k-means on latent train subset to extract centroids for level 3 (use the latent vectors as features)
4.2 Save the representative centroids and the cluster labels for each module in a file

Outputs:
- modules_l2.npz : Contains `centroids` (shape: [n_modules, 7 ]) and `cluster_labels` (shape: [n_train]) for the original train subset.
- modules_l3.npz : Contains `centroids` (shape: [n_modules, latent_dim]) and `cluster_labels` (shape: [n_train]) for the latent train subset.


"""

import numpy as np
import torch
import hydra
from omegaconf import DictConfig
from sklearn.cluster import KMeans
from envs.static_arm_goals_dataset import StaticArmGoalsDataset
from encoders.vae import VAEEncoder


# -------------------------- level 3 --------------------------
def create_modules_lvl3(cfg, n_modules):

    # Sample 9k training goals from the latent space (Normal distribution) and save them in a file
    latent_dim = 4
    latent_train_subset = np.random.randn(9000, latent_dim).astype(np.float32)

    np.savez("latent_train_subset.npz", latent_train_subset=latent_train_subset)

    # Run k-means on the latent train subset
    kmeans_l3 = KMeans(n_clusters=n_modules, random_state=cfg.seed, n_init="auto")
    cluster_labels_l3 = kmeans_l3.fit_predict(latent_train_subset) # (n_train,)
    centroids_l3 = kmeans_l3.cluster_centers_  # (n_modules, latent_dim)


    np.savez("modules_l3.npz", centroids=centroids_l3, cluster_labels=cluster_labels_l3)




# -------------------------- level 2 --------------------------
def create_modules_lvl2(cfg, n_modules):

  # --------------------- Load the static pre-generated dataset ------------------
    static_dataset = StaticArmGoalsDataset(cfg)
    static_dataset.load()
    og_train_subset = static_dataset.get_split("train")


    # --------------------- Run k-means on original train and save  ------------------

    # Extract features for k-means (concatenate position and quat for each goal, resulting in a 7D feature vector)
    features_l2 = np.stack([np.concatenate([goal.position, goal.quat]) for goal in og_train_subset])  # (n_train, 7)

    # Run k-means on the original train subset
    kmeans_l2 = KMeans(n_clusters=n_modules, random_state=cfg.seed, n_init="auto")
    cluster_labels_l2 = kmeans_l2.fit_predict(features_l2) # (n_train,)
    centroids_l2 = kmeans_l2.cluster_centers_  # (n_modules, 7)

    np.savez("modules_l2.npz", centroids=centroids_l2, cluster_labels=cluster_labels_l2)






# -------------------------- main function --------------------------

@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig):

    n_modules = 20
    #create_modules_lvl2(cfg, n_modules)
    create_modules_lvl3(cfg, n_modules)
    
    


if __name__ == "__main__":
    main()
