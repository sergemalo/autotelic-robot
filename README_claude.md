# RIG — Modern PyTorch Reimplementation

Reinforcement Learning with Imagined Goals (RIG) on LIBERO,
with a pretrained vision encoder (R3M) replacing the original β-VAE,
SAC instead of TD3, and Hydra + WandB for configuration and tracking.

---

## Project Structure

```
.
├── configs/
│   ├── config.yaml          # top-level defaults
│   ├── encoder/             # r3m.yaml, vae.yaml
│   ├── reward/              # privileged.yaml, latent.yaml
│   ├── agent/               # sac.yaml
│   ├── replay_buffer/       # default.yaml
│   ├── env/                 # libero.yaml
│   └── wandb/               # default.yaml
├── encoders/                # R3MEncoder, VAEEncoder, factory
├── rewards/                 # PrivilegedReward, LatentReward, factory
├── agents/                  # SACAgent, BCTrainer, networks
├── envs/                    # LiberoEnv wrapper
├── utils/                   # logging, WandBLogger
├── replay_buffer.py         # ReplayBuffer with goal relabeling
├── trainer.py               # main training loop
├── train.py                 # entry point
└── eval.py                  # evaluation / demo script
```

---

## Installation

```bash
# 1. Install LIBERO (montrealrobotics fork)
git clone https://github.com/montrealrobotics/LIBERO
cd LIBERO && pip install -e . && cd ..

# 2. Install R3M
pip install r3m

# 3. Install remaining deps
pip install -r requirements.txt
```

---

## Usage

### Training

```bash
# R3M encoder + privileged reward (recommended first step)
python train.py reward.object_pos_key=akita_black_bowl_1_pos

# VAE encoder + latent reward
python train.py encoder=vae reward=latent

# Different task
python train.py env.task_suite=libero_object env.task_id=2 \
                reward.object_pos_key=<object_key>

# Load BC pretrained actor
python train.py agent.pretrained_bc_path=checkpoints/bc_actor.pt \
                reward.object_pos_key=akita_black_bowl_1_pos

# Disable WandB
python train.py wandb.enabled=false reward.object_pos_key=akita_black_bowl_1_pos
```

### Evaluation / Demo

```bash
python eval.py checkpoint=checkpoints/sac_final.pt \
               reward.object_pos_key=akita_black_bowl_1_pos
```

---

## Key Design Decisions

| Component       | Choice            | Rationale                                      |
|-----------------|-------------------|------------------------------------------------|
| Encoder         | R3M (frozen)      | Trained for manipulation temporal progress     |
| RL algorithm    | SAC               | More stable than TD3, better exploration       |
| Reward (step 1) | Privileged        | Direct object position from obs dict           |
| Reward (step 2) | Latent distance   | Pure image-based, RIG-style                    |
| Goal sampling   | Mixed (buffer)    | 50% future + 50% random buffer transitions     |
| Policy/Q input  | [z, z_goal]       | Full R3M latent, no projection head            |
| Image resolution| 224×224           | Native R3M resolution                          |

---

## Flexibility Axes

All three main design choices are swappable via command line:

```
encoder=r3m      ↔  encoder=vae
reward=privileged ↔  reward=latent
agent.pretrained_bc_path=null  ↔  agent.pretrained_bc_path=checkpoints/bc.pt
```
