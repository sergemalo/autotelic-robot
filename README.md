# Autotelic Robot
IFT-6163 Project

## Linux prerequisites
I strongly suggest we stick to Python 3.10
```
sudo apt-get update
sudo apt-get install -y cmake build-essential
sudo apt install python3.10 python3.10-venv python3.10-dev
```

## How to prepare your venv
I have created a requirements.txt file that integrates all python lib dependencies for Pytorch and LIBERO. You simply need to install them with the following commands:
```
python3.10 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## LIBERO
1. Clone the Serge Malo's fork:
```
git clone https://github.com/sergemalo/LIBERO
```
2. Install it:
```
cd LIBERO/
touch libero/__init__.py
touch libero/libero/__init__.py
pip install -e .
```
3. Download one LIBERO Dataset:
```
python3.10 benchmark_scripts/download_libero_datasets.py --datasets libero_spatial --use-huggingface
```
4. Setup LIBERO Macros:
```
cd ..
python3.10 venv/lib/python3.10/site-packages/robosuite/scripts/setup_macros.py
```


## Setup sanity checks
### Pytorch
```
python3.10 check_pytorch.py
```

### LIBERO 
```
python3.10 check_pytorch.py
```

____
____
## How to re-create requirements.txt
These are just in case we end up with a bad venv and need to recreate it.

```
python3.10 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
python3.10 check_pytorch.py
pip install wandb
pip install git+https://github.com/facebookresearch/r3m.git


git clone https://github.com/sergemalo/LIBERO
cd LIBERO/
touch libero/__init__.py
touch libero/libero/__init__.py
pip install -U pip setuptools wheel
pip install -r requirements.txt 
pip install "hydra-core>=1.2,<1.4" omegaconf
pip install transformers==4.37.2 accelerate sentencepiece
pip install ipykernel jupyter
pip install minigrid
pip install -U --force-reinstall --no-cache-dir numpy matplotlib scipy opencv-python
pip install -e .

python3.10 benchmark_scripts/download_libero_datasets.py --use-huggingface

cd ..
python3.10 venv/lib/python3.10/site-packages/robosuite/scripts/setup_macros.py
python3.10 check_libero.py
```
