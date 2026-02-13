# Autotelic Robot
IFT-6163 Project


## Python virtual environment setup
```
python3.10 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## LIBERO
```
git clone https://github.com/montrealrobotics/LIBERO
cd LIBERO/
touch libero/__init__.py
touch libero/libero/__init__.py
pip install -r requirements.txt 
pip install -e .
```

## Setup checks
### Pytorch
```
python3.10 check_pytorch.py
```

### LIBERO 