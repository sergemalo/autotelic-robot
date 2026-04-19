import os
import random
import numpy as np
import torch

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"

def set_global_seed(seed: int, deterministic: bool = True):
    """
    Set seed for reproducibility across:
    - Python
    - NumPy
    - PyTorch (CPU + CUDA)
    - cuDNN
    - Hash seed
    
    Args:
        seed (int): The seed value to use.
        deterministic (bool): If True, forces deterministic algorithms
                              (may reduce performance).
    """

    # 1 Python random
    random.seed(seed)

    # 2 NumPy
    np.random.seed(seed)

    # 3 Python hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)

    # 4 PyTorch CPU
    torch.manual_seed(seed)

    # 5 PyTorch CUDA
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # 6 cuDNN settings
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

