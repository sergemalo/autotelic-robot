import sys
import torch

def check_pytorch():
    print(f"Python version: {sys.version}")
    print(f"PyTorch version: {torch.__version__}")

    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"CUDA runtime version: {torch._C._cuda_getCompiledVersion()}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("CUDA is not available")

if __name__ == "__main__":
    check_pytorch()