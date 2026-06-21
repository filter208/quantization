import sys
import torch
import torchvision

print("=== Python 环境信息 ===")
print(f"Python 版本: {sys.version}")

print("\n=== PyTorch & CUDA 库信息 ===")
print(f"PyTorch 版本: {torch.__version__}")
print(f"Torchvision 版本: {torchvision.__version__}")
print(f"CUDA 是否可用: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"PyTorch 编译使用的 CUDA 版本: {torch.version.cuda}")
    print(f"cuDNN 版本: {torch.backends.cudnn.version()}")
    print(f"当前使用的 GPU: {torch.cuda.get_device_name(0)}")
else:
    print("未检测到可用的 GPU。")