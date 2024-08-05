import torch

print("CUDA Available:", torch.cuda.is_available())  # 检查 CUDA 是否可用
print("CUDA Version:", torch.version.cuda)  # 查看 PyTorch 支持的 CUDA 版本
print("CUDA Device Count:", torch.cuda.device_count())  # 检查可用的 CUDA 设备数量
