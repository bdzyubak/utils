import torch
from typing import Union


def get_tensor_size(data: Union[torch.Tensor, dict]):
    if isinstance(data, dict):
        data = data['data']  # Try to get tensorf from typical loader object

    size_in_bytes = data.nelement() * data.element_size()
    size_in_mb = round(size_in_bytes / 1000 / 1000, 1)
    print(f'The size of the tensor is {size_in_mb} MB')
    return size_in_mb


def set_cap_gpu_memory(gpu_memory_target_in_gb):
    gpu_memory_total = round(torch.cuda.get_device_properties(0).total_memory / 1000 / 1000 / 1000, 1)
    if gpu_memory_target_in_gb > gpu_memory_total:
        print(f'WARNING: Selected/default GPU memory request {gpu_memory_target_in_gb} GB is greater than the GPU '
              f'memory available {gpu_memory_total} GB')
        gpu_memory_target_in_gb = round(gpu_memory_total * 0.8, 1)
        print(f'Setting target to {gpu_memory_target_in_gb} GB')
    return gpu_memory_target_in_gb


def tensor_to_numpy(output):
    return output.detach().cpu().numpy()
