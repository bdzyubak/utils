import torch
from typing import Union


def get_tensor_size(data: Union[torch.Tensor, dict]):
    if isinstance(data, dict):
        data = data['data']  # Try to get tensorf from typical loader object

    size_in_bytes = data.nelement() * data.element_size()
    size_in_mb = round(size_in_bytes / 1000 / 1000, 1)
    print(f'The size of the tensor is {size_in_mb} MB')
    return size_in_mb
