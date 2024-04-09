import numpy as np
from typing import Union

import torch
from torch.utils.data import DataLoader

from os_utils import endswith_list


def get_tensor_size(data: Union[torch.Tensor, dict]) -> float:
    if isinstance(data, dict):
        data = data['data']  # Try to get tensor from typical loader object

    size_in_bytes = data.nelement() * data.element_size()
    size_in_mb = round(size_in_bytes / 1000 / 1000, 1)
    print(f'The size of the tensor is {size_in_mb} MB')
    return size_in_mb


def set_cap_gpu_memory(gpu_memory_target_in_gb: float) -> float:
    gpu_memory_total = round(torch.cuda.get_device_properties(0).total_memory / 1000 / 1000 / 1000, 1)
    if gpu_memory_target_in_gb > gpu_memory_total:
        print(f'WARNING: Selected/default GPU memory request {gpu_memory_target_in_gb} GB is greater than the GPU '
              f'memory available {gpu_memory_total} GB')
        gpu_memory_target_in_gb = round(gpu_memory_total * 0.8, 1)
        print(f'Setting target to {gpu_memory_target_in_gb} GB')
    return gpu_memory_target_in_gb


def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    if not isinstance(tensor, torch.Tensor):
        print(f'The input is not a tensor. Returning self')
        return tensor
    return tensor.detach().cpu().numpy()


def average_round_metric(metric: list, decimal_points: int = 3) -> float:
    return round(sum(metric) / len(metric), decimal_points)


def predict_tokenized_classification(model: torch.nn.Module, test_dataloader: DataLoader,
                                     device: Union[str, torch.device] = 'cuda:0') -> list:
    model.to(device)
    model.eval()
    preds = list()
    for idx, batch in enumerate(test_dataloader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        output = model(input_ids, attention_mask=attention_mask)
        preds_batch = torch.argmax(output['logits'], axis=1)
        preds += list(tensor_to_numpy(preds_batch))

    return preds


def freeze_layers(layers_to_freeze: list, model: torch.nn.Module) -> torch.nn.Module:
    for param_name, param in model.named_parameters():
        if endswith_list(param_name, layers_to_freeze):
            param.requires_grad = True
        else:
            print('Kept parameter trainable: ' + param_name)
            param.requires_grad = False
    return model


def unfreeze_layers(model: torch.nn.Module, layers_to_unfreeze: Union[list, str] = 'all') -> torch.nn.Module:
    for param_name, param in model.named_parameters():
        if layers_to_unfreeze == 'all' or endswith_list(param_name, layers_to_unfreeze):
            param.requires_grad = True
        else:
            print('Kept parameter trainable: ' + param_name)
            param.requires_grad = False
    return model


def get_frozen_layers(model: torch.nn.Module) -> tuple[list[str], list[str]]:
    trainable = list()
    frozen = list()
    for param_name, param in model.named_parameters():
        if param.requires_grad:
            print(f"{param_name}: Trainable")
            trainable.append(param_name)
        else:
            print(f"{param_name}: Frozen")
            frozen.append(param_name)
    return trainable, frozen


def get_model_size_mb(model: torch.nn.Module) -> int:
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = round((param_size + buffer_size) / 1024 ** 2)
    print(f"Model size: {size_all_mb} MB")
    return size_all_mb


def get_model_param_num(model: torch.nn.Module) -> tuple[int, int]:
    param_number = 0
    param_number_trainable = 0
    for param in model.parameters():
        param_number += param.numel()
        if param.requires_grad:
            param_number_trainable += param.numel()
    param_number_million = round(param_number / 1e6)
    param_trainable_million = round(param_number / 1e6)
    print(f"Parameters: {param_number_million} million")
    print(f"Trainable Parameters: {param_trainable_million} million")
    return param_number, param_number_trainable


def get_model_size(model: torch.nn.Module) -> tuple[int, int, int]:
    size_all_mb = get_model_size_mb(model)
    param_number, param_number_trainable = get_model_param_num(model)
    return size_all_mb, param_number, param_number_trainable
