import numpy as np
from typing import Union, Optional

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader

from os_utils import startswith_list, number_to_order_of_magnitude_string


def get_tensor_size(data: Union[torch.Tensor, dict]) -> float:
    """

    Args:
        data: A tensor or dataloader (dict) from which to get size size_in_mb

    Returns:
        Size of the tensor in memory
    """
    if isinstance(data, dict):
        data = data['data']  # Try to get tensor from typical loader object

    size_in_bytes = data.nelement() * data.element_size()
    size_in_mb = round(size_in_bytes / 1000 / 1000, 1)
    print(f'The size of the tensor is {size_in_mb} MB')
    return size_in_mb


def set_cap_gpu_memory(gpu_memory_target_in_gb: float) -> float:
    """
    Args:
        gpu_memory_target_in_gb: A GPU memory target to reserve. If not available, will warn and scale relative to
        available memory

    Returns:
        Actual memory reserved
    """
    gpu_memory_total = round(torch.cuda.get_device_properties(0).total_memory / 1000 / 1000 / 1000, 1)
    if gpu_memory_target_in_gb > gpu_memory_total:
        print(f'WARNING: Selected/default GPU memory request {gpu_memory_target_in_gb} GB is greater than the GPU '
              f'memory available {gpu_memory_total} GB')
        gpu_memory_target_in_gb = round(gpu_memory_total * 0.8, 1)
        print(f'Setting target to {gpu_memory_target_in_gb} GB')
    return gpu_memory_target_in_gb


def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    # Wrapper to convert tensor to numpy
    if not isinstance(tensor, torch.Tensor):
        print(f'The input is not a tensor. Returning self')
        return tensor
    return tensor.detach().cpu().numpy()


def average_round_metric(metric: list, decimal_points: int = 3) -> float:
    # A wrapper to average a metric calculated over time and return with given precision. Handled natively in Lightning
    # using accumulate()
    return round(sum(metric) / len(metric), decimal_points)


def predict_tokenized_classification(model: torch.nn.Module, test_dataloader: DataLoader,
                                     device: Union[str, torch.device] = 'cuda:0') -> list:
    """
    LLM inference runner to predict classification from a batch with tokens+attention mask
    Args:
        model: Fine-tuned LLM model class
        test_dataloader: A dataloader with tokenized input_ids and attention_mask
        device: Which GPU/cpu to run on

    Returns:
    Predicted classes for each case in test_loader
    """
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


def freeze_layers(layers_keep_training: list, model: torch.nn.Module) -> torch.nn.Module:
    params_to_train = list()
    for param_name, param in model.named_parameters():
        if startswith_list(param_name, layers_keep_training):
            param.requires_grad = True
            params_to_train.append(param_name)
        else:
            param.requires_grad = False
    params_missing_from_net = [name for name in layers_keep_training if
                               (name + '.bias' not in params_to_train) or (name + '.weight' not in params_to_train)]
    if params_missing_from_net:
        raise ValueError(f"Not all specified parameters were present in the network {params_missing_from_net}")
    return model


def unfreeze_layers(model: torch.nn.Module, layers_to_unfreeze: Union[list, str] = 'all') -> torch.nn.Module:
    for param_name, param in model.named_parameters():
        if layers_to_unfreeze == 'all' or startswith_list(param_name, layers_to_unfreeze):
            param.requires_grad = True
        else:
            param.requires_grad = False
    return model


def get_frozen_layers(model: torch.nn.Module, print_results=False) -> dict:
    layers = {'trainable': list(), 'frozen': list()}
    for param_name, param in model.named_parameters():
        if param.requires_grad:
            layers['trainable'].append(param_name)
        else:
            layers['frozen'].append(param_name)

    if print_results:
        print(f"Frozen layers: {layers['frozen']}")
        print(f"Trainable layers: {layers['trainable']}")
    return layers


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


def get_model_param_num(model: torch.nn.Module) -> tuple[str, str]:
    param_number = 0
    param_number_trainable = 0
    for param in model.parameters():
        param_number += param.numel()
        if param.requires_grad:
            param_number_trainable += param.numel()
    param_number_str = number_to_order_of_magnitude_string(param_number)
    param_trainable_str = number_to_order_of_magnitude_string(param_number_trainable)

    print("Trainable Parameters: " + param_trainable_str)
    print("Total Parameters: " + param_number_str)
    return param_number_str, param_trainable_str


def get_model_size(model: torch.nn.Module) -> tuple[int, int, int]:
    size_all_mb = get_model_size_mb(model)
    param_number, param_number_trainable = get_model_param_num(model)
    return size_all_mb, param_number, param_number_trainable


def clear_layers_replace_dropout_rate(model: torch.nn.Module, layers_to_fix: str,
                                      dropout_rate: Optional[Union[float, list]] = None):
    print(f"DEPRECATION WARNING: If fc/cnn layers will be cleared, it is not a good idea to keep dropout. "
          f"This is not used and is planned to be removed.")
    if isinstance(layers_to_fix, str):
        layers_to_fix = list(layers_to_fix)
    layers_to_clear = [name for name in layers_to_fix if 'dropout' not in name]
    input_features = getattr(model, layers_to_clear[0]).in_features.shape[0]
    clear_layers(model, layers_to_clear)

    if dropout_rate is not None:
        layers_to_update_dropout = [name for name in model.named_parameters() if 'dropout' in name]
        if len(layers_to_update_dropout) != len(dropout_rate):
            raise ValueError(f'Provide dropout rate as a single float, or match the number of dropout layers in model')
        update_dropout_rate(model, layer_names=layers_to_update_dropout, dropout_rate=dropout_rate)
    return input_features


def clear_layers(model: torch.nn.Module, layer_names: Union[str, list]):
    # Make one or more layers not do anything. This is a way to delete layers without having to update forward()
    # To avoid having to overwrite the model forward method, set the layer to a passthrough (doesn't do anything)
    if isinstance(layer_names, str):
        layer_names = list(layer_names)
    for layer_name in layer_names:
        setattr(model, layer_name, Identity())


def update_dropout_rate(model: torch.nn.Module, layer_names: Union[str, list], dropout_rate: float):
    # Update dropout for one or more layers
    if isinstance(layer_names, str):
        layer_names = list(layer_names)
    for layer_name in layer_names:
        setattr(model, layer_name, nn.Dropout(p=dropout_rate))


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


def display_tensor_with_PIL(pixel_values):
    embeddings_np = np.swapaxes(np.swapaxes(tensor_to_numpy(torch.squeeze(pixel_values)), 0, 2), 0, 1)
    embeddings_np = embeddings_np + np.abs(np.min(embeddings_np))  # Zero mean
    embeddings_np = embeddings_np * 255 / np.max(embeddings_np)
    embeddings = Image.fromarray(embeddings_np.astype(np.uint8))
    embeddings.show()
