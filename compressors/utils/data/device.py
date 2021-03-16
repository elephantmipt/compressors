"""
From https://github.com/catalyst-team/catalyst/blob/master/catalyst/utils/torch.py
"""
import numpy as np

import torch
from torch import nn


def any2device(value, device):
    """
    Move tensor, list of tensors, list of list of tensors,
    dict of tensors, tuple of tensors to target device.
    Args:
        value: Object to be moved
        device: target device ids
    Returns:
        Same structure as value, but all tensors and np.arrays moved to device
    """
    if isinstance(value, dict):
        return {k: any2device(v, device) for k, v in value.items()}
    elif isinstance(value, (tuple, list)):
        return [any2device(v, device) for v in value]
    elif torch.is_tensor(value):
        return value.to(device, non_blocking=True)
    elif isinstance(value, (np.ndarray, np.void)) and value.dtype.fields is not None:
        return {k: any2device(value[k], device) for k in value.dtype.fields.keys()}
    elif isinstance(value, np.ndarray):
        return torch.tensor(value, device=device)
    elif isinstance(value, nn.Module):
        return value.to(device)
    return value
