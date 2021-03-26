from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from numpy import ndarray

import os
from pathlib import Path

from catalyst.utils.checkpoint import load_checkpoint, unpack_checkpoint
from torch.nn import Module


def get_loss_coefs(alpha: float, beta: float = None) -> "ndarray":
    """
    Returns loss weights. Sum of the weights is 1.
    Args:
        alpha: logit for second loss.
        beta:  logit for third loss.
    Returns:
        Array with weights.
    """
    if beta is None:
        return torch.softmax([1, alpha]).numpy()
    return torch.softmax([1, alpha, beta]).numpy()


def load_model_from_path(model: Module, path: Path):
    if os.path.isdir(path):
        load_model_from_path(model, path / "best.pth")
    model_sd = load_checkpoint(path)
    try:
        unpack_checkpoint(model_sd, model=model)
    except KeyError:
        model_sd.load_state_dict(model_sd)


__all__ = ["load_model_from_path", "get_loss_coefs"]
