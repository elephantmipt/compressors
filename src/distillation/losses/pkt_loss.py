# Code from author: https://github.com/passalis/probabilistic_kt
from typing import Tuple
import torch

from torch import FloatTensor


def cosine_similarity_loss(
    s_hidden_states: Tuple[FloatTensor], t_hidden_states: Tuple[FloatTensor], eps: float = 1e-7
) -> FloatTensor:
    """Loss between distributions over features similarity with cosine similarity kernel.

    Args:
        s_hidden_states (FloatTensor): student hiddens
        t_hidden_states (FloatTensor): teacher hiddens
        eps (float, optional): small value. Defaults to 1e-7.

    Returns:
        FloatTensor: loss
    """
    # Normalize each vector by its norm
    output_net_norm = torch.sqrt(torch.sum(s_hidden_states ** 2, dim=1, keepdim=True))
    s_hidden_states = s_hidden_states / (output_net_norm + eps)
    s_hidden_states[s_hidden_states != s_hidden_states] = 0

    target_net_norm = torch.sqrt(torch.sum(t_hidden_states ** 2, dim=1, keepdim=True))
    t_hidden_states = t_hidden_states / (target_net_norm + eps)
    t_hidden_states[t_hidden_states != t_hidden_states] = 0

    # Calculate the cosine similarity
    model_similarity = torch.mm(s_hidden_states, s_hidden_states.transpose(0, 1))
    target_similarity = torch.mm(t_hidden_states, t_hidden_states.transpose(0, 1))

    # Scale cosine similarity to 0..1
    model_similarity = (model_similarity + 1.0) / 2.0
    target_similarity = (target_similarity + 1.0) / 2.0

    # Transform them into probabilities
    model_similarity = model_similarity / torch.sum(
        model_similarity, dim=1, keepdim=True
    )
    target_similarity = target_similarity / torch.sum(
        target_similarity, dim=1, keepdim=True
    )

    # Calculate the KL-divergence
    loss = torch.mean(
        target_similarity
        * torch.log((target_similarity + eps) / (model_similarity + eps))
    )

    return loss
