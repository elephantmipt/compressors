from typing import Tuple

import torch
from torch import FloatTensor, LongTensor
from torch import nn
from torch.nn import functional as F


def mse_loss(
    s_hidden_states: Tuple[FloatTensor],
    t_hidden_states: Tuple[FloatTensor],
    normalize: bool = False,
) -> FloatTensor:
    """mse loss for hidden states

    Args:
        s_hidden_states (Tuple[FloatTensor]): student hiddens
        t_hidden_states (Tuple[FloatTensor]): teacher hiddens
        normalize (bool, optional): normalize embeddings. Defaults to False.

    Returns:
        FloatTensor: loss
    """

    if normalize:
        loss_fn = lambda s, t: nn.MSELoss(reduction="mean")(
            F.normalize(s), F.normalize(t)
        )
    else:
        loss_fn = nn.MSELoss(reduction="mean")

    return torch.stack(
        [
            loss_fn(s_hiddens, t_hiddens)  # loss for CLS token
            for s_hiddens, t_hiddens in zip(s_hidden_states, t_hidden_states)
        ]
    ).mean()


def mse_loss_mlm(
    s_logits: FloatTensor, t_logits: FloatTensor, attention_mask: LongTensor,
) -> FloatTensor:
    mask = attention_mask.unsqueeze(-1).expand_as(s_logits)
    # (bs, seq_lenth, voc_size)
    s_logits_slct = torch.masked_select(s_logits, mask)
    # (bs * seq_length * voc_size) modulo the 1s in mask
    s_logits_slct = s_logits_slct.view(-1, s_logits.size(-1))
    # (bs * seq_length, voc_size) modulo the 1s in mask
    t_logits_slct = torch.masked_select(t_logits, mask)
    # (bs * seq_length * voc_size) modulo the 1s in mask
    t_logits_slct = t_logits_slct.view(-1, s_logits.size(-1))
    # (bs * seq_length, voc_size) modulo the 1s in mask
    loss_mse = nn.MSELoss(reduction="mean")(s_logits_slct, t_logits_slct)
    return loss_mse


__all__ = ["mse_loss", "mse_loss_mlm"]
