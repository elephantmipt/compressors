from typing import Tuple, Union
import warnings

import torch
from torch import FloatTensor, LongTensor
from torch import nn
from torch.nn import functional as F


class MSEHiddenStatesLoss(nn.Module):
    def __init__(
        self,
        normalize: bool = False,
        need_mapping: bool = False,
        teacher_hidden_state_dim: int = None,
        student_hidden_state_dim: int = None,
        num_layers: int = None,
    ):
        super(MSEHiddenStatesLoss, self).__init__()
        self.loss_fn = nn.MSELoss()
        self.normalize = normalize
        self.need_mapping = need_mapping
        if need_mapping:
            self.teacher_hidden_state_dim = teacher_hidden_state_dim
            self.student_hidden_state_dim = student_hidden_state_dim
            if num_layers is not None:
                self.proj = nn.ModuleList([nn.Linear(
                    teacher_hidden_state_dim,
                    student_hidden_state_dim
                ) for _ in range(num_layers)])
            else:
                self.proj = nn.Linear(
                    teacher_hidden_state_dim,
                    student_hidden_state_dim
                )

    def forward(
        self,
        s_hidden_states: Union[FloatTensor, Tuple[FloatTensor]],
        t_hidden_states: Union[FloatTensor, Tuple[FloatTensor]]
    ) -> FloatTensor:
        if isinstance(s_hidden_states, FloatTensor):
            return self._forward_hidden(
                s_hidden_states=s_hidden_states, t_hidden_states=t_hidden_states
            )
        if len(s_hidden_states) != len(t_hidden_states):
            diff = len(t_hidden_states) - len(s_hidden_states)
            t_hidden_states = tuple([t_hidden_states[i] for i in range(diff, len(t_hidden_states))])
            warnings.warn("Warning! Teacher's and student's hidden states has different length."
                          "Using last teacher's hidden states for distillation.")
        loss = torch.stack([
            self._forward_hidden(cut_s, cur_t, idx)
            for idx, (cut_s, cur_t) in enumerate(zip(s_hidden_states, t_hidden_states))
        ]).mean()
        return loss

    def _forward_hidden(
        self, s_hidden_states: FloatTensor, t_hidden_states: FloatTensor, layer_idx: int = None
    ) -> FloatTensor:

        if self.need_mapping:
            if s_hidden_states.dim() > 3:
                raise TypeError("MSE loss with mapping can be applied only to flatten hidden state")
            assert s_hidden_states.size(-1) == self.student_hidden_state_dim
            assert t_hidden_states.size(-1) == self.teacher_hidden_state_dim
            s_hidden_states = s_hidden_states.reshape(-1, self.student_hidden_state_dim)
            if self.layer_idx is not None:
                t_hidden_states = self.proj[layer_idx](
                    t_hidden_states.reshape(-1, self.teacher_hidden_state_dim)
                )
            else:
                t_hidden_states = self.proj(t_hidden_states.reshape(-1, self.teacher_hidden_state_dim))
            if self.normalize:
                s_hidden_states = F.normalize(s_hidden_states)
                t_hidden_states = F.normalize(t_hidden_states)
        else:

            if s_hidden_states.dim() <= 3:
                hidden_dim = s_hidden_states.size(-1)
                s_hidden_states = s_hidden_states.reshape(-1, hidden_dim)
                t_hidden_states = t_hidden_states.reshape(-1, hidden_dim)
                if self.normalize:
                    s_hidden_states = F.normalize(s_hidden_states)
                    t_hidden_states = F.normalize(t_hidden_states)
            else:
                if self.normalize:
                    raise TypeError("Normalizing can be applied only to flatten hidden state")
                s_hidden_states = s_hidden_states.flatten()
                t_hidden_states = t_hidden_states.flatten()

        assert s_hidden_states.shape == t_hidden_states.shape
        return self.loss_fn(s_hidden_states, t_hidden_states)


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
        loss_fn = lambda s, t: nn.MSELoss(reduction="mean")(F.normalize(s), F.normalize(t))
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
