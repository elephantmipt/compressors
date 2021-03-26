from typing import Tuple

import torch
from torch import FloatTensor, nn
from torch.nn import functional as F


class AttentionLoss(nn.Module):
    def __init__(self, p: int = 2):
        super().__init__()
        self.p = p

    def forward(
        self, s_hidden_states: Tuple[FloatTensor], t_hidden_states: Tuple[FloatTensor]
    ) -> FloatTensor:

        return torch.stack(
            [self.at_loss(f_s, f_t) for f_s, f_t in zip(s_hidden_states, t_hidden_states)]
        ).mean()

    def at_loss(self, f_s, f_t):
        # code from
        # https://github.com/HobbitLong/RepDistiller/blob/9b56e974ad/distiller_zoo/AT.py#L18
        s_H, t_H = f_s.shape[2], f_t.shape[2]
        if s_H > t_H:
            f_s = F.adaptive_avg_pool2d(f_s, (t_H, t_H))
        elif s_H < t_H:
            f_t = F.adaptive_avg_pool2d(f_t, (s_H, s_H))
        else:
            pass
        return (self.at(f_s) - self.at(f_t)).pow(2).mean()

    def at(self, f):
        return F.normalize(f.pow(self.p).mean(1).view(f.size(0), -1))
