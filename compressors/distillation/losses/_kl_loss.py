"""Module with loss functions for knoweledge distillation.

"""
from torch import FloatTensor
from torch import nn
from torch.nn import functional as F


def kl_div_loss(
    s_logits: FloatTensor, t_logits: FloatTensor, temperature: float = 1.0
) -> FloatTensor:
    """KL-devergence loss

    Args:
        s_logits (FloatTensor): output for student model.
        t_logits (FloatTensor): output for teacher model.
        temperature (float, optional): Temperature for teacher distribution.
            Defaults to 1.

    Returns:
        FloatTensor: Divergence between student and teachers distribution.
    """
    loss_fn = nn.KLDivLoss()
    loss = loss_fn(
        F.log_softmax(s_logits / temperature, dim=1), F.softmax(t_logits / temperature, dim=1),
    ) * (temperature ** 2)
    return loss
