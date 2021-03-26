"""Module with loss functions for knoweledge distillation.

"""
from torch import FloatTensor, nn
from torch.nn import functional as F


class KLDivLoss(nn.Module):
    def __init__(self, temperature: float = 1.0):
        super(KLDivLoss, self).__init__()
        self.temperature = temperature
        self.criterion = nn.KLDivLoss()

    def forward(self, s_logits, t_logits):
        return self.criterion(
            F.log_softmax(s_logits / self.temperature, dim=1),
            F.softmax(t_logits / self.temperature, dim=1),
        ) * (self.temperature ** 2)


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
