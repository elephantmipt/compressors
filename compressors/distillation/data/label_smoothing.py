import torch


def probability_shift(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    From "Preparing Lessons: Improve Knowledge Distillation with Better Supervision"
    https://arxiv.org/abs/1911.07471. Swaps argmax and correct label in logits.

    Args:
        logits: logits from teacher model
        labels: correct labels

    Returns:
        smoothed labels
    """
    argmax_values, argmax_labels = logits.max(-1)
    arange_indx = torch.arange(logits.size(0))
    logits[arange_indx, argmax_labels] = logits[arange_indx, labels]
    logits[arange_indx, labels] = argmax_values
    return logits
