import numpy as np
import torch
from compressors.distillation.data import swap_smoothing


def test_simple():
    logits = torch.tensor(np.array([[1, 2, 1], [2, 1, 1], [1, 1, 2]]), dtype=torch.float32)
    labels = torch.tensor(np.array([0, 0, 0]))
    s_logits = swap_smoothing(logits, labels)
    target_tensor = torch.tensor(np.array([[2, 1, 1], [2, 1, 1], [2, 1, 1]]), dtype=torch.float32)
    assert torch.isclose(s_logits, target_tensor).type(torch.long).sum() == 9


def test_random():
    logits = torch.randn(100, 100, dtype=torch.float32)
    labels = torch.zeros(100, dtype=torch.long)
    s_logits = swap_smoothing(logits, labels)
    assert (s_logits.argmax(-1) == labels).type(torch.long).sum() == 100
