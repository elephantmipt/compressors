# Compressors

_Warning! Deep alpha version! This is not product-ready solutiion so far._

[![CodeFactor](https://www.codefactor.io/repository/github/elephantmipt/compressors/badge)](https://www.codefactor.io/repository/github/elephantmipt/compressors)

Compressors is a library with a lot of pipelines connected with model compression without significantly performance lose.

## Why Compressors?

Compressors provides many ways to compress your model. You can use it for CV and NLP task.

Library separated into three parts:

- Distillation
- Pruning
- Quantization

There are two ways to use Compressors: with Catalyst or just use functional API.

## Install

```bash
pip install git+https://github.com/elephantmipt/compressors.git
```

## Losses

| Loss               | References        | Status      |
| ----------------   | ----------------- | ----------- |
| KL-divergence      | [Hinton et al.](https://arxiv.org/abs/1503.02531)     | Implemented |
| MSE                | [Hinton et al.](https://arxiv.org/abs/1503.02531)     | Implemented |
| Probabilistic KT   | [Passalis et al.](https://arxiv.org/abs/1803.10837)   | Implemented |
| Cosine             | ???                                                   | Implemented |
| Attention Transfer | [Zagoruyko et al.](https://arxiv.org/abs/1612.03928)  | Implemented |
| Constrative Representation Distillation | [Tian et al.](https://arxiv.org/pdf/1910.10699.pdf)| Implemented (without dataset) |


## Minimal Example

```python
from itertools import chain

import torch
from torch import nn
from torch.utils.data import DataLoader

from catalyst.contrib.datasets import MNIST
from catalyst.callbacks import AccuracyCallback

from compressors.distillation.runners import EndToEndDistilRunner
from compressors.models import MLP
from compressors.utils.data import TorchvisionDatasetWrapper as Wrp


teacher = MLP(num_layers=4)
student = MLP(num_layers=3)

datasets = {
    "train": Wrp(MNIST("./data", train=True, download=True)),
    "valid": Wrp(MNIST("./data", train=False)),
}

loaders = {
    dl_key: DataLoader(dataset, shuffle=dl_key == "train", batch_size=32)
    for dl_key, dataset in datasets.item()
}

optimizer = torch.optim.Adam(chain(teacher.parameters(), student.parameters()))

runner = EndToEndDistilRunner(
    hidden_state_loss="mse",
    num_train_teacher_epochs=5
)

runner.train(
    model = {"teacher": teacher, "student": student},
    loaders=loaders,
    optimizer=optimizer,
    num_epochs=4,
    callbacks=[AccuracyCallback(input_key="logits", target_key="targets")],
    valid_metric="accuracy01",
    minimize_valid_metric=False,
    logdir="./logs"
)
```
