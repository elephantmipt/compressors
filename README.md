# Compressors

_Warning! Alpha version! This is not product-ready solution so far._

[![CodeFactor](https://www.codefactor.io/repository/github/elephantmipt/compressors/badge)](https://www.codefactor.io/repository/github/elephantmipt/compressors)

Compressors is a library with a lot of pipelines connected with model compression without significantly performance lose.


* [Compressors](#compressors)
   * [Why Compressors?](#why-compressors)
   * [Install](#install)
   * [Features](#features)
      * [Distillation](#distillation)
      * [Pruning](#pruning)
   * [Minimal Examples](#minimal-examples)
      * [Distillation](#distillation-1)
      * [Pruning](#pruning-1)



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

## Features

### Distillation

| Name               | References        | Status      |
| ----------------   | ----------------- | ----------- |
| KL-divergence      | [Hinton et al.](https://arxiv.org/abs/1503.02531)     | Implemented |
| MSE                | [Hinton et al.](https://arxiv.org/abs/1503.02531)     | Implemented |
| Probabilistic KT   | [Passalis et al.](https://arxiv.org/abs/1803.10837)   | Implemented |
| Cosine             | ???                                                   | Implemented |
| Attention Transfer | [Zagoruyko et al.](https://arxiv.org/abs/1612.03928)  | Implemented |
| Constrative Representation Distillation | [Tian et al.](https://arxiv.org/pdf/1910.10699.pdf)| Implemented (without dataset) |
| Probablility Shift  | [Wen et al.](https://arxiv.org/abs/1911.07471) | Implemented and tested |

### Pruning

| Name               | References        | Status      |
| ----------------   | ----------------- | ----------- |
| Lottery ticket hypothesis | [Frankle et al.](https://arxiv.org/abs/1803.03635) | Implemented |
| Iterative pruning  | [Paganini et al.](https://arxiv.org/pdf/2001.05050.pdf)   | Implemented |

## Minimal Examples

### Distillation

```python
from itertools import chain

import torch
from torch.utils.data import DataLoader

from torchvision import transforms as T

from catalyst.contrib.datasets import MNIST
from catalyst.callbacks import AccuracyCallback, OptimizerCallback

from compressors.distillation.runners import EndToEndDistilRunner
from compressors.models import MLP
from compressors.utils.data import TorchvisionDatasetWrapper as Wrp


teacher = MLP(num_layers=4)
student = MLP(num_layers=3)

datasets = {
    "train": Wrp(MNIST("./data", train=True, download=True, transform=T.ToTensor())),
    "valid": Wrp(MNIST("./data", train=False, transform=T.ToTensor())),
}

loaders = {
    dl_key: DataLoader(dataset, shuffle=dl_key == "train", batch_size=32)
    for dl_key, dataset in datasets.items()
}

optimizer = torch.optim.Adam(chain(teacher.parameters(), student.parameters()))

runner = EndToEndDistilRunner(
    hidden_state_loss="mse",
    num_train_teacher_epochs=5
)

runner.train(
    model = torch.nn.ModuleDict({"teacher": teacher, "student": student}),
    loaders=loaders,
    optimizer=optimizer,
    num_epochs=4,
    callbacks=[
        OptimizerCallback(metric_key="loss"), 
        AccuracyCallback(input_key="logits", target_key="targets")
    ],
    valid_metric="accuracy01",
    minimize_valid_metric=False,
    logdir="./logs",
    valid_loader="valid",
    criterion=torch.nn.CrossEntropyLoss()
)
```

### Pruning

```python
import torch
from torch.utils.data import DataLoader

from torchvision.transforms import ToTensor

from catalyst.callbacks import (
    PruningCallback, 
    OptimizerCallback, 
    CriterionCallback, 
    AccuracyCallback, 
    ControlFlowCallback
)
from catalyst.contrib.datasets import MNIST

from compressors.distillation.callbacks import MetricAggregationCallback
from compressors.distillation.callbacks import KLDivCallback
from compressors.models import MLP
from compressors.pruning.runners import FinePruneRunner
from compressors.utils.data import TorchvisionDatasetWrapper as Wrp

model = MLP(num_layers=3)

model = model.load_state_dict(torch.load("trained_model.pth"))

datasets = {
    "train": Wrp(MNIST("./data", train=True, download=True, transform=ToTensor())),
    "valid": Wrp(MNIST("./data", train=False, transform=ToTensor())),
}

loaders = {
    dl_key: DataLoader(dataset, shuffle=dl_key == "train", batch_size=32)
    for dl_key, dataset in datasets.items()
}

optimizer = torch.optim.Adam(model.parameters())

runner = FinePruneRunner(num_sessions=10)

runner.train(
    model=model,
    loaders=loaders,
    optimizer=optimizer,
    criterion=torch.nn.CrossEntropyLoss(),
    callbacks=[
        PruningCallback(pruning_fn="l1_unstructured", amount=0.2, remove_reparametrization_on_stage_end=False),
        OptimizerCallback(metric_key="loss"),
        CriterionCallback(input_key="logits", target_key="targets", metric_key="loss"),
        AccuracyCallback(input_key="logits", target_key="targets"),
        ControlFlowCallback(KLDivCallback(student_logits_key="logits"), loaders="train"),
        ControlFlowCallback(
            MetricAggregationCallback(
                prefix="loss",
                metrics={
                    "loss": 0.6,
                    "kl_div_loss": 0.4,   
                },
                mode="weighted_sum"
            ),
            loaders="train"
        )
    ],
    logdir="./pruned_model",
    valid_loader="valid",
    valid_metric="accuracy",
    minimize_valid_metric=False,
)
```
