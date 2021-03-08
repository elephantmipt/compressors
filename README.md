# Compressors

_Warning! Deep alpha version_

Compressors is a library with a lot of pipelines connected with model compression without significantly performance lose.

## Why Compressors?

Compressors provides many ways to compress your model. You can use it for CV and NLP task.

Library separated into three parts:

- Distillation
- Pruning
- Quantization

There are two ways to use Compressors: with Catalyst or just use functional API.

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
from compressors.models import BaseDistilModel
from compressors.utils.data import TorchvisionDatasetWrapper as Wrp


class ExampleModel(BaseDistilModel):
    def __init__(self, num_layers: int = 4, hidden_dim: int = 128):
        super().__init__()
        layers = []
        self.flatten = nn.Flatten()
        self.num_layers = num_layers
        for layer_idx in range(num_layers):
            if layer_idx == 0:
                layers.append(nn.Linear(28 * 28, hidden_dim))
            elif layer_idx == num_layers - 1:
                layers.append(nn.Linear(hidden_dim, 10))
            else:
                layers.append(nn.Linear(hidden_dim, hidden_dim))

        self.layers = nn.ModuleList(layers)

    def forward(self, inp, output_hidden_states: bool = False, return_dict: bool = False):
        cur_hidden = self.flatten(inp)
        if output_hidden_states:
            hiddens = []

        for layer_idx, layer in enumerate(self.layers):
            cur_hidden = layer(cur_hidden)
            if output_hidden_states:  #  accumulate hidden states
                hiddens.append(cur_hidden)

            if layer_idx != self.num_layers - 1:  # last layer case
                cur_hidden = torch.relu(cur_hidden)

        logits = cur_hidden
        if return_dict:
            output = {"logits": logits}
            if output_hidden_states:
                output["hidden_states"] = tuple(hiddens)
            return output

        if output_hidden_states:
            return logits, tuple(hiddens)

        return logits

teacher = ExampleModel(num_layers=4)
student = ExampleModel(num_layers=3)

datasets = {
    "train": Wrp(MNIST("./data", train=True, download=True)),
    "valid": Wrp(MNIST("./data", train=False)),
}

loaders = {
    dl_key: DataLoader(dataset, shuffle=dl_key == "train", batch_size=32)
    for dl_key, dataset in datasets.items()
}

optimizer = torch.optim.Adam(chain(teacher.parameters(), student.parameters()))

runner = EndToEndDistilRunner(
    hidden_state_loss="pkt_loss",
    num_train_teacher_epochs=5
)

runner.train(
    model = {"teacher": teacher, "student": student},
    loaders=loaders,
    optimizer=optimizer,
    criterion=nn.CrossEntropyLoss(),
    num_epochs=4,
    callbacks=[AccuracyCallback(input_key="logits", target_key="targets")],
    valid_metric="accuracy01",
    minimize_valid_metric=False,
    check=True
)
```

## Install

```bash
pip install git+https://github.com/elephantmipt/compressors.git
```
