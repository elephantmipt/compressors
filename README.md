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
            * [MNIST](#mnist)
            * [CIFAR100 ResNet](#cifar100-resnet)
            * [AG NEWS BERT (transformers)](#ag-news-bert-transformers)
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

#### MNIST

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

#### CIFAR100 ResNet

```python
from catalyst.callbacks import (
    AccuracyCallback,
    ControlFlowCallback,
    CriterionCallback,
    OptimizerCallback,
    SchedulerCallback,
)
import torch
from torch.hub import load_state_dict_from_url
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR100

from compressors.distillation.callbacks import (
    AttentionHiddenStatesCallback,
    KLDivCallback,
    MetricAggregationCallback,
)
from compressors.distillation.runners import DistilRunner
from compressors.models.cv import resnet_cifar_8, resnet_cifar_56

from compressors.utils.data import TorchvisionDatasetWrapper as Wrp


transform_train = transforms.Compose(
    [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
           transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]
)

transform_test = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]
)

datasets = {
    "train": Wrp(CIFAR100(root=".", train=True, download=True, transform=transform_train)),
    "valid": Wrp(CIFAR100(root=".", train=False, transform=transform_test)),
}

loaders = {
    k: DataLoader(v, batch_size=args.batch_size, shuffle=k == "train", num_workers=2)
    for k, v in datasets.items()
}

teacher_sd = load_state_dict_from_url(
    "https://github.com/chenyaofo/CIFAR-pretrained-models/releases/download/resnet/cifar100-resnet56-2f147f26.pth"
)
teacher_model = resnet_cifar_56(num_classes=100)
teacher_model.load_state_dict(teacher_sd)
student_model = resnet_cifar_8(num_classes=100)

optimizer = torch.optim.SGD(
    student_model.parameters(), lr=1e-2, momentum=0.9, weight_decay=5e-4
)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

runner = DistilRunner(apply_probability_shift=args.probability_shift)
runner.train(
    model={"teacher": teacher_model, "student": student_model},
    loaders=loaders,
    optimizer=optimizer,
    scheduler=scheduler,
    valid_metric="accuracy",
    minimize_valid_metric=False,
    logdir="./cifar100_logs",
    callbacks=[
        ControlFlowCallback(AttentionHiddenStatesCallback(), loaders="train"),
        ControlFlowCallback(KLDivCallback(temperature=4), loaders="train"),
        CriterionCallback(input_key="s_logits", target_key="targets", metric_key="cls_loss"),
        ControlFlowCallback(
            MetricAggregationCallback(
                prefix="loss",
                metrics={
                    "attention_loss": 1000,
                    "kl_div_loss": 0.9,
                    "cls_loss": 0.1,
                },
                mode="weighted_sum",
            ),
            loaders="train",
        ),
        AccuracyCallback(input_key="s_logits", target_key="targets"),
        OptimizerCallback(metric_key="loss", model_key="student"),
        SchedulerCallback(),
    ],
    valid_loader="valid",
    num_epochs=200,
    criterion=torch.nn.CrossEntropyLoss(),
)
```

#### AG NEWS BERT (transformers)

```python
import torch
from torch.utils.data import DataLoader

from datasets import load_dataset, load_metric
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from catalyst.callbacks import ControlFlowCallback, OptimizerCallback
from catalyst.callbacks.metric import LoaderMetricCallback

from compressors.distillation.callbacks import (
    HiddenStatesSelectCallback,
    KLDivCallback,
    LambdaPreprocessCallback,
    MetricAggregationCallback,
    MSEHiddenStatesCallback,
)
from compressors.distillation.runners import HFDistilRunner
from compressors.metrics.hf_metric import HFMetric
from compressors.runners.hf_runner import HFRunner


datasets = load_dataset("ag_news")

tokenizer = AutoTokenizer.from_pretrained("google/bert_uncased_L-4_H-128_A-2")
datasets = datasets.map(
    lambda e: tokenizer(e["text"], truncation=True, padding="max_length", max_length=128),
    batched=True,
)
datasets = datasets.map(lambda e: {"labels": e["label"]}, batched=True)
datasets.set_format(
    type="torch", columns=["input_ids", "token_type_ids", "attention_mask", "labels"],
)
loaders = {
    "train": DataLoader(datasets["train"], batch_size=64, shuffle=True),
    "valid": DataLoader(datasets["test"], batch_size=64),
}
metric_callback = LoaderMetricCallback(
    metric=HFMetric(metric=load_metric("accuracy")), input_key="logits", target_key="labels",
)

################### Teacher Training #####################

teacher_model = AutoModelForSequenceClassification.from_pretrained(
    "google/bert_uncased_L-4_H-128_A-2", num_labels=4
)
runner = HFRunner()
runner.train(
    model=teacher_model,
    loaders=loaders,
    optimizer=torch.optim.Adam(teacher_model.parameters(), lr=1e-4),
    callbacks=[metric_callback],
    num_epochs=5,
    valid_metric="accuracy",
    minimize_valid_metric=False,
    verbose=True
)

############### Distillation ##################

slct_callback = ControlFlowCallback(
    HiddenStatesSelectCallback(hiddens_key="t_hidden_states", layers=[1, 3]), loaders="train",
)

lambda_hiddens_callback = ControlFlowCallback(
    LambdaSelectCallback(
        lambda s_hiddens, t_hiddens: (
            [c_s[:, 0] for c_s in s_hiddens],
            [t_s[:, 0] for t_s in t_hiddens],  # tooks only CLS token
        )
    ),
    loaders="train",
)

mse_hiddens = ControlFlowCallback(MSEHiddenStatesCallback(), loaders="train")

kl_div = ControlFlowCallback(KLDivCallback(temperature=4), loaders="train")

aggregator = ControlFlowCallback(
    MetricAggregationCallback(
        prefix="loss",
        metrics={"kl_div_loss": 0.2, "mse_loss": 0.2, "task_loss": 0.6},
        mode="weighted_sum",
    ),
    loaders="train",
)

runner = HFDistilRunner()

student_model = AutoModelForSequenceClassification.from_pretrained(
    "google/bert_uncased_L-2_H-128_A-2", num_labels=4
)

metric_callback = LoaderMetricCallback(
    metric=HFMetric(metric=load_metric("accuracy")), input_key="s_logits", target_key="labels",
)

runner.train(
    model=torch.nn.ModuleDict({"teacher": teacher_model, "student": student_model}),
    loaders=loaders,
    optimizer=torch.optim.Adam(student_model.parameters(), lr=1e-4),
    callbacks=[
        metric_callback,
        slct_callback,
        lambda_hiddens_callback,
        mse_hiddens,
        kl_div,
        aggregator,
        OptimizerCallback(metric_key="loss"),
    ],
    num_epochs=5,
    valid_metric="accuracy",
    minimize_valid_metric=False,
    valid_loader="valid",
    verbose=True
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
