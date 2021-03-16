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

runner = EndToEndDistilRunner(hidden_state_loss="mse", num_train_teacher_epochs=5)

runner.train(
    model={"teacher": teacher, "student": student},
    loaders=loaders,
    optimizer=optimizer,
    num_epochs=4,
    callbacks=[AccuracyCallback(input_key="logits", target_key="targets")],
    valid_metric="accuracy01",
    minimize_valid_metric=False,
    logdir="./logs",
)
