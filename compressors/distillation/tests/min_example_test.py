from itertools import chain

from catalyst.callbacks import AccuracyCallback, OptimizerCallback
from catalyst.contrib.datasets import MNIST
import torch
from torch.utils.data import DataLoader
from torchvision import transforms as T

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

runner = EndToEndDistilRunner(hidden_state_loss="mse", num_train_teacher_epochs=5)

runner.train(
    model=torch.nn.ModuleDict({"teacher": teacher, "student": student}),
    loaders=loaders,
    optimizer=optimizer,
    num_epochs=4,
    callbacks=[
        OptimizerCallback(metric_key="loss"),
        AccuracyCallback(input_key="logits", target_key="targets"),
    ],
    valid_metric="accuracy01",
    minimize_valid_metric=False,
    logdir="./logs",
    valid_loader="valid",
    criterion=torch.nn.CrossEntropyLoss(),
    check=True,
)
