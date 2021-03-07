import torch
from torch import nn
from torch.utils.data import DataLoader

from catalyst.contrib.datasets import MNIST
from catalyst.callbacks import AccuracyCallback, CriterionCallback
from catalyst.runners import SupervisedRunner

from compressors.utils.data import TorchvisionDatasetWrapper as Wrp
from compressors.models import BaseDistilModel
from compressors.distillation.callbacks import (
    MSEHiddenStatesCallback,
    HiddenStatesSelectCallback,
    KLDivCallback,
    MetricAggregationCallback,
)
from compressors.distillation.runners import DistilRunner


class ExampleModel(BaseDistilModel):
    def __init__(self, num_layers: int = 4, hidden_dim: int = 128):
        super().__init__()
        layers = []
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
        cur_hidden = inp
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
    for dl_key, dataset in datasets.item()
}

optimizer = torch.optim.Adam(teacher.parameters(), lr=1e-2)

runner = SupervisedRunner()

runner.train(
    model=teacher,
    loaders=loaders,
    optimizer=optimizer,
    criterion=nn.CrossEntropyLoss(),
    callbacks=[AccuracyCallback(input_key="logits", target_key="targets")],
    valid_metric="accuracy01",
    minimize_valid_metric=False,
    num_epochs=5,
)

mse_callback = MSEHiddenStatesCallback()
select_last_hidden_states = HiddenStatesSelectCallback(layers=[2, 3])
kl_callback = KLDivCallback()

runner = DistilRunner(output_hidden_states=True)
optimizer = torch.optim.Adam(student.parameters(), lr=1e-2)

runner.train(
    model={"teacher": teacher, "student": student},
    loaders=loaders,
    optimier=optimizer,
    criterion=nn.CrossEntropyLoss(),
    callbacks=[
        AccuracyCallback(input_key="s_logits", target_key="targets"),
        CriterionCallback(input_key="s_logits"),
        mse_callback,
        select_last_hidden_states,
        kl_callback,
        MetricAggregationCallback({"kl_loss": 0.2, "mse_loss": 0.2, "loss": 0.6}),
    ],
    valid_metric="accuracy01",
    minimize_valid_metric=False,
    num_epochs=5,
)
