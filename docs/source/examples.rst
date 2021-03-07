Examples
========

Minimal Example
---------------

Imports

.. code-block:: python

    from itertools import chain

    import torch
    from torch import nn
    from torch.utils.data import DataLoader

    from catalyst.contrib.datasets import MNIST
    from catalyst.callbacks import AccuracyCallback

    from src.distillation.runners import EndToEndDistilRunner
    from src.models import BaseDistilModel
    from src.utils.data import TorchvisionDatasetWrapper as Wrp

Now we can create tiny model class.
The main and the only difference from ordinary pytorch model
is that forward should also supports ``output_hidden_states`` and ``return_dict`` args.

If ``output_hidden_states`` is set to ``True`` model should also output tuple of hidden states.

If ``return_dict`` is set to ``True`` model should be type of dict.

.. code-block:: python

    class ExampleModel(BaseDistilModel):
        def __init__(self, num_layers: int = 4, hidden_dim: int = 128):
            super().__init__()
            layers = []
            self.flatten =nn.Flatten()
            self.num_layers = num_layers
            for layer_idx in range(num_layers):
                if layer_idx == 0:
                    layers.append(nn.Linear(28*28, hidden_dim))
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

Distillation starts here:

.. code-block:: python

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
        num_epochs=4,
        callbacks=[AccuracyCallback(input_key="logits", target_key="targets")],
        valid_metric="accuracy01",
        minimize_valid_metric=False,
        logdir="./logs"
    )

Minimal Complex Example
-----------------------

First of all imports:

.. code-block:: python

    import torch
    from torch import nn
    from torch.utils.data import DataLoader

    from catalyst.contrib.datasets import MNIST
    from catalyst.callbacks import AccuracyCallback, CriterionCallback
    from catalyst.runners import SupervisedRunner

    from src.utils.data import TorchvisionDatasetWrapper as Wrp
    from src.models import BaseDistilModel
    from src.distillation.callbacks import MSEHiddenStatesCallback, HiddenStatesSelectCallback, KLDivCallback, MetricAggregationCallback
    from src.distillation.runners import DistilRunner

Now we can create tiny model class.
The main and the only difference from ordinary pytorch model
is that forward should also supports ``output_hidden_states`` and ``return_dict`` args.

If ``output_hidden_states`` is set to ``True`` model should also output tuple of hidden states.

If ``return_dict`` is set to ``True`` model should be type of dict.

.. code-block:: python

    class ExampleModel(BaseDistilModel):
        def __init__(self, num_layers: int = 4, hidden_dim: int=128):
            layers = []
            self.num_layers = num_layers
            for layer_idx in range(num_layers):
                if layer_idx == 0:
                    layers.append(nn.Linear(28*28, hidden_dim))
                elif layer_idx == num_layers-1:
                    layers.append(nn.Linear(hidden_dim, 10))
                else:
                    layers.append(nn.Linear(hidden_dim, hidden_dim))

            self.layers = nn.ModuleList(*layers)

        def forward(
            self,
            inp,
            output_hidden_states: bool=False,
            return_dict: bool = False
        ):
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

Now we are all-set. Let's begin and define our teacher and student models.

.. code-block:: python

    teacher = ExampleModel(num_layers=4)
    student = ExampleModel(num_layers=3)

Here is data preprocessing:

.. code-block:: python

    datasets = {
        "train": Wrp(MNIST("./data", train=True, download=True)),
        "valid": Wrp(MNIST("./data", train=False))
    }

    loaders = {
        dl_key: DataLoader(dataset, shuffle=dl_key=="train", batch_size=32) for dl_key, dataset in datasets.item()
    }

Now we are ready to train our teacher. This is just simple supervised learning pipeline.

.. code-block:: python

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

Here begins distillation code.

First of all let's define our losses: in addition to ``CrossEntropyLoss``
we will count ``MSELoss`` between hidden states of teacher and student model and
``KLDivLoss`` between output distributions of class probabilities.

But our teacher model has more layers then student model and has more hidden states.
Therefore we will took only last two hidden states of teacher model. We can do it with
``HiddenStatesSelectCallback`` and set ``layers=[2, 3]``.

.. code-block:: python

    select_last_hidden_states = HiddenStatesSelectCallback(layers=[2, 3])

Now we can simply initialize ``MSEHiddenStatesCallback`` for MSE loss and ``KLDivCallback`` for KL-divergence loss

.. code-block:: python

    mse_callback = MSEHiddenStatesCallback()
    kl_callback = KLDivCallback()

Here we can initialize our ``DistilRunner`` and set ``output_hidden_states=True`` as we are using hidden_states in loss

.. code-block:: python

    runner = DistilRunner(output_hidden_states=True)

We can provide only students parameters to optimizer.

.. code-block:: python

    optimizer = torch.optim.Adam(student.parameters(), lr=1e-2)

Now we can run distillation! We also add ``MetricAggregationCallback`` to
callbacks, as we need final loss to be sum of the several losses.

.. math::

    L_\text{loss} = w_1\cdot L_\text{task loss} + w_2\cdot L_\text{KL} + w_3\cdot L_\text{MSE}

We are also setting weights to losses.

.. code-block:: python

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
            MetricAggregationCallback({
                "kl_loss": 0.2,
                "mse_loss": 0.2,
                "loss": 0.6
            })
        ],
        valid_metric="accuracy01",
        minimize_valid_metric=False,
        num_epochs=5,
    )

NLP
---

.. toctree::
    Examples/classification_huggingface_transformers

CV
---
