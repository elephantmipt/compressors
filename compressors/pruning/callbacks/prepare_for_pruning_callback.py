from catalyst.core import Callback, CallbackNode, CallbackOrder, IRunner
from torch.utils.data import DataLoader

from compressors.distillation.data import LogitsDataset


class PrepareForFinePruningCallback(Callback):
    def __init__(
        self, train_loader_key: str = "train", *logits_dataset_args, **logits_dataset_kwargs
    ):
        super(PrepareForFinePruningCallback, self).__init__(
            order=CallbackOrder.External, node=CallbackNode.Master
        )
        self.train_loader_key = train_loader_key
        self.logits_dataset_args_kwargs = (logits_dataset_args, logits_dataset_kwargs)

    def on_experiment_start(self, runner: "IRunner") -> None:
        train_loader = runner.loaders[self.train_loader_key]
        train_dataset = LogitsDataset(
            train_loader.dataset,
            runner.model,
            *self.logits_dataset_args_kwargs[0],
            **self.logits_dataset_args_kwargs[1]
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=train_loader.batch_size,
            sampler=train_loader.sampler,
            collate_fn=train_loader.collate_fn,
            num_workers=train_loader.num_workers,
            drop_last=train_loader.drop_last,
        )
        runner.loaders[self.train_loader_key] = train_loader
