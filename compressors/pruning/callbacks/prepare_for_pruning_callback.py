from torch.utils.data import DataLoader

from catalyst.core import Callback, CallbackOrder, CallbackNode, IRunner
from compressors.distillation.data import LogitsDataset


class PrepareForFinePruningCallback(Callback):
    def __init__(self, train_loader_key: str = "train"):
        super(PrepareForFinePruningCallback, self).__init__(
            order=CallbackOrder.External, node=CallbackNode.Master
        )
        self.train_loader_key = train_loader_key

    def on_stage_start(self, runner: "IRunner") -> None:
        if not runner.stage_key == "preparing":
            return
        train_loader = runner.loaders[self.train_loader_key]
        train_dataset = LogitsDataset(train_loader.dataset, runner.model)
        train_loader = DataLoader(
            train_dataset,
            batch_size=train_loader.batch_size,
            sampler=train_loader.sampler,
            collate_fn=train_loader.collate_fn,
            num_workers=train_loader.num_workers,
            drop_last=train_loader.drop_last,
        )
        runner.loaders[self.train_loader_key] = train_loader
