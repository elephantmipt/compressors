from typing import Callable

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm.autonotebook import tqdm

from compressors.utils import any2device


class LogitsDataset(Dataset):
    """
    Dataset wrapper for taking logits from trained model.

    Args:
        dataset: base dataset
        model: trained model
        batched: flag. If true then getting logits with dataloader.
        get_logits_fn: function for taking logits from model and batch.
        merge_logits_with_batch_fn: function to merge data from dataset and logits.
        **data_loader_kwargs: kwargs for dataloader. For exapmle, {"batch_size": 32}.
    """

    def __init__(
        self,
        dataset: Dataset,
        model: nn.Module,
        batched: bool = True,
        get_logits_fn: Callable = None,
        merge_logits_with_batch_fn: Callable = None,
        **data_loader_kwargs,
    ):
        """
        Dataset wrapper for taking logits from trained model.

        Args:
            dataset: base dataset
            model: trained model
            batched: flag. If true then getting logits with dataloader.
            get_logits_fn: function for taking logits from model and batch.
            merge_logits_with_batch_fn: function to merge data from dataset and logits.
            **data_loader_kwargs: kwargs for dataloader. For exapmle, {"batch_size": 32}.
        """
        self.dataset = dataset
        self.model = model
        self.batched = batched
        self.dataloader_kwargs = data_loader_kwargs
        self.device = next(model.parameters()).device
        self.get_logits_fn = get_logits_fn or self._base_get_logits_fn
        self.logits = None
        self._compute_logits()
        self.merge_logits_with_batch_fn = merge_logits_with_batch_fn or self._base_merge_fn

    def __getitem__(self, item):
        c_logits = self.logits[item]
        data_item = self.dataset[item]
        return self._base_merge_fn(data_item, c_logits)

    def __len__(self):
        return len(self.dataset)

    @torch.no_grad()
    def _compute_logits(self):
        self.model.eval()
        dataloader = self.dataset
        batch_size = 32
        if self.batched:
            dataloader = DataLoader(dataset=self.dataset, **self.dataloader_kwargs)
            batch_size = dataloader.batch_size

        logits = None
        for idx, batch in enumerate(tqdm(dataloader, desc="Taking logits from model", ncols=1000)):
            batch = any2device(batch, self.device)
            c_logits = self.get_logits_fn(self.model, batch).cpu()
            start_idx = batch_size * idx
            indx = torch.arange(start_idx, start_idx + c_logits.size(0))
            if logits is None:
                logits = torch.zeros(len(self.dataset), c_logits.size(1), dtype=torch.float32)
            logits[indx] += c_logits
        self.logits = logits

    @staticmethod
    def _base_get_logits_fn(model, batch):
        return model(batch["features"])

    @staticmethod
    def _base_merge_fn(batch, logits):
        if isinstance(batch, dict):
            batch["t_logits"] = logits
        elif isinstance(batch, torch.Tensor):
            batch = {"features": batch, "t_logits": logits}
        else:
            raise TypeError(f"Can't handle batch type of {type(batch)}")
        return batch
