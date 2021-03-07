import torch
from torch.utils import data


class TorchvisionDatasetWrapper(data.Dataset):
    """Simple wrapper. We need all datasets to output dict.

    Args:
        torchvision_dataset (data.Dataset): base dataset.
    """

    def __init__(self, torchvision_dataset: data.Dataset):
        """Simple wrapper. We need all datasets to output dict.

        Args:
            torchvision_dataset (data.Dataset): base dataset.
        """
        self.dataset = torchvision_dataset

    def __getitem__(self, item):
        features, targets = self.dataset[item]
        return {
            "features": torch.tensor(features, dtype=torch.float32),
            "targets": torch.tensor(targets, dtype=torch.long),
        }

    def __len__(self):
        return len(self.dataset)


__all__ = ["TorchvisionDatasetWrapper"]
