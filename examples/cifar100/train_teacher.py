import argparse

import torch
from catalyst.callbacks import (
    CriterionCallback, OptimizerCallback, SchedulerCallback, AccuracyCallback
)
from catalyst.runners import SupervisedRunner
from catalyst.utils import set_global_seed
from torch.utils.data import DataLoader

from torchvision import transforms
from torchvision.datasets import CIFAR100

from compressors.models.cv import (
    resnet_cifar_8,
    resnet_cifar_14,
    resnet_cifar_20,
    resnet_cifar_32,
    resnet_cifar_44,
    resnet_cifar_56,
    resnet_cifar_110,
)
from compressors.utils.data import TorchvisionDatasetWrapper as Wrp

NAME2MODEL = {
    "resnet8": resnet_cifar_8,
    "resnet14": resnet_cifar_14,
    "resnet20": resnet_cifar_20,
    "resnet32": resnet_cifar_32,
    "resnet44": resnet_cifar_44,
    "resnet56": resnet_cifar_56,
    "resnet110": resnet_cifar_110,
}


def main(args):

    set_global_seed(args.seed)

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
    model = NAME2MODEL[args.model](num_classes=100)

    optimizer = torch.optim.SGD(
        model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, [150, 180, 210], gamma=0.1
    )

    runner = SupervisedRunner()
    runner.train(
        model=model,
        loaders=loaders,
        optimizer=optimizer,
        scheduler=scheduler,
        valid_metric="accuracy",
        minimize_valid_metric=False,
        logdir=args.logdir,
        callbacks=[
            CriterionCallback(input_key="logits", target_key="targets", metric_key="loss"),
            AccuracyCallback(input_key="logits", target_key="targets"),
            OptimizerCallback(metric_key="loss"),
            SchedulerCallback(),
        ],
        valid_loader="valid",
        num_epochs=args.num_epochs,
        criterion=torch.nn.CrossEntropyLoss(),
        seed=args.seed
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        choices=[
            "resnet8",
            "resnet14",
            "resnet20",
            "resnet32",
            "resnet44",
            "resnet56",
            "resnet110",
        ],
        default="resnet14",
        type=str,
    )
    parser.add_argument("--lr", default=0.05, type=float)
    parser.add_argument("--num-epochs", default=240, type=int)
    parser.add_argument("--batch-size", default=64, type=int)
    parser.add_argument("--logdir", default="cifar100_teacher")
    parser.add_argument("--seed", default=42, type=int)
    args = parser.parse_args()
    main(args)
