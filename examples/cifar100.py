import argparse

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

NAME2URL = {
    "resnet20": "https://github.com/chenyaofo/pytorch-cifar-models/releases/download/resnet/cifar100-resnet20-8412cc70.pth",
    "resnet32": "https://github.com/chenyaofo/pytorch-cifar-models/releases/download/resnet/cifar100-resnet32-6568a0a0.pth",
    "resnet44": "https://github.com/chenyaofo/pytorch-cifar-models/releases/download/resnet/cifar100-resnet44-20aaa8cf.pth",
    "resnet56": "https://github.com/chenyaofo/pytorch-cifar-models/releases/download/resnet/cifar100-resnet56-2f147f26.pth",
}

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

    teacher_sd = load_state_dict_from_url(NAME2URL[args.teacher])
    teacher_model = NAME2MODEL[args.teacher](num_classes=100)
    teacher_model.load_state_dict(teacher_sd)
    student_model = NAME2MODEL[args.student](num_classes=100)

    optimizer = torch.optim.SGD(
        student_model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4
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
                        "attention_loss": args.beta,
                        "kl_div_loss": args.alpha,
                        "cls_loss": 1 - args.alpha,
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--teacher",
        default="resnet56",
        choices=["resnet20", "resnet32", "resnet44", "resnet56"],
        type=str,
    )
    parser.add_argument(
        "--student",
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
    parser.add_argument("--lr", default=0.01, type=float)
    parser.add_argument("--num-epochs", default=200, type=int)
    parser.add_argument("--probability-shift", action="store_true")
    parser.add_argument("--batch-size", default=64, type=int)
    parser.add_argument("--alpha", default=0.9, type=float)
    parser.add_argument("--beta", default=1000, type=float)
    args = parser.parse_args()
    main(args)
