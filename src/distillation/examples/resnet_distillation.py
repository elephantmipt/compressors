import argparse
import os

import torch
from torch.optim.lr_scheduler import MultiStepLR
from torch import nn
from torch.utils.data import DataLoader

import wandb

from torchvision import transforms
from torchvision import datasets
from models.resnet import PreActResNet18, PreActResNet50
from catalyst.callbacks import AccuracyCallback, CriterionCallback, ControlFlowCallback
from catalyst.runners import SupervisedRunner
from catalyst.utils import set_global_seed

from src.distillation.callbacks import (
    MSEHiddensCallback,
    MetricAggregationCallback,
    KLDivCallback,
)
from src.utils.data import TorchvisionDatasetWrapper as Wrp
from src.distillation.runners import DistilRunner
from src.distillation.utils import get_loss_coefs, load_model_from_path


NAME2MODEL = {
    "pa_resnet_18": PreActResNet18,
    "pa_resnet_50": PreActResNet50,
}

NAME2OPTIM = {
    "sgd": torch.optim.SGD,
    "adam": torch.optim.Adam,
}


def main(args):
    set_global_seed(42)
    # dataloader initialization
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

    train_dataset = Wrp(
        datasets.CIFAR10(root=os.getcwd(), train=True, transform=transform_train, download=True)
    )
    valid_dataset = Wrp(datasets.CIFAR10(root=os.getcwd(), train=False, transform=transform_test))
    train_dataloader = DataLoader(
        dataset=train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2
    )
    valid_dataloader = DataLoader(dataset=valid_dataset, batch_size=128, num_workers=2)
    loaders = {
        "train": train_dataloader,
        "valid": valid_dataloader,
    }
    # model initialization
    model = PreActResNet18()
    model.fc = nn.Linear(512, 10)
    if args.teacher_model is not None:
        is_kd = True
        teacher_model = NAME2MODEL[args.teacher_model]()
        load_model_from_path(model=teacher_model, path=args.teacher_path)
        model = {
            "student": model,
            "teacher": teacher_model,
        }
        output_hiddens = args.beta is None
        is_kd_on_hiddens = output_hiddens
        runner = DistilRunner(device=args.device, output_hidden_states=output_hiddens)
        parameters = model["student"].parameters()
    else:
        is_kd = False
        runner = SupervisedRunner()
        parameters = model.parameters()
    # optimizer
    optimizer_cls = NAME2OPTIM[args.optimizer]
    optimizer_kwargs = {"params": parameters, "lr": args.lr}
    if args.optimizer == "sgd":
        optimizer_kwargs["momentum"] = args.momentum
    else:
        optimizer_kwargs["betas"] = (args.beta1, args.beta2)
    optimizer = optimizer_cls(**optimizer_kwargs)
    scheduler = MultiStepLR(optimizer, milestones=[80, 120], gamma=args.gamma)
    logdir = "logs"
    # callbacks
    callbacks = [
        AccuracyCallback(input_key="s_logits", target_key="targets", num_classes=10),
    ]
    if is_kd:
        metrics = {}
        callbacks.append(
            CriterionCallback(input_key="s_logits", target_key="targets", metric_key="cls_loss")
        )
        callbacks.append(ControlFlowCallback(KLDivCallback(), loaders="train"))
        coefs = get_loss_coefs(args.alpha, args.beta)
        metrics["cls_loss"] = coefs[0]
        metrics["diff_output_loss"] = coefs[1]
        if is_kd_on_hiddens:
            callbacks.append(ControlFlowCallback(MSEHiddensCallback(), loaders="train",))
            metrics["diff_hidden_loss"] = coefs[2]

        aggregator_callback = MetricAggregationCallback(
            prefix="loss", metrics=metrics, mode="weighted_sum"
        )
        wrapped_agg_callback = ControlFlowCallback(aggregator_callback, loaders=["train"])
        callbacks.append(wrapped_agg_callback)

    runner.train(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=nn.CrossEntropyLoss(),
        loaders=loaders,
        callbacks=callbacks,
        num_epochs=args.epoch,
        logdir=logdir,
        verbose=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="pa_resnet50", type=str, help="model to train")
    parser.add_argument("--teacher-model", default=None, type=str, help="teacher arch")
    parser.add_argument("--teacher-path", default=None, type=str, help="path to teacher weights")
    parser.add_argument("--optimizer", default="sgd", type=str)
    parser.add_argument("--lr", default=0.001, type=float)
    parser.add_argument("--batch-size", default=128, type=int)
    parser.add_argument("--momentum", default=0.9, type=float)
    parser.add_argument("--beta1", default=0.99, type=float)
    parser.add_argument("--beta2", default=0.999, type=float)
    parser.add_argument("-d", "--device", default="cuda:0", type=str)
    parser.add_argument("--epoch", default=150, type=int)
    parser.add_argument("--gamma", default=0.1, type=float)
    parser.add_argument(
        "--alpha", default=None, type=float, help="weight for output diff loss",
    )
    parser.add_argument("--beta", default=None, type=float, help="weight for hidden diff loss")
    args = parser.parse_args()
    main(args)
