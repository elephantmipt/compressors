import argparse
from functools import partial

import torch
from torch import nn

from catalyst import dl
from catalyst import utils
from catalyst.contrib.datasets import MNIST
from catalyst.contrib.nn import Flatten
from catalyst.data.transforms import ToTensor

from compressors.pruning.runners import PruneRunner
from compressors.utils.data import TorchvisionDatasetWrapper
from compressors.distillation.callbacks import KLDivCallback, MetricAggregationCallback
from compressors.pruning.callbacks import LotteryTicketCallback, PrepareForFinePruningCallback


def validate_model(runner, loader, pruning_fn, num_sessions):
    accuracy_scores = []
    pruned_weights = []
    c_p = 1
    for pruning_idx in range(num_sessions):
        correct = 0
        len_dataset = 0
        for batch in loader:
            outp = runner.predict_batch(utils.any2device(batch, "cuda"))
            c_correct = torch.sum(
                outp["logits"].argmax(-1).detach().cpu() == batch["targets"]
            ).item()
            correct += c_correct
            len_dataset += batch["features"].size(0)
        pruned_weights.append(c_p)
        c_p *= 0.9
        accuracy_scores.append(correct / len_dataset)
        pruning_fn(runner.model)
    return accuracy_scores, pruned_weights


def main(args):
    train_dataset = TorchvisionDatasetWrapper(
        MNIST(root="./", download=True, train=True, transform=ToTensor())
    )
    val_dataset = TorchvisionDatasetWrapper(
        MNIST(root="./", download=True, train=False, transform=ToTensor())
    )

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=64)
    loaders = {"train": train_dataloader, "valid": val_dataloader}
    utils.set_global_seed(args.seed)
    net = nn.Sequential(
        Flatten(),
        nn.Linear(28 * 28, 300),
        nn.ReLU(),
        nn.Linear(300, 100),
        nn.ReLU(),
        nn.Linear(100, 10),
    )
    initial_state_dict = net.state_dict()

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters())
    if args.vanilla_pruning:
        runner = dl.SupervisedRunner()

        runner.train(
            model=net,
            criterion=criterion,
            optimizer=optimizer,
            loaders=loaders,
            callbacks=[
                dl.AccuracyCallback(input_key="logits", target_key="targets", num_classes=10),
            ],
            logdir="./logdir",
            num_epochs=args.num_epochs,
            load_best_on_end=True,
            valid_metric="accuracy01",
            minimize_valid_metric=False,
            valid_loader="valid",
        )
        pruning_fn = partial(
            utils.pruning.prune_model,
            pruning_fn=args.pruning_method,
            amount=args.amount,
            keys_to_prune=["weights"],
            dim=args.dim,
            l_norm=args.n,
        )
        acc, amount = validate_model(
            runner, pruning_fn=pruning_fn, loader=loaders["valid"], num_sessions=args.num_sessions
        )
        torch.save(acc, "accuracy.pth")
        torch.save(amount, "amount.pth")

    else:
        runner = PruneRunner(num_sessions=args.num_sessions)
        callbacks = [
            dl.AccuracyCallback(input_key="logits", target_key="targets", num_classes=10),
            dl.PruningCallback(
                args.pruning_method,
                keys_to_prune=["weight"],
                amount=args.amount,
                remove_reparametrization_on_stage_end=False,
            ),
            dl.CriterionCallback(input_key="logits", target_key="targets", metric_key="loss"),
            dl.OptimizerCallback(metric_key="loss"),
        ]
        if args.lottery_ticket:
            callbacks.append(LotteryTicketCallback(initial_state_dict=initial_state_dict))
        if args.kd:
            net.load_state_dict(torch.load(args.state_dict))
            callbacks.append(
                PrepareForFinePruningCallback(probability_shift=args.probability_shift)
            )
            callbacks.append(KLDivCallback(temperature=4, student_logits_key="logits"))
            callbacks.append(
                MetricAggregationCallback(
                    prefix="loss", metrics={"loss": 0.1, "kl_div_loss": 0.9}, mode="weighted_sum"
                )
            )

        runner.train(
            model=net,
            criterion=criterion,
            optimizer=optimizer,
            loaders=loaders,
            callbacks=callbacks,
            logdir=args.logdir,
            num_epochs=args.num_epochs,
            load_best_on_end=True,
            valid_metric="accuracy01",
            minimize_valid_metric=False,
            valid_loader="valid",
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_sessions", default=35, type=int)
    parser.add_argument("--num-epochs", default=10, type=int)
    parser.add_argument("--amount", default=0.1, type=float)
    parser.add_argument(
        "--pruning-method",
        type=str,
        choices=["ln_structured", "l1_unstructured", "random_unstructured", "random_structured"],
        default="l1_unstructured",
    )
    parser.add_argument("--dim", type=int, choices=[0, 1, None], default=None)
    parser.add_argument("-n", type=int, default=None)
    parser.add_argument("--kd", action="store_true")
    parser.add_argument("--lottery-ticket", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--vanilla-pruning", action="store_true")
    parser.add_argument("--probability-shift", action="store_true")
    parser.add_argument("--logdir", default="logs", type=str)
    parser.add_argument("--state-dict", type=str)
    args = parser.parse_args()
    main(args)
