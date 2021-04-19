import argparse

from catalyst.callbacks import ControlFlowCallback, OptimizerCallback, CheckpointCallback
from catalyst.callbacks.metric import LoaderMetricCallback
from catalyst.utils import unpack_checkpoint, set_global_seed
from datasets import load_dataset, load_metric
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from compressors.distillation.callbacks import (
    HiddenStatesSelectCallback,
    KLDivCallback,
    LambdaPreprocessCallback,
    MetricAggregationCallback,
    MSEHiddenStatesCallback,
)
from compressors.distillation.runners import HFDistilRunner
from compressors.metrics.hf_metric import HFMetric


def main(args):
    if args.wandb:
        import wandb
        wandb.init()
        logdir = args.logdir + "/" + wandb.run.name
    else:
        logdir = args.logdir
    set_global_seed(args.seed)
    datasets = load_dataset(args.dataset)

    tokenizer = AutoTokenizer.from_pretrained(args.teacher_model)
    datasets = datasets.map(
        lambda e: tokenizer(e["text"], truncation=True, padding="max_length", max_length=128),
        batched=True,
    )
    datasets = datasets.map(lambda e: {"labels": e["label"]}, batched=True)
    datasets.set_format(
        type="torch", columns=["input_ids", "token_type_ids", "attention_mask", "labels"],
    )
    loaders = {
        "train": DataLoader(datasets["train"], batch_size=args.batch_size, shuffle=True),
        "valid": DataLoader(datasets["test"], batch_size=args.batch_size),
    }
    teacher_model = AutoModelForSequenceClassification.from_pretrained(
        args.teacher_model, num_labels=args.num_labels
    )
    unpack_checkpoint(torch.load(args.teacher_path), model=teacher_model)
    metric_callback = LoaderMetricCallback(
        metric=HFMetric(metric=load_metric("accuracy")), input_key="s_logits", target_key="labels",
    )
    layers = [int(layer) for layer in args.layers.split(",")]
    slct_callback = ControlFlowCallback(
        HiddenStatesSelectCallback(hiddens_key="t_hidden_states", layers=layers), loaders="train",
    )

    lambda_hiddens_callback = ControlFlowCallback(
        LambdaPreprocessCallback(
            lambda s_hiddens, t_hiddens: (
                [c_s[:, 0] for c_s in s_hiddens],
                [t_s[:, 0] for t_s in t_hiddens],  # tooks only CLS token
            )
        ),
        loaders="train",
    )

    mse_hiddens = ControlFlowCallback(MSEHiddenStatesCallback(), loaders="train")

    kl_div = ControlFlowCallback(KLDivCallback(temperature=args.kl_temperature), loaders="train")

    runner = HFDistilRunner()

    student_model = AutoModelForSequenceClassification.from_pretrained(
        args.student_model, num_labels=args.num_labels
    )
    callbacks = [
        metric_callback,
        slct_callback,
        lambda_hiddens_callback,
        kl_div,
        OptimizerCallback(metric_key="loss"),
        CheckpointCallback(
            logdir=logdir,
            loader_key="valid",
            mode="model",
            metric_key="accuracy",
            minimize=False
        )
    ]
    if args.beta > 0:
        aggregator = ControlFlowCallback(
            MetricAggregationCallback(
                prefix="loss",
                metrics={
                    "kl_div_loss": args.alpha, "mse_loss": args.beta, "task_loss": 1 - args.alpha
                },
                mode="weighted_sum",
            ),
            loaders="train",
        )
        callbacks.append(mse_hiddens)
        callbacks.append(aggregator)
    else:
        aggregator = ControlFlowCallback(
            MetricAggregationCallback(
                prefix="loss",
                metrics={
                    "kl_div_loss": args.alpha, "task_loss": 1 - args.alpha
                },
                mode="weighted_sum",
            ),
            loaders="train",
        )
        callbacks.append(aggregator)
    runner.train(
        model=torch.nn.ModuleDict({"teacher": teacher_model, "student": student_model}),
        loaders=loaders,
        optimizer=torch.optim.Adam(student_model.parameters(), lr=args.lr),
        callbacks=callbacks,
        num_epochs=args.num_epochs,
        valid_metric="accuracy",
        logdir=logdir,
        minimize_valid_metric=False,
        valid_loader="valid",
        verbose=args.verbose,
        seed=args.seed
    )

    if args.wandb:
        import csv
        import shutil
        with open(logdir + "/valid.csv") as fi:
            reader = csv.DictReader(fi)
            accuracy = []
            for row in reader:
                if row["accuracy"] == "accuracy":
                    continue
                accuracy.append(float(row["accuracy"]))

        wandb.log({"accuracy": max(accuracy[-args.num_epochs:])})
        shutil.rmtree(logdir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", "-d", default="ag_news")
    parser.add_argument("--teacher-model", default="google/bert_uncased_L-8_H-512_A-8", type=str)
    parser.add_argument("--student-model", default="google/bert_uncased_L-4_H-512_A-8", type=str)
    parser.add_argument("--teacher-path", default="bert_teacher/checkpoint/best.pth", type=str)
    parser.add_argument("--layers", default="1,3,5,7", type=str)
    parser.add_argument("--alpha", default=0.3, type=float)
    parser.add_argument("--beta", default=1., type=float)
    parser.add_argument("--num-labels", default=4, type=int)
    parser.add_argument("--num-epochs", default=5, type=int)
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--logdir", default="bert_student")
    parser.add_argument("--batch-size", default=32, type=int)
    parser.add_argument("--kl-temperature", default=4.0, type=float)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--seed", default=42, type=int)
    args = parser.parse_args()
    main(args)
