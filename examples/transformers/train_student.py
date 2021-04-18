import argparse

from catalyst.callbacks import ControlFlowCallback, OptimizerCallback
from catalyst.callbacks.metric import LoaderMetricCallback
from catalyst.utils import unpack_checkpoint
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

    runner = HFDistilRunner()

    student_model = AutoModelForSequenceClassification.from_pretrained(
        args.student_model, num_labels=args.num_labels
    )
    runner.train(
        model=torch.nn.ModuleDict({"teacher": teacher_model, "student": student_model}),
        loaders=loaders,
        optimizer=torch.optim.Adam(student_model.parameters(), lr=args.lr),
        callbacks=[
            metric_callback,
            slct_callback,
            lambda_hiddens_callback,
            mse_hiddens,
            kl_div,
            aggregator,
            OptimizerCallback(metric_key="loss"),
        ],
        num_epochs=args.num_epochs,
        valid_metric="accuracy",
        logdir=args.logdir,
        minimize_valid_metric=False,
        valid_loader="valid",
        verbose=args.verbose
    )


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
    args = parser.parse_args()
    main(args)
