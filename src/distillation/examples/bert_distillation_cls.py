import argparse

import torch
from torch.utils.data import DataLoader

from datasets import load_dataset, load_metric
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from catalyst.runners import Runner
from catalyst.callbacks.metric import LoaderMetricCallback
from catalyst.callbacks import ControlFlowCallback

from src.runners.hf_runner import HFRunner
from src.metrics.hf_metric import HFMetric
from src.distillation.callbacks import (
    MetricAggregationCallback,
    HiddensSlctCallback,
    MSEHiddensCallback,
    KLDivCallback,
)
from src.distillation.runners import HFDistilRunner
from src.distillation.student_init.bert import init_bert_model_with_teacher


def main(args):
    datasets = load_dataset(args.dataset)

    tokenizer = AutoTokenizer.from_pretrained("google/bert_uncased_L-4_H-128_A-2")
    datasets = datasets.map(
        lambda e: tokenizer(
            e["text"], truncation=True, padding="max_length", max_length=128
        ),
        batched=True
    )
    datasets = datasets.map(
        lambda e: {"labels": e["label"]},
        batched=True
    )
    datasets.set_format(
        type="torch", columns=["input_ids", "token_type_ids", "attention_mask", "labels"]
    )
    loaders = {
        "train": DataLoader(datasets["train"], batch_size=args.batch_size, shuffle=True),
        "valid": DataLoader(datasets["test"], batch_size=args.batch_size),
    }
    metric_callback = LoaderMetricCallback(
        metric=HFMetric(metric=load_metric("accuracy")),
        input_key="logits",
        target_key="labels",
    )
    teacher_model = AutoModelForSequenceClassification.from_pretrained(
        "google/bert_uncased_L-4_H-128_A-2", num_labels=args.num_labels
    )
    runner = HFRunner()
    runner.train(
        model=teacher_model,
        loaders=loaders,
        optimizer=torch.optim.Adam(teacher_model.parameters(), lr=args.train_lr),
        callbacks=[metric_callback],
        num_epochs=args.num_train_epochs,
        valid_metric="accuracy",
        minimize_valid_metric=False,
        check=True,
    )
    metric_callback = LoaderMetricCallback(
        metric=HFMetric(metric=load_metric("accuracy")),
        input_key="s_logits",
        target_key="labels",
    )

    slct_callback = ControlFlowCallback(
        HiddensSlctCallback(hiddens_key="t_hidden_states", layers=[1, 3]),
        loaders="train"
    )

    mse_hiddens = ControlFlowCallback(
        MSEHiddensCallback(),
        loaders="train"
    )

    kl_div = ControlFlowCallback(
        KLDivCallback(),
        loaders="train"
    )

    aggregator = ControlFlowCallback(
        MetricAggregationCallback(
            weights={"kl_div_loss": 0.2, "mse_loss": 0.2, "task_loss": 0.6}
        ),
        loaders="train"
    )

    runner = HFDistilRunner()

    student_model = AutoModelForSequenceClassification.from_pretrained(
        "google/bert_uncased_L-2_H-128_A-2", num_labels=args.num_labels
    )
    runner.train(
        model={"teacher": teacher_model, "student": student_model,},
        loaders=loaders,
        optimizer=torch.optim.Adam(student_model.parameters(), lr=args.distil_lr),
        callbacks=[metric_callback, slct_callback, mse_hiddens, kl_div, aggregator],
        check=True,
        num_epochs=args.num_distil_epochs,
        valid_metric="accuracy",
        minimize_valid_metric=False
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", "-d", default="ag_news")
    parser.add_argument("--num-labels", default=4, type=int)
    parser.add_argument("--num-train-epochs", default=5, type=int)
    parser.add_argument("--num-distil-epochs", default=5, type=int)
    parser.add_argument("--train-lr", default=1e-4, type=float)
    parser.add_argument("--distil-lr", default=1e-4, type=float)
    parser.add_argument("--batch-size", default=32, type=int)
    args = parser.parse_args()
    main(args)
