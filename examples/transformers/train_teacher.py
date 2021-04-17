import argparse

from catalyst.callbacks import OptimizerCallback
from catalyst.callbacks.metric import LoaderMetricCallback
from datasets import load_dataset, load_metric
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from compressors.metrics.hf_metric import HFMetric
from compressors.runners.hf_runner import HFRunner


def main(args):
    datasets = load_dataset(args.dataset)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
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
    metric_callback = LoaderMetricCallback(
        metric=HFMetric(metric=load_metric("accuracy")), input_key="logits", target_key="labels",
    )
    teacher_model = AutoModelForSequenceClassification.from_pretrained(
        args.model, num_labels=args.num_labels
    )
    runner = HFRunner()
    runner.train(
        model=teacher_model,
        loaders=loaders,
        optimizer=torch.optim.Adam(teacher_model.parameters(), lr=args.lr),
        callbacks=[
            metric_callback,
            OptimizerCallback(metric_key="loss"),
        ],
        num_epochs=args.num_epochs,
        valid_metric="accuracy",
        minimize_valid_metric=False,
        logdir=args.logdir,
        valid_loader="valid",
        verbose=args.verbose
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", "-d", default="ag_news")
    parser.add_argument("--model", default="google/bert_uncased_L-8_H-512_A-8")
    parser.add_argument("--num-labels", default=4, type=int)
    parser.add_argument("--num-epochs", default=5, type=int)
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--batch-size", default=32, type=int)
    parser.add_argument("--logdir", default="bert_teacher", type=str)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    main(args)
