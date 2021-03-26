from catalyst.callbacks import ControlFlowCallback, OptimizerCallback
from catalyst.callbacks.metric import LoaderMetricCallback
from datasets import load_dataset, load_metric
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from compressors.distillation.callbacks import (
    HiddenStatesSelectCallback,
    KLDivCallback,
    LambdaSelectCallback,
    MetricAggregationCallback,
    MSEHiddenStatesCallback,
)
from compressors.distillation.runners import HFDistilRunner
from compressors.metrics.hf_metric import HFMetric
from compressors.runners.hf_runner import HFRunner


def test_hf():
    """tests example pipeline"""
    datasets = load_dataset("ag_news")
    tokenizer = AutoTokenizer.from_pretrained("google/bert_uncased_L-4_H-128_A-2")
    datasets = datasets.map(
        lambda e: tokenizer(e["text"], truncation=True, padding="max_length", max_length=128),
        batched=True,
    )
    datasets = datasets.map(lambda e: {"labels": e["label"]}, batched=True)
    datasets.set_format(
        type="torch", columns=["input_ids", "token_type_ids", "attention_mask", "labels"],
    )
    loaders = {
        "train": DataLoader(datasets["train"], batch_size=32, shuffle=True),
        "valid": DataLoader(datasets["test"], batch_size=32),
    }
    metric_callback = LoaderMetricCallback(
        metric=HFMetric(metric=load_metric("accuracy")), input_key="logits", target_key="labels",
    )
    teacher_model = AutoModelForSequenceClassification.from_pretrained(
        "google/bert_uncased_L-4_H-128_A-2", num_labels=4
    )
    runner = HFRunner()
    runner.train(
        model=teacher_model,
        loaders=loaders,
        optimizer=torch.optim.Adam(teacher_model.parameters(), lr=1e-4),
        callbacks=[metric_callback],
        num_epochs=3,
        valid_metric="accuracy",
        minimize_valid_metric=False,
        check=True,
    )
    metric_callback = LoaderMetricCallback(
        metric=HFMetric(metric=load_metric("accuracy")), input_key="s_logits", target_key="labels",
    )

    slct_callback = ControlFlowCallback(
        HiddenStatesSelectCallback(hiddens_key="t_hidden_states", layers=[1, 3]), loaders="train",
    )

    lambda_hiddens_callback = ControlFlowCallback(
        LambdaSelectCallback(
            lambda s_hiddens, t_hiddens: (
                [c_s[:, 0] for c_s in s_hiddens],
                [t_s[:, 0] for t_s in t_hiddens],  # tooks only CLS token
            )
        ),
        loaders="train",
    )

    mse_hiddens = ControlFlowCallback(MSEHiddenStatesCallback(), loaders="train")

    kl_div = ControlFlowCallback(KLDivCallback(), loaders="train")

    aggregator = ControlFlowCallback(
        MetricAggregationCallback(
            prefix="loss",
            metrics={"kl_div_loss": 0.2, "mse_loss": 0.2, "task_loss": 0.6},
            mode="weighted_sum",
        ),
        loaders="train",
    )

    runner = HFDistilRunner()

    student_model = AutoModelForSequenceClassification.from_pretrained(
        "google/bert_uncased_L-2_H-128_A-2", num_labels=4
    )
    runner.train(
        model=torch.nn.ModuleDict({"teacher": teacher_model, "student": student_model}),
        loaders=loaders,
        optimizer=torch.optim.Adam(student_model.parameters(), lr=1e-4),
        callbacks=[
            metric_callback,
            slct_callback,
            lambda_hiddens_callback,
            mse_hiddens,
            kl_div,
            aggregator,
            OptimizerCallback(metric_key="loss"),
        ],
        check=True,
        num_epochs=3,
        valid_metric="accuracy",
        minimize_valid_metric=False,
        valid_loader="valid",
    )
