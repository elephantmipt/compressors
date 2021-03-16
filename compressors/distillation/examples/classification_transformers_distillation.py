from tqdm.autonotebook import trange, tqdm

import torch
from torch.utils.data import DataLoader

from datasets import load_dataset, load_metric
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from compressors.distillation.losses import (
    MSEHiddenStatesLoss,
    KLDivLoss,
)

num_epochs = 5
device = "cuda" if torch.cuda.is_available() else "cpu"

datasets = load_dataset("ag_news")
metric_fn = load_metric("accuracy")

tokenizer = AutoTokenizer.from_pretrained("google/bert_uncased_L-4_H-512_A-8")

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

teacher_model = AutoModelForSequenceClassification.from_pretrained(
    "google/bert_uncased_L-4_H-512_A-8", num_labels=4
).to(device)
teacher_model.load_state_dict(torch.load("teacher.pth"))

student_model = AutoModelForSequenceClassification.from_pretrained(
    "google/bert_uncased_L-2_H-128_A-2", num_labels=4
).to(device)

kl_div_loss = KLDivLoss()
mse_hiddens_loss = MSEHiddenStatesLoss(
    normalize=True,
    need_mapping=True,
    teacher_hidden_state_dim=512,
    student_hidden_state_dim=128,
    num_layers=2,
).to(device)

optimizer = torch.optim.Adam(student_model.parameters(), lr=1e-4)
mapping_optimizer = torch.optim.Adam(mse_hiddens_loss.parameters(), lr=1e-3)


def train_iter(
    student,
    teacher,
    batch,
    logits_criterion,
    hiddens_criterion,
    weights,
    optimizer,
    mapping_optimizer,
    metric_fn,
):
    student.train()
    teacher.eval()

    with torch.no_grad():
        teacher_outputs = teacher(**batch, output_hidden_states=True, retrun_dict=True)
    student_outputs = student(**batch, output_hidden_states=True, retrun_dict=True)

    predictions = student_outputs["logits"].argmax(-1).detach().cpu().numpy()
    metric_fn.add_batch(predictions=predictions, references=batch["labels"])

    task_loss = student_outputs["loss"]

    logits_loss = logits_criterion(student_outputs["logits"], teacher_outputs["logits"])

    t_hiddens = teacher_outputs["hidden_states"]
    t_hiddens = tuple([t_h[:, 0] for t_h in t_hiddens])
    s_hiddens = student_outputs["hidden_states"]
    s_hiddens = tuple([s_h[:, 0] for s_h in s_hiddens])
    hiddens_loss = hiddens_criterion(s_hiddens, t_hiddens)

    final_loss = weights[0] * task_loss + weights[1] * logits_loss, weights[2] * hiddens_loss

    optimizer.zero_grad()
    mapping_optimizer.zero_grad()
    final_loss.backward()
    optimizer.step()
    mapping_optimizer.step()
    return {
        "final_loss": final_loss.item(),
        "task_loss": task_loss.item(),
        "logits_loss": logits_loss.iten(),
        "hidden_state_loss": hiddens_loss.item(),
    }


@torch.no_grad()
def val_iter(student, batch, metric_fn):
    student.eval()
    loss, logits = student(**batch, output_hidden_states=False)
    predictions = logits.argmax(-1).detach().cpu().numpy()
    metric_fn.add_batch(predictions=predictions, references=batch["labels"])


def dict_to_device(batch, device):
    return {k: v.to(device) for k, v in batch.items()}


pbar_epochs = trange(num_epochs, leave=False)
pbar_epochs.set_description("Epoch: ")
best_accuracy = 0
for epoch in pbar_epochs:
    for loader_key, loader in loaders.items():
        pbar_loader = tqdm(loader, leave=False)
        for batch in pbar_loader:
            batch = dict_to_device(batch, device)
            if loader_key.startswith("train"):
                metrics = train_iter(
                    student=student_model,
                    teacher=teacher_model,
                    batch=batch,
                    logits_criterion=kl_div_loss,
                    hiddens_criterion=mse_hiddens_loss,
                    weights=[0.6, 0.3, 0.1],
                    optimizer=optimizer,
                    mapping_optimizer=mapping_optimizer,
                    metric_fn=metric_fn,
                )
            else:
                metrics = val_iter(student=student_model, batch=batch, metric_fn=metric_fn,)
            log_str = " ".join([f"{key}: {met:.3f}" for key, met in metrics.items()])
            pbar_loader.set_description(log_str)
        accuracy = metric_fn.compute()["accuracy"]
        if not loader_key.startswith("train"):
            if accuracy > best_accuracy:
                torch.save(student_model.state_dict(), "best_student.pth")
                best_accuracy = accuracy
        print(f"{loader_key} accuracy: {accuracy}")
