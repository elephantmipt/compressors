# BERT Distillation

First of all you should clone repo:

```bash
git clone https://github.com/elephantmipt/compressors.git && cd compressors
```

## Teacher training

To train teacher you can run
```bash
python examples/transformers/train_teacher
```
Or you can run sweep with Weights and Biases
```bash
wandb sweep examples/transformers/configs/teacher.yml
```

I trained two models with this script on AG News dataset:

| [BERT Medium](https://huggingface.co/google/bert_uncased_L-8_H-512_A-8) | [BERT Small](https://huggingface.co/google/bert_uncased_L-4_H-512_A-8) |
| ----------- | ---------- |
| [0.9454](https://wandb.ai/torchwave/bert-distillation/sweeps/0xtkz0gf)      | [0.9429](https://wandb.ai/torchwave/bert-distillation/sweeps/8wyq3t4x) |

## Student training

After teacher training you can use it to train student network.

```bash
python examples/transformers/train_student.py --teacher-path bert_teacher/checkpoints/best.pth
```

Or you can use Weights and Biases for parameter tuning

```bash
wandb sweep examples/transformers/configs/student.yml
```

Here are the logs for models I trained:

| Teacher | Student | Accuracy | Accuracy (without kd) | Improvement |
| ------- | ------- | -------- | --------------------- | ----------- |
| BERT Medium | BERT Small | [0.9443](https://wandb.ai/torchwave/bert-distillation/sweeps/68nz4bmr) | 0.9429 | +0.014 |
| BERT Medium | BERT Medium | [0.9466](https://wandb.ai/torchwave/bert-distillation/sweeps/mpcocdp6) | 0.9454 | +0.012 |
