from catalyst.core import Callback

from compressors.distillation.callbacks.order import CallbackOrder
from compressors.distillation.losses import KLDivLoss


class KLDivCallback(Callback):
    def __init__(
        self,
        output_key: str = "kl_div_loss",
        temperature: float = 1.0,
        student_logits_key: str = "s_logits",
        teacher_logits_key: str = "t_logits",
    ):
        super().__init__(order=CallbackOrder.Metric)
        self.output_key = output_key
        self.criterion = KLDivLoss(temperature=temperature)
        self.teacher_logits_key = teacher_logits_key
        self.student_logits_key = student_logits_key

    def on_batch_end(self, runner):
        runner.batch_metrics[self.output_key] = self.criterion(
            s_logits=runner.batch[self.student_logits_key],
            t_logits=runner.batch[self.teacher_logits_key],
        )


__all__ = ["KLDivCallback"]
