from src.distillation.losses import kl_div_loss
from src.distillation.callbacks.order import CallbackOrder
from catalyst.core import Callback


class KLDivCallback(Callback):
    def __init__(self, output_key: str = "kl_div_loss"):
        super().__init__(order=CallbackOrder.Metric)
        self.output_key = output_key

    def on_batch_end(self, runner):
        runner.batch_metrics[self.output_key] = kl_div_loss(
            s_logits=runner.batch["s_logits"],
            t_logits=runner.batch["t_logits"],
        )


__all__ = ["KLDivCallback"]
