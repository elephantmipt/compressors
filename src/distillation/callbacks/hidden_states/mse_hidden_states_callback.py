from src.distillation.losses import mse_loss
from src.distillation.callbacks.order import CallbackOrder
from catalyst.core import Callback


class MSEHiddenStatesCallback(Callback):
    def __init__(self, output_key: str = "mse_loss"):
        super().__init__(order=CallbackOrder.Metric)
        self.output_key = output_key

    def on_batch_end(self, runner):
        runner.batch_metrics[self.output_key] = mse_loss(
            s_hidden_states=runner.batch["s_hidden_states"],
            t_hidden_states=runner.batch["t_hidden_states"],
        )


__all__ = ["MSEHiddenStatesCallback"]
