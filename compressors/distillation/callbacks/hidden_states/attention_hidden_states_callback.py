from compressors.distillation.losses import AttentionLoss
from compressors.distillation.callbacks.order import CallbackOrder
from catalyst.core import Callback


class AttentionHiddenStatesCallback(Callback):
    """


    Args:
        output_key: name for loss. Defaults to attention_loss.
        exclude_first_and_last: If set to True takes only last hidden state.
                Usually cosine loss applied in this way. Defaults to True.
    """
    def __init__(
        self,
        output_key: str = "attention_loss",
        exclude_first_and_last: bool = True,
        p: int = 2,
    ):
        """
        Cosine loss for difference between hidden states of teacher and student model.

        Args:
             output_key: name for loss. Defaults to cosine_loss.
             exclude_first_and_last: If set to True takes only last hidden state.
                Usually cosine loss applied in this way. Defaults to True.
        """
        super().__init__(order=CallbackOrder.Metric)
        self.output_key = output_key
        self.exclude_first_and_last = exclude_first_and_last
        self.criterion = AttentionLoss(p=p)

    def on_batch_end(self, runner):
        s_hiddens = runner.batch["s_hidden_states"]
        t_hiddens = runner.batch["t_hidden_states"]
        if self.exclude_first_and_last:
            s_hiddens = s_hiddens[1:-1]
            t_hiddens = t_hiddens[1:-1]
        runner.batch_metrics[self.output_key] = self.criterion(
            s_hiddens, t_hiddens
        )


__all__ = ["AttentionHiddenStatesCallback"]
