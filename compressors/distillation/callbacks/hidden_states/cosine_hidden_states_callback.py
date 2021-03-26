from catalyst.core import Callback

from compressors.distillation.callbacks.order import CallbackOrder
from compressors.distillation.losses import CosineHiddenStateLoss


class CosineHiddenStatesCallback(Callback):
    """
    Cosine loss for difference between hidden states of teacher and student model.

    Args:
        output_key: name for loss. Defaults to cosine_loss.
        last_only: If set to True takes only last hidden state.
                Usually cosine loss applied in this way. Defaults to True.
    """

    def __init__(
        self,
        output_key: str = "cosine_loss",
        last_only: bool = True,
        need_mapping: bool = False,
        teacher_hidden_state_dim: int = None,
        student_hidden_state_dim: int = None,
    ):
        """
        Cosine loss for difference between hidden states of teacher and student model.

        Args:
             output_key: name for loss. Defaults to cosine_loss.
             last_only: If set to True takes only last hidden state.
                Usually cosine loss applied in this way. Defaults to True.
        """
        super().__init__(order=CallbackOrder.Metric)
        self.output_key = output_key
        self.last_only = last_only
        self.criterion = CosineHiddenStateLoss(
            need_mapping=need_mapping,
            teacher_hidden_state_dim=teacher_hidden_state_dim,
            student_hidden_state_dim=student_hidden_state_dim,
        )

    def on_batch_end(self, runner):
        s_hiddens = runner.batch["s_hidden_states"]
        t_hiddens = runner.batch["t_hidden_states"]
        if self.last_only:
            s_hiddens = s_hiddens[-1]
            t_hiddens = t_hiddens[-1]
        runner.batch_metrics[self.output_key] = self.criterion(s_hiddens, t_hiddens)


__all__ = ["CosineHiddenStatesCallback"]
