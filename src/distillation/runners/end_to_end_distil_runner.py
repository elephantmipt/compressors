from typing import Mapping, Any, Union, Callable, Dict

from catalyst.callbacks import ControlFlowCallback
from catalyst.runners import Runner
from catalyst.utils import get_nn_from_ddp_module, set_requires_grad

from src.distillation.losses import mse_loss, pkt_loss, cosine_loss, kl_div_loss
from src.distillation.callbacks import MetricAggregationCallback


NAME2LOSS = {
    "mse_loss": mse_loss,
    "pkt_loss": pkt_loss,
    "cosine_loss": cosine_loss
}

NAME2LOGITSLOSS = {
    "kl_loss": kl_div_loss
}


class EndToEndDistilRunner(Runner):
    """
    End to end runner for distillation. Can be used without additional callbacks.

    Args:
        hidden_state_loss: loss between last hidden states. Defaults to None.
        logits_diff_loss: loss between logits. Defaults to kl_loss.
        loss_weights: weights for losses in final sum.
        num_train_teacher_epochs: If set to None, then just distilling the knowledge.
            If int then train teacher model for provided number of epochs. Defaults to None.
        *runner_args: runner args
        **runner_kwargs: runner kwargs
    """
    def __init__(
        self,
        hidden_state_loss: Union[str, Callable] = None,
        logits_diff_loss: Union[str, Callable] = "kl_loss",
        loss_weights: Dict = None,
        num_train_teacher_epochs: int = None,
        *runner_args,
        **runner_kwargs
    ):
        """
        End to end runner for distillation. Can be used without additional callbacks.

        Args:
            hidden_state_loss: loss between last hidden states. Defaults to None.
            logits_diff_loss: loss between logits. Defaults to kl_loss.
            loss_weights: weights for losses in final sum.
            num_train_teacher_epochs: If set to None, then just distilling the knowledge.
                If int then train teacher model for provided number of epochs. Defaults to None.
            *runner_args: runner args
            **runner_kwargs: runner kwargs
        """
        super(EndToEndDistilRunner, self).__init__(*runner_args, **runner_kwargs)
        self.hidden_state_loss = hidden_state_loss
        self.output_hidden_states = hidden_state_loss is not None
        self.logits_diff_loss = logits_diff_loss
        if loss_weights is None:
            if hidden_state_loss is None:
                if logits_diff_loss is None:
                    loss_weights = {"task_loss": 1.0}
                else:
                    loss_weights = {"task_loss": 0.8, "logits_diff_loss": 0.2}
            else:
                if logits_diff_loss is None:
                    loss_weights = {"task_loss": 0.8, "hidden_state_loss": 0.2}
                else:
                    loss_weights = {
                        "task_loss": 0.6, "logits_diff_loss": 0.2, "hidden_state_loss": 0.2
                    }
        self.loss_weights = loss_weights
        self.num_train_teacher_epochs = num_train_teacher_epochs
        if self.hidden_state_loss is not None:
            if isinstance(hidden_state_loss, Callable):
                self.hidden_state_loss_fn = hidden_state_loss
            elif isinstance(hidden_state_loss, str):
                self.hidden_state_loss_fn = self.get_hidden_state_loss(hidden_state_loss)
            else:
                raise TypeError("Hidden state loss should be string or function")
        if self.logits_diff_loss is not None:
            if isinstance(logits_diff_loss, Callable):
                self.logits_diff_loss_fn = logits_diff_loss
            elif isinstance(logits_diff_loss, str):
                self.logits_diff_loss_fn = self.get_logits_diff_loss(logits_diff_loss)
            else:
                raise TypeError("Logits diff loss should be string or function")

    @property
    def stages(self):
        if self.num_train_teacher_epochs is not None:
            return ["teacher_training", "distillation"]
        else:
            return ["distillation"]

    def get_stage_len(self, stage: str) -> int:
        if stage == "distillation":
            return self._num_epochs
        elif stage == "teacher_training":
            return self.num_train_teacher_epochs

    def get_callbacks(self, stage: str) -> "OrderedDict[str, Callback]":
        callbacks = super().get_callbacks(stage)
        if stage == "distillation":
            callbacks["_aggregation"] = ControlFlowCallback(
                MetricAggregationCallback(weights=self.loss_weights),
                loaders="train"
            )
        return callbacks

    def handle_batch(self, batch: Mapping[str, Any]) -> None:
        if self.stage_key == "teacher_training":
            self._handle_batch_teacher_training(batch)
        elif self.stage_key == "distillation":
            self._handle_batch_distillation(batch)

    def _handle_batch_teacher_training(self, batch):
        model = get_nn_from_ddp_module(self.model)
        teacher = model["teacher"]
        t_logits = teacher(batch["features"])
        loss = self.criterion(t_logits, batch["targets"])
        self.batch["logits"] = t_logits
        self.batch_metrics["loss"] = loss

    def _handle_batch_distillation(self, batch):
        model = get_nn_from_ddp_module(self.model)
        student, teacher = model["student"], model["teacher"]
        if self.is_train_loader:
            teacher.eval()
            set_requires_grad(teacher, False)
            t_outputs = teacher(
                batch["features"],
                output_hidden_states=self.output_hidden_states,
                return_dict=True,
            )
        s_outputs = student(
            batch["features"], output_hidden_states=self.output_hidden_states, return_dict=True,
        )
        self.batch["s_logits"] = s_outputs["logits"]
        if self.is_train_loader:
            self.batch["t_logits"] = t_outputs["logits"]
            if self.logits_diff_loss is not None:
                self.batch_metrics["logits_diff_loss"] = self.logits_diff_loss_fn(
                    self.batch["s_logits"],
                    self.batch["t_logits"]
                )
        if self.output_hidden_states and self.is_train_loader:
            self.batch["student_hidden_states"] = s_outputs["hidden_states"]
            self.batch["teacher_hidden_states"] = t_outputs["hidden_states"]
            if self.hidden_state_loss is not None:
                student_hidden_state = self.batch["student_hidden_states"][-1]
                teacher_hidden_state = self.batch["teacher_hidden_states"][-1]
                self.batch_metrics["hidden_state_loss"] = self.hidden_state_loss_fn(
                    student_hidden_state, teacher_hidden_state
                )
        self.batch_metrics["task_loss"] = self.criterion(batch["s_logits"], batch["targets"])
        self.batch["logits"] = self.batch["s_logits"]  # for accuracy callback or other metric callback

    @staticmethod
    def get_hidden_state_loss(loss_name: str):
        if loss_name in NAME2LOSS.keys():
            return NAME2LOSS[loss_name]
        else:
            raise TypeError(f"Hidden state loss should be in {NAME2LOSS.keys()}")

    @staticmethod
    def get_logits_diff_loss(loss_name: str):
        if loss_name in NAME2LOGITSLOSS.keys():
            return NAME2LOGITSLOSS[loss_name]
        else:
            raise TypeError(f"Loggits diff loss should be in {NAME2LOGITSLOSS.keys()}")

    def predict_batch(self, batch: Mapping[str, Any], **kwargs) -> Mapping[str, Any]:
        return self.model["student"](batch["features"])
