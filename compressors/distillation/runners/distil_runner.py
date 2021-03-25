from catalyst.utils import get_nn_from_ddp_module, set_requires_grad
from catalyst.runners import Runner

from compressors.distillation.data.label_smoothing import probability_shift


class DistilRunner(Runner):
    def __init__(
        self,
        output_hidden_states: bool = True,
        apply_probability_shift: bool = False,
        *runner_args,
        **runner_kwargs
    ):
        super().__init__(*runner_args, **runner_kwargs)
        self.output_hidden_states = output_hidden_states
        self.apply_probability_shift = apply_probability_shift

    def handle_batch(self, batch):
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
            if self.apply_probability_shift:
                self.batch["t_logits"] = probability_shift(
                    logits=self.batch["t_logits"], labels=self.batch["targets"]
                )
        if self.output_hidden_states:
            self.batch["s_hidden_states"] = s_outputs["hidden_states"]
            if self.is_train_loader:
                self.batch["t_hidden_states"] = t_outputs["hidden_states"]


__all__ = ["DistilRunner"]
