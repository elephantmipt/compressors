from catalyst.runners import Runner
from catalyst.utils import get_nn_from_ddp_module, set_requires_grad


class HFDistilRunner(Runner):
    """Simple runner for transformer model.
    """

    def handle_batch(self, batch):
        model = get_nn_from_ddp_module(self.model)
        student, teacher = model["student"], model["teacher"]
        if self.is_train_loader:
            teacher.eval()
            set_requires_grad(teacher, False)
            t_outputs = teacher(**batch, output_hidden_states=True, return_dict=True)

        s_outputs = student(**batch, output_hidden_states=True, return_dict=True)
        if self.is_train_loader:
            self.batch["t_logits"] = t_outputs["logits"]
            self.batch["t_hidden_states"] = t_outputs["hidden_states"]
        self.batch_metrics["task_loss"] = s_outputs["loss"]
        self.batch["s_logits"] = s_outputs["logits"]
        self.batch["s_hidden_states"] = s_outputs["hidden_states"]


__all__ = ["HFDistilRunner"]
