from catalyst.runners import Runner


class HFRunner(Runner):
    def handle_batch(self, batch):
        outputs = self.model(**batch, return_dict=True)
        self.batch_metrics["loss"] = outputs["loss"]
        self.batch["logits"] = outputs["logits"]


__all__ = ["HFRunner"]
