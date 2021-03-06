from catalyst.metrics import ICallbackLoaderMetric


class HFMetric(ICallbackLoaderMetric):
    def __init__(
        self,
        metric,
        regression: bool = False,
        compute_on_call: bool = True,
        prefix: str = None,
        suffix: str = None,
    ):
        super().__init__(
            compute_on_call=compute_on_call, prefix=prefix, suffix=suffix,
        )

        self.metric = metric
        self.regression = regression

    def reset(self):
        try:
            self.metric.compute()
        except ValueError:
            pass

    def update(self, logits, labels):
        logits = logits.detach().cpu().numpy()
        labels = labels.cpu().numpy()
        predictions = logits if self.regression else logits.argmax(-1)
        self.metric.add_batch(predictions=predictions, references=labels)

    def compute_key_value(self):
        return self.metric.compute()

    def compute(self):
        return self.metric.compute()
