from typing import Dict
from catalyst.core import Callback
from compressors.distillation.callbacks.order import CallbackOrder


class MetricAggregationCallback(Callback):
    def __init__(self, weights: Dict[str, float], output_key: str = "loss"):
        super().__init__(order=CallbackOrder.MetricAggregation)
        self.weights = weights
        self.output_key = output_key

    def on_batch_end(self, runner):
        aggregated = 0.0
        for metric_key, metric_value in self.weights.items():
            aggregated += metric_value * runner.batch_metrics[metric_key]
        runner.batch_metrics[self.output_key] = aggregated


__all__ = ["MetricAggregationCallback"]
