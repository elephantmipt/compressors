from .order import CallbackOrder
from catalyst.core import Callback, IRunner


class CriterionOptimizationCallback(Callback):
    def __init__(self, callback_key, optimizer):
        super().__init__(CallbackOrder.optimizer)
        self.callback_key = callback_key
        self.optimizer = optimizer

    def on_stage_start(self, runner: "IRunner") -> None:
        callback = runner.callbacks[self.callback_key]
        criterion = callback.criterion
        runner.engine.sync_device(criterion)
