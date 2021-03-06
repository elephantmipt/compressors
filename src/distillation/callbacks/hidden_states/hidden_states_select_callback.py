from typing import Union, List

from catalyst.core import Callback
from src.distillation.callbacks.order import CallbackOrder
from src.distillation.hidden_states import hidden_states_select


class HiddenStatesSelectCallback(Callback):
    def __init__(
        self, layers: Union[int, List[int]], hiddens_key: str = "t_hidden_states"
    ):
        super().__init__(order=CallbackOrder.hiddens_slct)
        self.layers = layers
        self.hiddens_key = hiddens_key

    def on_batch_end(self, runner):
        runner.batch[self.hiddens_key] = hidden_states_select(
            hidden_states=runner.batch[self.hiddens_key], layers=self.layers
        )


__all__ = ["HiddenStatesSelectCallback"]
