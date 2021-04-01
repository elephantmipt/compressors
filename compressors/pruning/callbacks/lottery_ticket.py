from typing import Any, Dict

from catalyst.core import Callback, CallbackOrder, IRunner


class LotteryTicketCallback(Callback):
    def __init__(self, initial_state_dict: Dict[str, Any]):
        """
        Reinitialize model with initial state dict
        Args:
            initial_state_dict:
        """
        super(LotteryTicketCallback, self).__init__(order=CallbackOrder.Internal)
        self.initial_state_dict = initial_state_dict

    def on_stage_start(self, runner: "IRunner") -> None:
        """
        Event handler.

        Args:
            runner: experiment runner
        """
        runner.model.load_state_dict(self.initial_state_dict)
