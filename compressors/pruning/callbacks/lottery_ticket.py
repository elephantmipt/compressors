from typing import Any, Dict

from catalyst.core import Callback, CallbackOrder, IRunner


class LotteryTicketCallback(Callback):
    def __init__(self, initial_state_dict: Dict[str, Any]):
        """
        Reinitialize model with initial state dict on stage end.
        Args:
            initial_state_dict: initial model state dict
        """
        super(LotteryTicketCallback, self).__init__(order=CallbackOrder.ExternalExtra)
        self.initial_state_dict = initial_state_dict

    @staticmethod
    def merge_state_dict(start_dict, new_dict):
        out_dict = {}
        for k, v in new_dict.items():
            if "mask" in k:
                out_dict[k] = v
            else:
                out_dict[k] = start_dict[k]
        return out_dict

    def on_stage_start(self, runner: "IRunner") -> None:
        """
        Event handler.

        Args:
            runner: experiment runner
        """
        state_dict = self.merge_state_dict(
            self.initial_state_dict, runner.model.state_dict()
        )
        runner.model.load_state_dict(state_dict)


__all__ = ["LotteryTicketCallback"]
