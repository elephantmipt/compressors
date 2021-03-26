from collections import OrderedDict
from typing import Mapping, Any

from catalyst.dl import Runner, Callback


class PruneRunner(Runner):
    def __init__(
            self,
            num_sessions: int = 5,
            num_epochs: int = 20,
            *runner_args,
            **runner_kwargs
    ):
        """
        Base runner for iterative pruning.
        Args:
            num_sessions: number of pruning sessions
            num_epochs: num epochs to tune every session
            *runner_args: runner args
            **runner_kwargs: runner kwargs
        """
        super().__init__(*runner_args, **runner_kwargs)
        self._num_epochs = num_epochs
        self._num_sessions = num_sessions

    @property
    def stages(self):
        return ["preparing"] + [f"pruning_session_{i}" for i in range(self._num_sessions)]

    def get_callbacks(self, stage: str) -> "OrderedDict[str, Callback]":
        if stage == "preparing":
            return OrderedDict([])
        return super().get_callbacks(stage=stage)

    def get_stage_len(self, stage: str) -> int:
        if stage == "preparing":
            return 0
        else:
            return self._num_epochs

    def handle_batch(self, batch: Mapping[str, Any]) -> None:
        self.batch["logits"] = self.model(batch["features"])


__all__ = ["PruneRunner"]
