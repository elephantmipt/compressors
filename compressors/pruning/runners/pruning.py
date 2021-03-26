from typing import Any, Mapping
from collections import OrderedDict

from catalyst.dl import Callback, Runner


class PruneRunner(Runner):
    def __init__(self, num_sessions: int = 5, *runner_args, **runner_kwargs):
        """
        Base runner for iterative pruning.
        Args:
            num_sessions: number of pruning sessions
            *runner_args: runner args
            **runner_kwargs: runner kwargs
        """
        super().__init__(*runner_args, **runner_kwargs)
        self._num_sessions = num_sessions

    @property
    def stages(self):
        return [f"pruning_session_{i}" for i in range(self._num_sessions)]

    def handle_batch(self, batch: Mapping[str, Any]) -> None:
        self.batch["logits"] = self.model(batch["features"])


__all__ = ["PruneRunner"]
