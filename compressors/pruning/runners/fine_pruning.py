from collections import OrderedDict

from catalyst.dl import Callback
from compressors.pruning.runners.pruning import PruneRunner
from compressors.pruning.callbacks.prepare_for_pruning_callback import PrepareForFinePruningCallback


class FinePruneRunner(PruneRunner):
    @property
    def stages(self):
        return ["preparing"] + [f"pruning_session_{i}" for i in range(self._num_sessions)]

    def get_stage_len(self, stage: str) -> int:
        if stage == "preparing":
            return 0
        return self._num_epochs

    def get_callbacks(self, stage: str) -> "OrderedDict[str, Callback]":
        callbacks = super().get_callbacks(stage=stage)
        if stage == "preparing":
            callbacks["preparing"] = PrepareForFinePruningCallback()
        return callbacks
