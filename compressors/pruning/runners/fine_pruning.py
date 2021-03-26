from collections import OrderedDict

from catalyst.dl import Callback
from compressors.pruning.runners.pruning import PruneRunner
from compressors.pruning.callbacks.prepare_for_pruning_callback import PrepareForFinePruningCallback


class FinePruneRunner(PruneRunner):
    def get_callbacks(self, stage: str) -> "OrderedDict[str, Callback]":
        if stage == "prepare":
            return OrderedDict([("prepare_fine_pruning", PrepareForFinePruningCallback())])
        return super().get_callbacks(stage=stage)
