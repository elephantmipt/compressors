from typing import Any, Mapping, Union, List

from catalyst.dl import Runner


class PruneRunner(Runner):
    def __init__(
        self,
        num_sessions: int = 5,
        input_key: Union[str, List[str]] = "features",
        output_key: Union[str, List[str]] = "logits",
        *runner_args,
        **runner_kwargs
    ):
        """
        Base runner for iterative pruning.
        Args:
            num_sessions: number of pruning sessions
            *runner_args: runner args
            **runner_kwargs: runner kwargs
        """
        super().__init__(*runner_args, **runner_kwargs)
        if isinstance(input_key, str):
            input_key = [input_key]
        if isinstance(output_key, str):
            output_key = [output_key]

        self.input_key = input_key
        self.output_key = output_key
        self._num_sessions = num_sessions

    @property
    def stages(self):
        return [f"pruning_session_{i}" for i in range(self._num_sessions)]

    def handle_batch(self, batch: Mapping[str, Any]) -> None:
        model_inp = [batch[key] for key in self.input_key]
        model_out = self.model(*model_inp)
        for idx, key in enumerate(self.output_key):
            self.batch[key] = model_out[idx]


__all__ = ["PruneRunner"]
