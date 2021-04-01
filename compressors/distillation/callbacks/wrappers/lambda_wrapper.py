from typing import Callable, List, Union
from copy import deepcopy

from catalyst.core import Callback


class LambdaWrapperCallback(Callback):
    """Wraps input for your callback with specified function.

        Args:
            base_callback (Callback): Base callback.
            lambda_fn (Callable): Function to apply.
            keys_to_apply (Union[List[str], str], optional): Keys in batch dict to apply function.
                Defaults to ["s_hidden_states", "t_hidden_states"].

        Raises:
            TypeError: When keys_to_apply is not str or list.
    """

    def __init__(
        self,
        base_callback: Callback,
        lambda_fn: Callable,
        keys_to_apply: Union[List[str], str] = ["s_hidden_states", "t_hidden_states"],
    ):
        """Wraps input for your callback with specified function.

        Args:
            base_callback (Callback): Base callback.
            lambda_fn (Callable): Function to apply.
            keys_to_apply (Union[List[str], str], optional): Keys in batch dict to apply function.
                Defaults to ["s_hidden_states", "t_hidden_states"].

        Raises:
            TypeError: When keys_to_apply is not str or list.
        """
        super().__init__(order=base_callback.order)
        self.base_callback = base_callback
        if not isinstance(keys_to_apply, (list, str)):
            raise TypeError("keys to apply should be str or list of str.")
        self.keys_to_apply = keys_to_apply
        self.lambda_fn = lambda_fn

    def on_batch_end(self, runner):
        orig_batch = deepcopy(runner.batch)
        batch = runner.batch

        if isinstance(self.keys_to_apply, list):
            fn_inp = [batch[key] for key in self.keys_to_apply]
            fn_output = self.lambda_fn(*fn_inp)
            if isinstance(fn_output, tuple):
                for idx, key in enumerate(self.keys_to_apply):
                    batch[key] = fn_output[idx]
            elif isinstance(fn_output, dict):
                for outp_k, outp_v in fn_output.items():
                    batch[outp_k] = outp_v
            else:
                raise Exception(
                    "If keys_to_apply is list, then function output should be tuple or dict."
                )
        elif isinstance(self.keys_to_apply, str):
            batch[self.keys_to_apply] = self.lambda_fn(self.keys_to_apply)
        runner.batch = batch
        self.base_callback.on_batch_end(runner)
        runner.batch = orig_batch


__all__ = ["LambdaWrapperCallback"]
