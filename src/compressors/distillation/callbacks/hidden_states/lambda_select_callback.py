from typing import Union, List, Callable

from catalyst.core import Callback
from compressors.distillation.callbacks.order import CallbackOrder


class LambdaSelectCallback(Callback):
    """Filters output with your lambda function. Inplace analog of ``LambdaWrp``.

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
        lambda_fn: Callable,
        keys_to_apply: Union[List[str], str] = ["s_hidden_states", "t_hidden_states"],
    ):
        """Filters output with your lambda function.

        Args:
            lambda_fn (Callable): Function to apply.
            keys_to_apply (Union[List[str], str], optional): Keys in batch dict to apply function.
                Defaults to ["s_hidden_states", "t_hidden_states"].

        Raises:
            TypeError: When keys_to_apply is not str or list.
        """
        super().__init__(order=CallbackOrder.HiddensSlct)
        if not (isinstance(keys_to_apply, list) or isinstance(keys_to_apply, str)):
            raise TypeError("keys to apply should be str or list of str.")
        self.keys_to_apply = keys_to_apply
        self.lambda_fn = lambda_fn

    def on_batch_end(self, runner):

        if isinstance(self.keys_to_apply, list):
            fn_inp = [runner.batch[key] for key in self.keys_to_apply]
            fn_output = self.lambda_fn(*fn_inp)
            if isinstance(fn_output, tuple):
                for idx, key in enumerate(self.keys_to_apply):
                    runner.batch[key] = fn_output[idx]
            elif isinstance(fn_output, dict):
                for outp_k, outp_v in fn_output.items():
                    runner.batch[outp_k] = outp_v
            else:
                raise Exception(
                    "If keys_to_apply is list, then function output should be tuple or dict."
                )
        elif isinstance(self.keys_to_apply, str):
            runner.batch[self.keys_to_apply] = self.lambda_fn(self.keys_to_apply)


__all__ = ["LambdaSelectCallback"]
