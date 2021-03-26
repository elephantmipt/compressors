from typing import Mapping, Tuple, Union

from torch.nn import Module


class BaseDistilModel(Module):
    """Base model for running knoweledge distillation"""

    def forward(
        self, output_hidden_states: bool = False, return_dict: bool = False, *args, **kwargs,
    ) -> Union[Tuple, Mapping]:
        """Forward method for model.

        Args:
            output_hidden_states (bool, optional): If true adds hidden states to output.
                Defaults to False.
            return_dict (bool, optional): If true returns dict else tuple. Defaults to False.

        Returns:
            Union[Tuple, Mapping]: Model outputs.
        """


__all__ = ["BaseDistilModel"]
