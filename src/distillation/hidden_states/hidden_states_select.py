from typing import List, Union, Tuple

import torch
from torch import FloatTensor


def hidden_states_select(
    hidden_states: Tuple[FloatTensor], layers: Union[int, List[int]]
) -> FloatTensor:
    """Selects specified layers.

    Args:
        t_hidden_states (FloatTensor): teacher hiddens.
        layers (Union[int, List[int]]): list of layers numbers.

    Returns:
        FloatTensor: Selected hidden states
    """

    if isinstance(layers, list):
        return tuple([hidden_states[l_idx] for l_idx in layers])
    return hidden_states[layers]


__all__ = ["hidden_states_select"]
