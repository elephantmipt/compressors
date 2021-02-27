import torch
from torch import FloatTensor, LongTensor
from torch import nn


def cosine_loss(
    s_hidden_states: FloatTensor,
    t_hidden_states: FloatTensor,
    attention_mask: LongTensor = None,
) -> FloatTensor:
    """Cosine loss between hidden states.

    Args:
        s_hidden_states (FloatTensor): student hiddens
        t_hidden_states (FloatTensor): teacher hiddens
        attention_mask (LongTensor, optional): attention mask if you are using transformers.
            Defaults to None.

    Returns:
        FloatTensor: [description]
    """
    if attention_mask is not None:
        # HF transformers case
        return _cosine_loss_hf(
            s_hidden_states=s_hidden_states,
            t_hidden_states=t_hidden_states,
            attention_mask=attention_mask,
        )

    loss_fn = nn.CosineEmbeddingLoss()
    hidden_dim = s_hidden_states.size(-1)
    s_hidden_states = s_hidden_states.reshape(-1, hidden_dim)
    t_hidden_states = t_hidden_states.reshape(-1, hidden_dim)
    assert s_hidden_states.shape == t_hidden_states.shape
    target = torch.ones(t_hidden_states.size(0))
    return loss_fn(s_hidden_states, t_hidden_states, target)


def _cosine_loss_hf(
    s_hidden_states: FloatTensor,
    t_hidden_states: FloatTensor,
    attention_mask: LongTensor,
):
    mask = attention_mask.unsqueeze(-1).expand_as(
        s_hidden_states
    )  # (bs, seq_length, dim)
    assert s_hidden_states.size() == t_hidden_states.size()
    dim = s_hidden_states.size(-1)

    s_hidden_states_slct = torch.masked_select(
        s_hidden_states, mask
    )  # (bs * seq_length * dim)
    s_hidden_states_slct = s_hidden_states_slct.view(-1, dim)  # (bs * seq_length, dim)
    t_hidden_states_slct = torch.masked_select(
        t_hidden_states, mask
    )  # (bs * seq_length * dim)
    t_hidden_states_slct = t_hidden_states_slct.view(-1, dim)  # (bs * seq_length, dim)

    target = s_hidden_states_slct.new(s_hidden_states_slct.size(0)).fill_(
        1
    )  # (bs * seq_length,)
    loss_cos = nn.CosineEmbeddingLoss()(
        s_hidden_states_slct, t_hidden_states_slct, target
    )
    return loss_cos


__all__ = ["cosine_loss"]
