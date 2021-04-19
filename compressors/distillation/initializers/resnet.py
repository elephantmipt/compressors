import torch


def init_with_first_layers(student: torch.nn.Module, teacher: torch.nn.Module) -> None:
    """
    Initialize student with first layers of the teacher in block.

    Args:
        student: student network to be initialized
        teacher: teacher network
    """
    student.load_state_dict(teacher.state_dict(), strict=False)


def init_with_max_magnitude_kernels(
    student: torch.nn.Module,
    teacher: torch.nn.Module,
    norm: str = "fro"
):
    """
    Initialize student with max magnitude kernels from teacher

    Args:
        student: student network to be initialized
        teacher: teacher network
        norm: "fro" for Frobenius norm or "nuc" for nuclear norm.
    """
    first_conv_kernels = _get_kernels_from_block(teacher.layer1, "conv1")
    magnitudes = torch.linalg.norm(first_conv_kernels, ord=norm, dim=(3, 4))
    argsorted = torch.argsort(magnitudes, dim=0)




def _get_kernels_from_block(block, attribute: str) -> torch.Tensor:
    result = None
    for layer in block:
        conv = getattr(layer, attribute)
        weight = conv.weight.data
        if result is None:
            result = weight
        else:
            result = torch.cat((result, weight), dim=0)
    return result





