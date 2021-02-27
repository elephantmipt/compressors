from typing import List
from src.settings import IS_TRANSFORMERS_AVAILABLE

if IS_TRANSFORMERS_AVAILABLE:
    from transformers import BertModel


def init_bert_model_with_teacher(
    student: BertModel, teacher: BertModel, layers_to_transfer: List[int] = None,
) -> BertModel:
    """Initialize student model with teacher layers.

    Args:
        student (BertModel): Student model.
        teacher (BertModel): Teacher model.
        layers_to_transfer (List[int], optional): Defines which layers will be transfered.
            If None then will transfer last layers. Defaults to None.

    Returns:
        BertModel: [description]
    """
    teacher_hidden_size = teacher.config.hidden_size
    student_hidden_size = student.config.hidden_size
    if teacher_hidden_size != student_hidden_size:
        raise Exception("Teacher and student hidden size should be the same")
    teacher_layers_num = teacher.config.num_hidden_layers
    student_layers_num = student.config.num_hidden_layers

    if layers_to_transfer is None:
        layers_to_transfer = list(
            range(teacher_layers_num - student_layers_num, teacher_layers_num)
        )

    prefix_teacher = list(teacher.state_dict().keys())[0].split(".")[0]
    prefix_student = list(student.state_dict().keys())[0].split(".")[0]
    student_sd = _extract_layers(teacher_model=teacher, layers=layers_to_transfer,)
    student.load_state_dict(student_sd)
    return student


def _extract_layers(
    teacher_model: BertModel,
    layers: List[int],
    prefix_teacher="bert",
    prefix_student="bert",
    encoder_name="encoder",
):
    state_dict = teacher_model.state_dict()
    compressed_sd = {}

    # extract embeddings
    for w in ["word_embeddings", "position_embeddings"]:
        compressed_sd[f"{prefix_student}.embeddings.{w}.weight"] = state_dict[
            f"{prefix_teacher}.embeddings.{w}.weight"
        ]
    for w in ["weight", "bias"]:
        compressed_sd[f"{prefix_student}.embeddings.LayerNorm.{w}"] = state_dict[
            f"{prefix_teacher}.embeddings.LayerNorm.{w}"
        ]
    # extract encoder

    for std_idx, teacher_idx in enumerate(layers):
        for w in ["weight", "bias"]:
            compressed_sd[
                f"{prefix_student}.encoder.layer.{std_idx}.attention.q_lin.{w}"  # noqa: E501
            ] = state_dict[
                f"{prefix_teacher}.encoder.layer.{teacher_idx}.attention.self.query.{w}"  # noqa: E501
            ]
            compressed_sd[
                f"{prefix_student}.encoder.layer.{std_idx}.attention.k_lin.{w}"  # noqa: E501
            ] = state_dict[
                f"{prefix_teacher}.encoder.layer.{teacher_idx}.attention.self.key.{w}"  # noqa: E501
            ]
            compressed_sd[
                f"{prefix_student}.encoder.layer.{std_idx}.attention.v_lin.{w}"  # noqa: E501
            ] = state_dict[
                f"{prefix_teacher}.encoder.layer.{teacher_idx}.attention.self.value.{w}"  # noqa: E501
            ]

            compressed_sd[
                f"{prefix_student}.encoder.layer.{std_idx}.attention.out_lin.{w}"  # noqa: E501
            ] = state_dict[
                f"{prefix_teacher}.encoder.layer.{teacher_idx}.attention.output.dense.{w}"  # noqa: E501
            ]
            compressed_sd[
                f"{prefix_student}.encoder.layer.{std_idx}.sa_layer_norm.{w}"  # noqa: E501
            ] = state_dict[
                f"{prefix_teacher}.encoder.layer.{teacher_idx}.attention.output.LayerNorm.{w}"  # noqa: E501
            ]

            compressed_sd[
                f"{prefix_student}.encoder.layer.{std_idx}.ffn.lin1.{w}"  # noqa: E501
            ] = state_dict[
                f"{prefix_teacher}.encoder.layer.{teacher_idx}.intermediate.dense.{w}"  # noqa: E501
            ]
            compressed_sd[
                f"{prefix_student}.encoder.layer.{std_idx}.ffn.lin2.{w}"  # noqa: E501
            ] = state_dict[
                f"{prefix_teacher}.encoder.layer.{teacher_idx}.output.dense.{w}"  # noqa: E501
            ]
            compressed_sd[
                f"{prefix_student}.encoder.layer.{std_idx}.output_layer_norm.{w}"  # noqa: E501
            ] = state_dict[
                f"{prefix_teacher}.encoder.layer.{teacher_idx}.output.LayerNorm.{w}"  # noqa: E501
            ]

    # extract vocab
    compressed_sd["cls.predictions.decoder.weight"] = state_dict[
        "cls.predictions.decoder.weight"
    ]
    compressed_sd["cls.predictions.bias"] = state_dict["cls.predictions.bias"]

    for w in ["weight", "bias"]:
        compressed_sd[f"vocab_transform.{w}"] = state_dict[
            f"cls.predictions.transform.dense.{w}"
        ]
        compressed_sd[f"vocab_layer_norm.{w}"] = state_dict[
            f"cls.predictions.transform.LayerNorm.{w}"
        ]

    return compressed_sd
