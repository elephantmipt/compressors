from compressors.settings import IS_TRANSFORMERS_AVAILABLE


if IS_TRANSFORMERS_AVAILABLE:
    from .bert import init_bert_model_with_teacher

    __all__ = ["init_bert_model_with_teacher"]
