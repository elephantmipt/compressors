"""Defines which libraries are available.
"""

IS_TORCHVISION_AVAILABLE = True

try:
    import torchvision
except ImportError:
    IS_TORCHVISION_AVAILABLE = False

IS_TRANSFORMERS_AVAILABLE = True

try:
    import transformers  # noqa: F401
except ImportError:
    IS_TRANSFORMERS_AVAILABLE = False

__all__ = ["IS_TRANSFORMERS_AVAILABLE", "IS_TORCHVISION_AVAILABLE"]
