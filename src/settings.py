"""Defines which libraries are available.
"""

IS_TRANSFORMERS_AVAILABLE = True

try:
    import transformers  # noqa: F401
except ImportError:
    IS_TRANSFORMERS_AVAILABLE = False

__all__ = ["IS_TRANSFORMERS_AVAILABLE"]
