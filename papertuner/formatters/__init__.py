from .base import BaseFormatter
from .huggingface import HuggingFaceFormatter

def create_formatter(formatter_type: str, **kwargs):
    """Create a formatter instance."""
    if formatter_type == "huggingface":
        return HuggingFaceFormatter(**kwargs)
    else:
        raise ValueError(f"Unsupported formatter type: {formatter_type}")
