from typing import Dict, Any, Optional, Type
from .base import BaseFormatter
from .jsonl import JsonlFormatter

# Import new formatters if they exist
try:
    from .huggingface import HuggingFaceFormatter
    huggingface_available = True
except ImportError:
    huggingface_available = False

try:
    from .csv_formatter import CSVFormatter
    csv_available = True
except ImportError:
    csv_available = False

try:
    from .yaml_formatter import YAMLFormatter
    yaml_available = True
except ImportError:
    yaml_available = False

try:
    from .chat_formatter import ChatFormatter
    chat_available = True
except ImportError:
    chat_available = False

try:
    from .template_formatter import TemplateFormatter
    template_available = True
except ImportError:
    template_available = False


def create_formatter(formatter_type: str, **kwargs) -> BaseFormatter:
    """
    Factory function to create formatter instances.

    Args:
        formatter_type: Type of formatter to create ("jsonl", etc.)
        **kwargs: Additional arguments to pass to the formatter constructor

    Returns:
        An instance of the requested formatter class

    Raises:
        ValueError: If the formatter type is not supported or the required module is not installed
    """
    # Build the formatter map dynamically based on available formatters
    formatter_map = {
        "jsonl": JsonlFormatter,
    }

    # Add optional formatters if available
    if huggingface_available:
        formatter_map["huggingface"] = HuggingFaceFormatter

    if csv_available:
        formatter_map["csv"] = CSVFormatter

    if yaml_available:
        formatter_map["yaml"] = YAMLFormatter

    if chat_available:
        formatter_map["chat"] = ChatFormatter

    if template_available:
        formatter_map["template"] = TemplateFormatter

    # Check if the requested formatter is available
    if formatter_type not in formatter_map:
        available_formatters = list(formatter_map.keys())
        raise ValueError(f"Unknown formatter type: {formatter_type}. Available types: {available_formatters}")

    # Create the formatter instance
    formatter_class = formatter_map[formatter_type]

    try:
        return formatter_class(**kwargs)
    except TypeError as e:
        # If the formatter doesn't accept the provided arguments, provide a helpful error
        raise TypeError(f"Error creating formatter of type '{formatter_type}': {str(e)}. "
                       f"Check the required parameters for {formatter_class.__name__}.") from e
