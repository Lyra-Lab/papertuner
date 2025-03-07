from typing import Dict, Any, Optional
from .base import BaseFormatter
from .jsonl import JsonlFormatter


def create_formatter(formatter_type: str, **kwargs) -> BaseFormatter:
    """
    Factory function to create formatter instances.
    
    Args:
        formatter_type: Type of formatter to create ("jsonl", etc.)
        **kwargs: Additional arguments to pass to the formatter constructor
        
    Returns:
        An instance of the requested formatter class
    """
    formatter_map = {
        "jsonl": JsonlFormatter,
    }
    
    if formatter_type not in formatter_map:
        raise ValueError(f"Unknown formatter type: {formatter_type}. Available types: {list(formatter_map.keys())}")
    
    return formatter_map[formatter_type](**kwargs) 