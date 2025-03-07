from typing import Dict, Any, Optional
from .base import BaseSource
from .arxiv import ArxivSource


def create_source(source_type: str, **kwargs) -> BaseSource:
    """
    Factory function to create source instances.
    
    Args:
        source_type: Type of source to create ("arxiv", etc.)
        **kwargs: Additional arguments to pass to the source constructor
        
    Returns:
        An instance of the requested source class
    """
    source_map = {
        "arxiv": ArxivSource,
    }
    
    if source_type not in source_map:
        raise ValueError(f"Unknown source type: {source_type}. Available types: {list(source_map.keys())}")
    
    return source_map[source_type](**kwargs) 