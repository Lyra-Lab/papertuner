from .base import BaseSource
from .arxiv import ArxivSource

def create_source(source_type: str, **kwargs):
    """Create a source instance."""
    if source_type == "arxiv":
        return ArxivSource(**kwargs)
    else:
        raise ValueError(f"Unsupported source type: {source_type}")