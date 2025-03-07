from abc import ABC, abstractmethod
from typing import List, Dict, Any

class BaseFormatter(ABC):
    """Base class for dataset formatters."""
    
    @abstractmethod
    def format_entry(self, paper_metadata: Dict[str, Any], text: str) -> Dict[str, Any]:
        """Format a single paper entry for the dataset."""
        pass
    
    @abstractmethod
    def save(self, entries: List[Dict[str, Any]], output_path: str) -> None:
        """Save the formatted entries to a file."""
        pass