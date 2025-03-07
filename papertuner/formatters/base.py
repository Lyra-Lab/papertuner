from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, IO


class BaseFormatter(ABC):
    """Base class for dataset formatters."""
    
    @abstractmethod
    def format_entry(self, paper_metadata: Dict[str, Any], text: str) -> Dict[str, Any]:
        """
        Format a single paper entry for the dataset.
        
        Args:
            paper_metadata: Metadata about the paper
            text: Extracted text from the paper
            
        Returns:
            Formatted entry
        """
        pass
    
    @abstractmethod
    def save(self, entries: List[Dict[str, Any]], output_path: str) -> None:
        """
        Save the formatted entries to a file.
        
        Args:
            entries: List of formatted entries
            output_path: Path to save the dataset to
        """
        pass 