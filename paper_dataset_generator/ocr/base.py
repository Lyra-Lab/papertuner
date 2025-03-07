from abc import ABC, abstractmethod
from typing import Dict, Any, Optional


class BaseOCR(ABC):
    """Base class for OCR implementations."""
    
    @abstractmethod
    def process_url(self, url: str) -> str:
        """Process a document from a URL and return the extracted text."""
        pass
    
    @abstractmethod
    def process_file(self, file_path: str) -> str:
        """Process a document from a local file and return the extracted text."""
        pass 