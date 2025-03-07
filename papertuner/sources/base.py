from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

class BaseSource(ABC):
    """Base class for data sources."""
    
    @abstractmethod
    def search(self, query: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """
        Search for papers matching the query.
        
        Args:
            query: Search query
            max_results: Maximum number of results to return
            
        Returns:
            List of paper metadata dictionaries
        """
        pass
    
    @abstractmethod
    def get_pdf_url(self, paper_id: str) -> str:
        """
        Get the URL to the PDF for a paper.
        
        Args:
            paper_id: ID of the paper
            
        Returns:
            URL to the PDF
        """
        pass