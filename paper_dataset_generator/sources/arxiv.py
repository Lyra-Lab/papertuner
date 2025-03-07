import arxiv
from typing import List, Dict, Any, Generator, Optional
from .base import BaseSource


class ArxivSource(BaseSource):
    """Source implementation for ArXiv papers."""
    
    def search(self, query: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """
        Search for papers on ArXiv matching the query.
        
        Args:
            query: Search query
            max_results: Maximum number of results to return
            
        Returns:
            List of paper metadata dictionaries
        """
        client = arxiv.Client()
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.Relevance
        )
        
        results = []
        for result in client.results(search):
            results.append({
                "id": result.entry_id.split("/")[-1],
                "title": result.title,
                "authors": [author.name for author in result.authors],
                "summary": result.summary,
                "published": result.published,
                "pdf_url": result.pdf_url,
                "categories": result.categories,
            })
        
        return results
    
    def get_pdf_url(self, paper_id: str) -> str:
        """
        Get the URL to the PDF for an ArXiv paper.
        
        Args:
            paper_id: ArXiv ID of the paper
            
        Returns:
            URL to the PDF
        """
        # Clean the ID if it has the arxiv prefix
        if "/" in paper_id:
            paper_id = paper_id.split("/")[-1]
        
        # Handle versions (e.g., 2201.04234v1)
        base_id = paper_id.split("v")[0] if "v" in paper_id else paper_id
        
        return f"https://arxiv.org/pdf/{paper_id}" 