import arxiv
from typing import List, Dict, Any, Optional
from .base import BaseSource

class ArxivSource(BaseSource):
    """Source implementation for ArXiv papers."""

    def __init__(self, max_results: int = 100, sort_by: arxiv.SortCriterion = arxiv.SortCriterion.Relevance):
        """
        Initialize the ArXiv source.

        Args:
            max_results: Default maximum number of results to return
            sort_by: Default sorting criterion for ArXiv queries
        """
        self.client = arxiv.Client()
        self.default_max_results = max_results
        self.default_sort_by = sort_by

    def search(self, query: str, max_results: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Search for papers on ArXiv matching the query.

        Args:
            query: Search query
            max_results: Maximum number of results to return (overrides default)

        Returns:
            List of paper metadata dictionaries
        """
        max_results = max_results or self.default_max_results

        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=self.default_sort_by
        )

        results = []
        for result in self.client.results(search):
            paper_data = {
                "id": result.entry_id.split("/")[-1],
                "title": result.title,
                "authors": [author.name for author in result.authors],
                "summary": result.summary,
                "published": result.published,
                "pdf_url": result.pdf_url,
                "categories": result.categories,
            }
            results.append(paper_data)

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

        return f"https://arxiv.org/pdf/{paper_id}"