import arxiv
import logging
from typing import List, Dict, Any, Optional
from tqdm import tqdm
from .base import BaseSource

logger = logging.getLogger(__name__)

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
        logger.debug(f"ArxivSource initialized with max_results={max_results}")

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
        logger.debug(f"Searching arXiv for: {query} (max_results={max_results})")

        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=self.default_sort_by
        )

        results = []
        try:
            # Create iterator but don't consume it yet
            iterator = self.client.results(search)

            # Wrap the iterator with tqdm to show progress as results come in
            # We don't know the total count until we fetch all results
            for i, result in enumerate(tqdm(iterator, desc="Fetching papers", unit="paper", leave=False)):
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

        except Exception as e:
            logger.error(f"Error searching arXiv: {e}")

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
