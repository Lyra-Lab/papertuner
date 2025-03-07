import arxiv
from typing import List, Dict, Any, Generator, Optional, Union
from .base import BaseSource


class ArxivSource(BaseSource):
    """Source implementation for ArXiv papers with enhanced integration."""

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
        self._cache = {}  # Cache for paper results to avoid duplicate API calls

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

        # Check if we have cached results for this query
        cache_key = f"{query}_{max_results}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=self.default_sort_by
        )

        results = []
        for result in self.client.results(search):
            # Store the original arxiv.Result object for direct access later
            paper_data = {
                "id": result.entry_id.split("/")[-1],
                "title": result.title,
                "authors": [author.name for author in result.authors],
                "summary": result.summary,
                "published": result.published,
                "pdf_url": result.pdf_url,
                "categories": result.categories,
                "_arxiv_result": result  # Store original result for direct access
            }
            results.append(paper_data)

        # Cache the results
        self._cache[cache_key] = results
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

    def get_paper_by_id(self, paper_id: str) -> Dict[str, Any]:
        """
        Get a paper by its ArXiv ID.

        Args:
            paper_id: ArXiv ID of the paper

        Returns:
            Paper metadata dictionary

        Raises:
            ValueError: If the paper is not found
        """
        # Clean the ID if it has the arxiv prefix
        if "/" in paper_id:
            paper_id = paper_id.split("/")[-1]

        # Check if we already have this paper in our cache
        for cache_key, papers in self._cache.items():
            for paper in papers:
                if paper["id"] == paper_id:
                    return paper

        # If not in cache, fetch directly
        search = arxiv.Search(
            id_list=[paper_id],
            max_results=1
        )

        try:
            result = next(self.client.results(search))
            paper_data = {
                "id": result.entry_id.split("/")[-1],
                "title": result.title,
                "authors": [author.name for author in result.authors],
                "summary": result.summary,
                "published": result.published,
                "pdf_url": result.pdf_url,
                "categories": result.categories,
                "_arxiv_result": result  # Store original result for direct access
            }
            return paper_data
        except StopIteration:
            raise ValueError(f"Paper with ID {paper_id} not found on ArXiv")

    def get_pdf_data(self, paper_id: str) -> bytes:
        """
        Download the PDF data for a paper directly.

        Args:
            paper_id: ArXiv ID of the paper

        Returns:
            Raw PDF data as bytes

        Raises:
            ValueError: If the paper is not found
        """
        # First try to get the paper from cache or API
        try:
            paper = self.get_paper_by_id(paper_id)
            if "_arxiv_result" in paper:
                # Use the original arxiv.Result object to download
                result = paper["_arxiv_result"]
                return result.download_pdf()
        except (ValueError, KeyError):
            pass

        # Fallback to direct download
        import requests
        pdf_url = self.get_pdf_url(paper_id)
        response = requests.get(pdf_url)
        response.raise_for_status()
        return response.content
