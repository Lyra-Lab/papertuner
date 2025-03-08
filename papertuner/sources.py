# papertuner/sources.py
"""
Source clients for fetching papers from different repositories.
"""

import arxiv
import os
import logging
import tempfile
import requests
import shutil
import time
from urllib.parse import urlparse
from pathlib import Path

logger = logging.getLogger(__name__)

class SourceClient:
    """Base class for source clients."""

    def fetch_papers(self):
        """Fetch papers from the source."""
        raise NotImplementedError("Subclasses must implement fetch_papers")


class ArxivClient(SourceClient):
    """Client for fetching papers from arXiv."""

    def __init__(self, max_papers=100, queries=None, categories=None):
        """
        Initialize ArxivClient.

        Args:
            max_papers (int): Maximum number of papers to fetch
            queries (list): List of query strings
            categories (list): List of arXiv categories
        """
        self.max_papers = max_papers
        self.queries = queries or []
        self.categories = categories or []

        if not self.queries and not self.categories:
            raise ValueError("At least one query or category must be provided")

    def fetch_papers(self):
        """
        Fetch papers from arXiv based on queries and categories.

        Returns:
            list: List of dictionaries containing paper metadata and local PDF path
        """
        papers = []
        temp_dir = tempfile.mkdtemp(prefix="papertuner_")
        
        try:
            for query in self.queries:
                logger.info(f"Fetching papers for query: {query}")
                search_query = query

                # Add categories to the search query if specified
                if self.categories:
                    category_str = " OR ".join(f"cat:{cat}" for cat in self.categories)
                    search_query = f"{search_query} AND ({category_str})"

                search = arxiv.Search(
                    query=search_query,
                    max_results=self.max_papers,
                    sort_by=arxiv.SortCriterion.Relevance
                )

                for i, result in enumerate(search.results()):
                    # Add a small delay every few requests to avoid rate limiting
                    if i > 0 and i % 5 == 0:
                        time.sleep(1)
                        
                    # Download the PDF
                    pdf_path = self._download_pdf(result.pdf_url, temp_dir)

                    if pdf_path:
                        papers.append({
                            "id": result.entry_id,
                            "title": result.title,
                            "authors": [author.name for author in result.authors],
                            "abstract": result.summary,
                            "pdf_url": result.pdf_url,
                            "pdf_path": pdf_path,
                            "categories": result.categories,
                            "published": result.published.strftime("%Y-%m-%d"),
                            "updated": result.updated.strftime("%Y-%m-%d") if result.updated else None
                        })

            logger.info(f"Fetched {len(papers)} papers")
            return papers
        finally:
            # Clean up the temporary directory
            shutil.rmtree(temp_dir, ignore_errors=True)

    def _download_pdf(self, url, temp_dir):
        """
        Download PDF from a URL to a temporary directory.

        Args:
            url (str): URL of the PDF
            temp_dir (str): Temporary directory path

        Returns:
            str: Path to the downloaded PDF, or None if download failed
        """
        try:
            # Extract file name from URL
            parsed_url = urlparse(url)
            file_name = os.path.basename(parsed_url.path)

            # Ensure file has .pdf extension
            if not file_name.lower().endswith('.pdf'):
                file_name = f"{file_name}.pdf"

            file_path = os.path.join(temp_dir, file_name)

            # Download the file
            response = requests.get(url, stream=True)
            response.raise_for_status()

            with open(file_path, 'wb') as file:
                for chunk in response.iter_content(chunk_size=8192):
                    file.write(chunk)

            return file_path

        except Exception as e:
            logger.error(f"Error downloading PDF from {url}: {e}")
            return None
