from abc import abstractmethod
from typing import Dict, Any, Optional, Union
import tempfile
import os
from .base import BaseOCR


class ArxivIntegratedOCR(BaseOCR):
    """Base class for OCR implementations with direct ArXiv integration."""

    def process_arxiv_id(self, arxiv_id: str, source=None) -> str:
        """
        Process a paper directly from ArXiv using its ID.

        Args:
            arxiv_id: ArXiv ID of the paper
            source: Optional ArxivSource instance to use for fetching the paper

        Returns:
            Extracted text from the paper
        """
        if source is None:
            # Import here to avoid circular imports
            from ..sources.factory import create_source
            source = create_source("arxiv")

        # Get the PDF data directly
        pdf_data = source.get_pdf_data(arxiv_id)

        # Process the PDF data
        return self.process_pdf_data(pdf_data)

    def process_url(self, url: str) -> str:
        """
        Process a document from a URL and return the extracted text.

        Args:
            url: URL to the PDF

        Returns:
            Extracted text from the document
        """
        # Check if this is an ArXiv URL to optimize
        if "arxiv.org" in url:
            # Try to extract the ArXiv ID from the URL
            import re
            arxiv_id_match = re.search(r'(\d+\.\d+(?:v\d+)?)', url)
            if arxiv_id_match:
                arxiv_id = arxiv_id_match.group(1)
                return self.process_arxiv_id(arxiv_id)

        # Otherwise, download and process normally
        import requests
        response = requests.get(url)
        response.raise_for_status()

        return self.process_pdf_data(response.content)

    def process_file(self, file_path: str) -> str:
        """
        Process a document from a local file and return the extracted text.

        Args:
            file_path: Path to the local file

        Returns:
            Extracted text from the document
        """
        with open(file_path, 'rb') as f:
            pdf_data = f.read()

        return self.process_pdf_data(pdf_data)

    @abstractmethod
    def process_pdf_data(self, pdf_data: bytes) -> str:
        """
        Process raw PDF data and extract text.

        Args:
            pdf_data: Raw PDF data as bytes

        Returns:
            Extracted text from the PDF
        """
        pass
