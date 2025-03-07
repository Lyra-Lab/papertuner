import os
from typing import Dict, Any, Optional
import tempfile

from mistralai import Mistral
from .arxiv_integrated import ArxivIntegratedOCR


class MistralOCR(ArxivIntegratedOCR):
    """OCR implementation using Mistral's OCR API with ArXiv integration."""

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the Mistral OCR client.

        Args:
            api_key: Mistral API key. If None, it will be read from the MISTRAL_API_KEY environment variable.
        """
        self.api_key = api_key or os.environ.get("MISTRAL_API_KEY")
        if not self.api_key:
            raise ValueError("Mistral API key is required. Set it via the api_key parameter or MISTRAL_API_KEY environment variable.")

        self.client = Mistral(api_key=self.api_key)
        self.model = "mistral-ocr-latest"

    def process_pdf_data(self, pdf_data: bytes) -> str:
        """Process raw PDF data using Mistral OCR."""
        response = self.client.ocr.process(
            model=self.model,
            document={
                "type": "document_content",
                "document_content": pdf_data
            }
        )
        return response.text
