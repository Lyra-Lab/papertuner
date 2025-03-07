import os
from typing import Dict, Any, Optional
import tempfile
import requests

from mistralai import Mistral
from .base import BaseOCR


class MistralOCR(BaseOCR):
    """OCR implementation using Mistral's OCR API."""
    
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
    
    def process_url(self, url: str) -> str:
        """Process a document from a URL using Mistral OCR."""
        response = self.client.ocr.process(
            model=self.model,
            document={
                "type": "document_url",
                "document_url": url
            }
        )
        return response.text
    
    def process_file(self, file_path: str) -> str:
        """Process a document from a local file using Mistral OCR."""
        with open(file_path, "rb") as f:
            file_content = f.read()
        
        response = self.client.ocr.process(
            model=self.model,
            document={
                "type": "document_content",
                "document_content": file_content
            }
        )
        return response.text 