"""OCR implementation using Google's Gemini API with robust PDF handling."""

import os
import tempfile
import base64
import time
from typing import Optional, Union, Dict, Any
import logging
import requests
from pathlib import Path
import google.generativeai as genai
from google.api_core.exceptions import GoogleAPIError
from .arxiv_integrated import ArxivIntegratedOCR

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GeminiOCR(ArxivIntegratedOCR):
    """OCR implementation using Google's Gemini Vision capabilities with robust PDF handling."""
    
    def __init__(
        self, 
        api_key: Optional[str] = None, 
        model: str = "gemini-1.5-pro-latest", # Using a more capable model for PDF processing
        max_retries: int = 3,
        retry_delay: float = 2.0,
        fallback_to_text_extraction: bool = True,
        chunk_size: int = 10000,  # Characters per chunk for text processing
    ):
        """
        Initialize the Gemini OCR for text extraction from PDFs.
        
        Args:
            api_key: Google API key for Gemini. If None, will try to get from GOOGLE_API_KEY environment variable.
            model: Gemini model to use
            max_retries: Maximum number of retries for API calls
            retry_delay: Delay between retries in seconds
            fallback_to_text_extraction: Whether to try extracting text directly if PDF processing fails
            chunk_size: Maximum size of text chunks for processing
        
        Raises:
            ValueError: If no API key is provided and none found in environment variables
        """
        self.model_name = model
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.fallback_to_text_extraction = fallback_to_text_extraction
        self.chunk_size = chunk_size
        
        # Get API key from parameters or environment
        api_key = api_key or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("Google API key is required. Set it via the api_key parameter or GOOGLE_API_KEY environment variable.")
        
        genai.configure(api_key=api_key)
        
    def process_arxiv_id(self, arxiv_id: str, source=None) -> str:
        """
        Process a paper directly from ArXiv using its ID.
        
        Args:
            arxiv_id: ArXiv ID of the paper
            source: Optional ArxivSource instance to use for fetching the paper
            
        Returns:
            Extracted text from the paper
        """
        # Save the PDF locally first to avoid base64 encoding issues
        pdf_path = self._download_arxiv_pdf(arxiv_id, source)
        
        try:
            # Process the local file
            return self.process_file(pdf_path)
        finally:
            # Clean up temporary file
            if os.path.exists(pdf_path):
                os.remove(pdf_path)
    
    def _download_arxiv_pdf(self, arxiv_id: str, source=None) -> str:
        """Download an ArXiv PDF to a local file and return the path."""
        if source is None:
            # Import here to avoid circular imports
            from ..sources.factory import create_source
            source = create_source("arxiv")
        
        # Get the PDF URL
        pdf_url = source.get_pdf_url(arxiv_id)
        
        # Download the PDF
        response = requests.get(pdf_url)
        response.raise_for_status()
        
        # Save to a temporary file
        temp_path = f"{arxiv_id}.pdf"
        with open(temp_path, "wb") as f:
            f.write(response.content)
        
        return temp_path
        
    def process_pdf_data(self, pdf_data: bytes) -> str:
        """Process raw PDF data using Gemini."""
        # Save PDF to temporary file to avoid base64 encoding issues
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temp_file:
            temp_file.write(pdf_data)
            temp_path = temp_file.name
        
        try:
            # Process using file
            return self.process_file(temp_path)
        finally:
            # Clean up temporary file
            if os.path.exists(temp_path):
                os.remove(temp_path)
    
    def process_file(self, file_path: Union[str, Path]) -> str:
        """
        Process a document from a local file with robust error handling.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            Extracted text from the document
        """
        file_path = Path(file_path)
        
        # Check if file exists
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Try the native Gemini PDF processing first
        try:
            return self._process_pdf_with_gemini(file_path)
        except Exception as e:
            logger.warning(f"Error processing PDF with Gemini: {e}")
            
            # If PDF processing fails and fallback is enabled, try text extraction
            if self.fallback_to_text_extraction:
                logger.info(f"Falling back to PyPDF2 text extraction for {file_path}")
                try:
                    return self._extract_text_from_pdf(file_path)
                except Exception as text_error:
                    logger.error(f"Text extraction fallback also failed: {text_error}")
            
            # Re-raise the original error if all methods fail
            raise
    
    def _process_pdf_with_gemini(self, file_path: Path) -> str:
        """Process a PDF file with Gemini with retries."""
        model = genai.GenerativeModel(self.model_name)
        
        # Read PDF file
        with open(file_path, "rb") as f:
            pdf_data = f.read()
        
        last_error = None
        for attempt in range(self.max_retries):
            try:
                logger.info(f"Processing PDF with Gemini (attempt {attempt+1}/{self.max_retries})")
                
                # Create a prompt that asks Gemini to extract text from the PDF
                prompt = [
                    "Extract all the text content from this research paper PDF. Maintain the structure, "
                    "formatting, equations, and references as much as possible. Return only the extracted text.",
                    {"mime_type": "application/pdf", "data": pdf_data}
                ]
                
                response = model.generate_content(prompt)
                
                # Check if we got a valid response
                if response and hasattr(response, 'text') and response.text:
                    return response.text
                else:
                    raise RuntimeError("Empty response from Gemini API")
                    
            except GoogleAPIError as e:
                last_error = e
                logger.warning(f"API error on attempt {attempt+1}: {e}")
                time.sleep(self.retry_delay)
            except Exception as e:
                last_error = e
                logger.warning(f"Error on attempt {attempt+1}: {e}")
                time.sleep(self.retry_delay)
        
        # If we get here, all attempts failed
        raise RuntimeError(f"Failed to process PDF after {self.max_retries} attempts: {last_error}")
    
    def _extract_text_from_pdf(self, file_path: Path) -> str:
        """Extract text from PDF using PyPDF2 as a fallback method."""
        try:
            import PyPDF2
        except ImportError:
            raise ImportError("PyPDF2 is required for text extraction fallback. Install with 'pip install PyPDF2'")
        
        text_parts = []
        
        try:
            with open(file_path, 'rb') as f:
                pdf_reader = PyPDF2.PdfReader(f)
                
                # Get number of pages
                num_pages = len(pdf_reader.pages)
                logger.info(f"Extracting text from {num_pages} pages")
                
                if num_pages == 0:
                    raise ValueError("PDF has no pages")
                
                # Extract text from each page
                for i in range(num_pages):
                    page = pdf_reader.pages[i]
                    page_text = page.extract_text()
                    text_parts.append(page_text)
            
            # Join all text
            full_text = "\n\n".join(text_parts)
            
            # If we got text but it's very short, try to process it with the LLM
            if len(full_text) < 500 and len(full_text) > 0:
                logger.info(f"Extracted text is very short ({len(full_text)} chars), attempting enhancement with Gemini")
                return self._enhance_extracted_text(full_text, file_path.name)
                
            return full_text
            
        except Exception as e:
            logger.error(f"Error in PyPDF2 text extraction: {e}")
            raise
    
    def _enhance_extracted_text(self, text: str, filename: str) -> str:
        """Use Gemini to enhance poorly extracted text."""
        model = genai.GenerativeModel("gemini-1.5-pro-latest")  # Use the most capable model for this
        
        prompt = f"""
        I have attempted to extract text from a research paper PDF titled "{filename}" but the extraction was poor.
        Here is the extracted text:
        
        "{text}"
        
        Please help reconstruct this into proper, well-formatted text that would represent the contents of this research paper.
        Focus on maintaining scientific accuracy and proper structure.
        """
        
        try:
            response = model.generate_content(prompt)
            if response and hasattr(response, 'text') and response.text:
                return response.text
            return text  # Return original if enhancement fails
        except Exception as e:
            logger.warning(f"Text enhancement failed: {e}")
            return text  # Return original if enhancement fails
