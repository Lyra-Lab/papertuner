"""
OCR module for extracting text from documents using the Gemini API.
"""

import os
import logging
import time
import google.generativeai as genai
import pymupdf  # PyMuPDF

logger = logging.getLogger(__name__)

class OCRBase:
    """Base class for OCR implementations."""

    def extract_text(self, pdf_path):
        """Extract text from a PDF file."""
        raise NotImplementedError("Subclasses must implement extract_text")

class GeminiOCR(OCRBase):
    """OCR implementation using Gemini API."""

    def __init__(self, api_key, hf_token=None):
        """
        Initialize GeminiOCR with API key.

        Args:
            api_key (str): Gemini API key
            hf_token (str, optional): HuggingFace token for additional capabilities
        """
        if not api_key or api_key.strip() == "":
            raise ValueError("Valid Gemini API key is required")

        self.api_key = api_key
        self.hf_token = hf_token
        genai.configure(api_key=api_key)

        # Test the API key with a simple request
        try:
            genai.GenerativeModel('gemini-1.5-flash')
            logger.info("Successfully initialized Gemini API")
        except Exception as e:
            raise ValueError(f"Failed to initialize Gemini API with provided key: {e}")

        self.model = genai.GenerativeModel('gemini-1.5-flash')

    def extract_text(self, pdf_path):
        """
        Extract text from a PDF file using Gemini's API.

        Args:
            pdf_path (str): Path to the PDF file

        Returns:
            str: Extracted text
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        logger.info(f"Starting OCR extraction for: {os.path.basename(pdf_path)}")

        # Check file size (Gemini has a 20MB limit)
        file_size = os.path.getsize(pdf_path) / (1024 * 1024)  # Size in MB
        logger.info(f"File size: {file_size:.2f}MB")

        if file_size > 20:
            logger.warning(f"File size ({file_size:.2f}MB) exceeds Gemini's limit. Results may be truncated.")

        max_retries = 3
        for attempt in range(max_retries):
            try:
                # Upload the PDF file
                logger.info(f"Uploading file to Gemini API (attempt {attempt+1}/{max_retries})")
                sample_file = genai.upload_file(path=pdf_path, display_name=os.path.basename(pdf_path))
                logger.info(f"Uploaded file '{sample_file.display_name}' as: {sample_file.uri}")

                # Generate content using the uploaded document with an improved prompt
                logger.info("Generating content from PDF")
                response = self.model.generate_content([
                    sample_file,
                    "Extract the full text content from this scientific paper, preserving sections, "
                    "equations, and important formatting. Ignore headers, footers, and page numbers."
                ])

                # Extract the text from the response
                extracted_text = response.text
                text_length = len(extracted_text)
                logger.info(f"Successfully extracted {text_length} characters of text")

                return extracted_text

            except Exception as e:
                if attempt < max_retries - 1:
                    logger.warning(f"Retry {attempt+1}/{max_retries} after error: {e}")
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    logger.error(f"Error extracting text with Gemini API after {max_retries} attempts: {e}")
                    raise Exception(f"OCR extraction failed: {e}")

class PyMuPDFOCR(OCRBase):
    """OCR implementation using PyMuPDF."""

    def __init__(self):
        """
        Initialize PyMuPDFOCR.
        """
        logger.info("Successfully initialized PyMuPDF OCR")

    def extract_text(self, pdf_path):
        """
        Extract text from a PDF file using PyMuPDF.

        Args:
            pdf_path (str): Path to the PDF file

        Returns:
            str: Extracted text
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        logger.info(f"Starting OCR extraction for: {os.path.basename(pdf_path)}")

        # Open the PDF file with PyMuPDF
        doc = pymupdf.open(pdf_path)
        extracted_text = ""

        # Extract text from each page
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            extracted_text += page.get_text()

        text_length = len(extracted_text)
        logger.info(f"Successfully extracted {text_length} characters of text")

        return extracted_text
