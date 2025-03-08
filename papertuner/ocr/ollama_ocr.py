import os
import tempfile
import base64
import time
import logging
import requests
from typing import Optional, Union, Dict, Any, List
from pathlib import Path
import io
from tqdm import tqdm

try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False

try:
    from pdf2image import convert_from_path
    PDF2IMAGE_AVAILABLE = True
except ImportError:
    PDF2IMAGE_AVAILABLE = False

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

from .base import BaseOCR

# Get logger
logger = logging.getLogger(__name__)

class OllamaOCR(BaseOCR):
    """OCR implementation using Ollama with vision-language models."""

    def __init__(
        self,
        model_name: str = "llava:latest",
        ollama_host: Optional[str] = None,
        max_retries: int = 3,
        retry_delay: float = 2.0,
        temperature: float = 0.2,
        max_tokens: int = 4096,
        **kwargs
    ):
        """
        Initialize the Ollama OCR processor.

        Args:
            model_name: Name of the Ollama model to use (default: "llava:latest")
            ollama_host: URL of the Ollama API server (default: from OLLAMA_HOST env var or http://127.0.0.1:11434)
            max_retries: Maximum number of retries for API calls
            retry_delay: Delay between retries in seconds
            temperature: Temperature for generation
            max_tokens: Maximum tokens to generate
            **kwargs: Additional parameters to pass to Ollama
        """
        # Check if required dependencies are installed
        if not OLLAMA_AVAILABLE:
            raise ImportError("The 'ollama' package is required. Install with 'pip install ollama'")

        if not PDF2IMAGE_AVAILABLE:
            raise ImportError("The 'pdf2image' package is required. Install with 'pip install pdf2image'")

        if not PIL_AVAILABLE:
            raise ImportError("Pillow is required. Install with 'pip install Pillow'")

        # Set up Ollama client with host from parameters, environment, or default
        self.model_name = model_name
        self.ollama_host = ollama_host or os.environ.get("OLLAMA_HOST") or "http://127.0.0.1:11434"
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.additional_params = kwargs

        # Configure the Ollama client
        ollama.host = self.ollama_host

        logger.debug(f"OllamaOCR initialized with model={model_name}, host={self.ollama_host}")

    def process_url(self, url: str) -> str:
        """Process a document from a URL and return the extracted text."""
        # Download the file with progress bar
        logger.debug(f"Downloading PDF from URL: {url}")

        # Stream the download with progress tracking
        response = requests.get(url, stream=True)
        response.raise_for_status()

        # Get total file size for progress bar
        total_size = int(response.headers.get('content-length', 0))

        # Use tqdm to show download progress
        progress = tqdm(
            total=total_size,
            unit='B',
            unit_scale=True,
            desc="Downloading PDF",
            leave=False
        )

        # Create a temporary file and download with progress
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temp_file:
            for chunk in response.iter_content(chunk_size=8192):
                temp_file.write(chunk)
                progress.update(len(chunk))
            temp_path = temp_file.name

        progress.close()

        try:
            # Process the local file
            return self.process_file(temp_path)
        finally:
            # Clean up
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def process_file(self, file_path: Union[str, Path]) -> str:
        """
        Process a PDF file using Ollama.

        Args:
            file_path: Path to the PDF file

        Returns:
            Extracted text from the PDF
        """
        file_path = Path(file_path)

        # Check if file exists
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        # Convert PDF to images with progress bar
        logger.debug(f"Converting PDF to images: {file_path}")
        try:
            # First get page count for progress bar (if possible)
            import PyPDF2
            try:
                with open(file_path, 'rb') as f:
                    pdf = PyPDF2.PdfReader(f)
                    total_pages = len(pdf.pages)
                    pdf_desc = f"Converting {total_pages} PDF pages"
            except:
                pdf_desc = "Converting PDF"

            # Convert with progress description
            with tqdm(desc=pdf_desc, unit="page", leave=False) as pbar:
                images = convert_from_path(
                    file_path,
                    dpi=300,
                    fmt="jpeg",
                    grayscale=False,
                    transparent=False,
                    thread_count=2,  # Use multiple threads for conversion
                )
                pbar.update(len(images))

            logger.debug(f"Converted PDF to {len(images)} images")
        except Exception as e:
            logger.error(f"Error converting PDF to images: {e}")
            raise

        if not images:
            raise ValueError(f"Failed to extract any images from the PDF")

        # Process the images in chunks to avoid overwhelming the model
        all_text = []
        chunk_size = min(5, len(images))  # Process max 5 pages at once
        total_pages = len(images)

        # Create progress bar for the OCR process
        with tqdm(total=total_pages, desc="OCR processing", unit="page", leave=False) as ocr_pbar:
            for i in range(0, total_pages, chunk_size):
                chunk_images = images[i:i+chunk_size]
                end_page = min(i+len(chunk_images), total_pages)
                ocr_pbar.set_description(f"OCR pages {i+1}-{end_page}")

                chunk_text = self._process_image_chunk(chunk_images, i)
                all_text.append(chunk_text)
                ocr_pbar.update(len(chunk_images))

        # Combine all extracted text
        return "\n\n".join(all_text)

    def _process_image_chunk(self, images: List, start_page: int) -> str:
        """
        Process a chunk of images using the Ollama model.

        Args:
            images: List of PIL images
            start_page: Starting page number for this chunk

        Returns:
            Extracted text from the images
        """
        # Create a combined prompt with all images
        prompt = f"Extract all text from these {len(images)} PDF pages. Return only the extracted text content."

        # Try to call the model with retries
        for attempt in range(self.max_retries):
            try:
                # Create a direct call to the Ollama API for each image
                responses = []

                # Process each image individually with progress
                progress_desc = f"Processing {len(images)} images"
                for idx, img in enumerate(tqdm(images, desc=progress_desc, leave=False)):
                    # Convert image to base64
                    with io.BytesIO() as output:
                        img.save(output, format="JPEG")
                        img_data = output.getvalue()
                        img_b64 = base64.b64encode(img_data).decode('utf-8')

                    # Call the API directly for each image
                    response = ollama.generate(
                        model=self.model_name,
                        prompt=f"Page {start_page + idx + 1}: {prompt}",
                        images=[img_b64],
                        options={
                            "temperature": self.temperature,
                            "num_predict": self.max_tokens,
                        }
                    )

                    if response and "response" in response:
                        responses.append(response["response"])
                    else:
                        logger.warning(f"Empty response from Ollama for page {start_page + idx + 1}")

                # Combine responses
                if responses:
                    combined_text = "\n\n".join(responses)
                    return combined_text
                else:
                    logger.warning("No text extracted from any page in this chunk")

            except Exception as e:
                if attempt < self.max_retries - 1:
                    backoff_time = self.retry_delay * (attempt + 1)  # Increase delay with each retry
                    logger.debug(f"Error on attempt {attempt+1}: {str(e)}. Retrying in {backoff_time}s")
                    # Show retry progress
                    for _ in tqdm(range(int(backoff_time)), desc=f"Retry {attempt+1}/{self.max_retries-1}", leave=False):
                        time.sleep(1)
                else:
                    logger.error(f"Failed after {self.max_retries} attempts: {str(e)}")

        # If all attempts fail, raise an exception
        raise RuntimeError(f"Failed to process images after {self.max_retries} attempts")
