import os
import tempfile
import base64
import time
import logging
import requests
from typing import Optional, Union, Dict, Any, List
from pathlib import Path
import io
import json

try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False

try:
    from pdf2image import convert_from_path, convert_from_bytes
    PDF2IMAGE_AVAILABLE = True
except ImportError:
    PDF2IMAGE_AVAILABLE = False

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

from .base import BaseOCR

# Set up logging
logging.basicConfig(level=logging.INFO)
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

        logger.info(f"Configuring Ollama to use host: {self.ollama_host}")

        # Configure the Ollama client
        ollama.host = self.ollama_host

    def process_url(self, url: str) -> str:
        """Process a document from a URL and return the extracted text."""
        # Download the file
        response = requests.get(url)
        response.raise_for_status()
        
        # Create a temporary file
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temp_file:
            temp_file.write(response.content)
            temp_path = temp_file.name
            
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

        # Convert PDF to images
        logger.info(f"Converting PDF to images: {file_path}")
        try:
            images = convert_from_path(
                file_path,
                dpi=300,
                fmt="jpeg",
                grayscale=False,
                transparent=False
            )
            logger.info(f"Converted PDF to {len(images)} images")
        except Exception as e:
            logger.error(f"Error converting PDF to images: {e}")
            raise

        if not images:
            raise ValueError(f"Failed to extract any images from the PDF: {file_path}")

        # Process the images in chunks to avoid overwhelming the model
        all_text = []
        chunk_size = min(5, len(images))  # Process max 5 pages at once

        for i in range(0, len(images), chunk_size):
            chunk_images = images[i:i+chunk_size]
            logger.info(f"Processing pages {i+1}-{i+len(chunk_images)} of {len(images)}")

            chunk_text = self._process_image_chunk(chunk_images, i)
            all_text.append(chunk_text)

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
        prompt = f"Extract all text from these {len(images)} PDF pages ({start_page+1}-{start_page+len(images)}). Return only the extracted text content."

        # Encode images to base64
        image_prompts = []
        for idx, img in enumerate(images):
            with io.BytesIO() as output:
                img.save(output, format="JPEG")
                img_data = output.getvalue()
                img_b64 = base64.b64encode(img_data).decode('utf-8')
                image_prompts.append(f"data:image/jpeg;base64,{img_b64}")

        # Create the message list
        messages = [
            {"role": "system", "content": "You are an expert OCR system designed to accurately extract text from PDF images."},
            {"role": "user", "content": [
                {"type": "text", "text": prompt}
            ]}
        ]

        # Add the images to the user message
        for img_data in image_prompts:
            messages[1]["content"].append({"type": "image", "image": img_data})

        # Try to call the model with retries
        for attempt in range(self.max_retries):
            try:
                logger.info(f"Calling Ollama model (attempt {attempt+1}/{self.max_retries})")

                response = ollama.chat(
                    model=self.model_name,
                    messages=messages,
                    options={
                        "temperature": self.temperature,
                        "num_predict": self.max_tokens,
                    }
                )

                if response and "content" in response:
                    return response["content"]
                else:
                    logger.warning(f"Empty or invalid response from Ollama: {response}")

            except Exception as e:
                logger.warning(f"Error on attempt {attempt+1}: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)

        # If all attempts fail, raise an exception
        raise RuntimeError(f"Failed to process images after {self.max_retries} attempts")