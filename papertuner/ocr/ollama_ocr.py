"""OCR implementation using Ollama with custom models like allenai/olmOCR-7B."""

import os
import tempfile
import base64
import time
import logging
import subprocess
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

from .arxiv_integrated import ArxivIntegratedOCR

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OllamaOCR(ArxivIntegratedOCR):
    """OCR implementation using Ollama with custom vision-language models."""

    def __init__(
        self,
        model_name: str = "olmocr",  # Default model name after creation
        create_model: bool = False,
        base_model: str = "qwen2-vl-7b-instruct", # Base model for olmOCR
        model_repo: str = "allenai/olmOCR-7B-0225-preview", # HF repo for the model
        ollama_host: Optional[str] = None,  # Will use environment variable if None
        max_retries: int = 3,
        retry_delay: float = 2.0,
        temperature: float = 0.2,
        max_tokens: int = 4096,
        chunk_overlap: int = 100,  # Pages to process in a single batch
        **kwargs
    ):
        """
        Initialize the Ollama OCR processor.

        Args:
            model_name: Name of the Ollama model to use (default: "olmocr")
            create_model: Whether to create the model if it doesn't exist
            base_model: Base model to use if creating a new model
            model_repo: HuggingFace repo for the model if different from base_model
            ollama_host: URL of the Ollama API server (default: from OLLAMA_HOST env var or http://127.0.0.1:11434)
            max_retries: Maximum number of retries for API calls
            retry_delay: Delay between retries in seconds
            temperature: Temperature for generation
            max_tokens: Maximum tokens to generate
            chunk_overlap: Number of pages to process in a single batch
            **kwargs: Additional parameters to pass to Ollama

        Raises:
            ImportError: If required dependencies are not installed
            RuntimeError: If Ollama is not available or model creation fails
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
        # Priority: 1. Parameter, 2. Environment variable, 3. Default
        self.ollama_host = ollama_host or os.environ.get("OLLAMA_HOST") or "http://127.0.0.1:11434"
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.chunk_overlap = chunk_overlap
        self.additional_params = kwargs

        logger.info(f"Configuring Ollama to use host: {self.ollama_host}")

        # Configure the Ollama client
        ollama.host = self.ollama_host

        # Create the model if requested and it doesn't exist
        if create_model:
            self._create_model(base_model, model_repo)
        else:
            # Check if the model exists
            self._check_model_exists()

    def _check_model_exists(self) -> bool:
        """Check if the model exists in Ollama."""
        try:
            models = ollama.list()
            model_exists = any(model['name'] == self.model_name for model in models.get('models', []))

            if not model_exists:
                logger.warning(f"Model '{self.model_name}' not found in Ollama. Available models: "
                              f"{[model['name'] for model in models.get('models', [])]}")
                logger.info("You can create the model with create_model=True or manually with "
                           "ollama pull qwen2-vl-7b-instruct or other base model")
                return False
            return True
        except Exception as e:
            logger.error(f"Error checking if model exists: {e}")
            return False

    def _create_model(self, base_model: str, model_repo: str) -> None:
        """
        Create a custom model in Ollama for OCR processing.

        Args:
            base_model: Base model to use
            model_repo: HuggingFace repo for the model (if using pull from HF)
        """
        try:
            # Check if model already exists
            if self._check_model_exists():
                logger.info(f"Model '{self.model_name}' already exists in Ollama")
                return

            # Create a Modelfile
            modelfile_content = f"""
FROM {base_model}
PARAMETER temperature {self.temperature}
PARAMETER max_tokens {self.max_tokens}

# System prompt for PDF OCR processing
SYSTEM You are an expert OCR system designed to accurately extract text from PDF images.
SYSTEM Your primary task is to transcribe all visible text in the images, preserving the original formatting as much as possible.
SYSTEM Analyze the document structure including headings, paragraphs, tables, equations, and other elements.
SYSTEM For scientific papers, pay special attention to proper extraction of mathematical notation, citations, and references.
SYSTEM Return ONLY the extracted text content with no additional commentary.
            """

            # Create a temporary modelfile
            with tempfile.NamedTemporaryFile(mode='w', suffix='.modelfile', delete=False) as f:
                f.write(modelfile_content)
                modelfile_path = f.name

            try:
                # Create the model
                logger.info(f"Creating Ollama model '{self.model_name}' based on {base_model}")
                result = ollama.create(model=self.model_name, modelfile=modelfile_path)
                logger.info(f"Model creation result: {result}")
            finally:
                # Clean up the modelfile
                if os.path.exists(modelfile_path):
                    os.remove(modelfile_path)

        except Exception as e:
            logger.error(f"Error creating Ollama model: {e}")
            raise RuntimeError(f"Failed to create Ollama model: {e}")

    def process_pdf_data(self, pdf_data: bytes) -> str:
        """
        Process raw PDF data using Ollama.

        Args:
            pdf_data: Raw PDF data as bytes

        Returns:
            Extracted text from the PDF
        """
        # Save PDF to temporary file
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
        Process a PDF file using Ollama with the olmOCR model.

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

    def _fallback_to_text_extraction(self, file_path: Path) -> str:
        """Extract text directly from PDF using PyPDF2 as a fallback."""
        try:
            import PyPDF2
        except ImportError:
            raise ImportError("PyPDF2 is required for fallback text extraction. Install with 'pip install PyPDF2'")

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
                    if page_text:
                        text_parts.append(page_text)

            # Join all text
            full_text = "\n\n".join(text_parts)
            return full_text

        except Exception as e:
            logger.error(f"Error in PyPDF2 text extraction: {e}")
            raise
