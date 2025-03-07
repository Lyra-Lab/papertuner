"""OCR implementation using HuggingFace Transformers library with vision-language models."""

import os
import tempfile
import base64
import time
import logging
import io
from typing import Optional, Union, Dict, Any, List, Tuple
from pathlib import Path

# Conditionally import dependencies
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from transformers import AutoModelForVision2Seq, AutoProcessor
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

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

from .arxiv_integrated import ArxivIntegratedOCR

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HFTransformersOCR(ArxivIntegratedOCR):
    """OCR implementation using HuggingFace Transformers with vision-language models."""

    def __init__(
        self,
        model_name: str = "allenai/olmOCR-7B-0225-preview",
        device: Optional[str] = None,
        torch_dtype: str = "auto",
        max_length: int = 4096,
        batch_size: int = 1,
        dpi: int = 300,
        max_retries: int = 2,
        use_auth_token: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize the HuggingFace Transformers OCR processor.

        Args:
            model_name: HuggingFace model name/path (default: "allenai/olmOCR-7B-0225-preview")
            device: Device to use ("cpu", "cuda", "mps", etc.). If None, will try to use CUDA if available.
            torch_dtype: PyTorch dtype to use ("auto", "float16", "bfloat16", "float32")
            max_length: Maximum length of generated text
            batch_size: Number of images to process in a single batch
            dpi: DPI to use when converting PDFs to images
            max_retries: Maximum number of retries for failed operations
            use_auth_token: HuggingFace token for accessing gated models (or use HF_TOKEN env var)
            **kwargs: Additional parameters to pass to the model

        Raises:
            ImportError: If required dependencies are not installed
            RuntimeError: If the model cannot be loaded
        """
        # Check if required dependencies are installed
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required. Install with 'pip install torch'")

        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("Transformers is required. Install with 'pip install transformers'")

        if not PDF2IMAGE_AVAILABLE:
            raise ImportError("pdf2image is required. Install with 'pip install pdf2image'")

        if not PIL_AVAILABLE:
            raise ImportError("Pillow is required. Install with 'pip install Pillow'")

        # Model parameters
        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size
        self.dpi = dpi
        self.max_retries = max_retries
        self.kwargs = kwargs

        # Get device (CPU/CUDA/MPS)
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            if self.device == "cpu" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = "mps"  # Use Apple Silicon GPU if available
        else:
            self.device = device

        # Get torch dtype
        if torch_dtype == "auto":
            if self.device == "cuda" and torch.cuda.is_available():
                # Use BF16 if available, otherwise FP16
                self.torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            else:
                self.torch_dtype = torch.float32
        elif torch_dtype == "float16":
            self.torch_dtype = torch.float16
        elif torch_dtype == "bfloat16":
            self.torch_dtype = torch.bfloat16
        else:
            self.torch_dtype = torch.float32

        # Get HuggingFace token
        self.use_auth_token = use_auth_token or os.environ.get("HF_TOKEN")

        # Load the model and processor
        logger.info(f"Loading model {model_name} on {self.device} with {self.torch_dtype}...")
        self._load_model_and_processor()

    def _load_model_and_processor(self):
        """Load the model and processor from HuggingFace."""
        try:
            # Load the processor
            self.processor = AutoProcessor.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                token=self.use_auth_token
            )

            # Load the model
            self.model = AutoModelForVision2Seq.from_pretrained(
                self.model_name,
                torch_dtype=self.torch_dtype,
                device_map=self.device if self.device != "mps" else None,  # device_map not supported for MPS
                trust_remote_code=True,
                token=self.use_auth_token
            )

            # Move model to device if using MPS (Apple Silicon)
            if self.device == "mps":
                self.model = self.model.to(self.device)

            logger.info(f"Successfully loaded model and processor")

        except Exception as e:
            logger.error(f"Error loading model and processor: {e}")
            raise RuntimeError(f"Failed to load model {self.model_name}: {e}")

    def process_pdf_data(self, pdf_data: bytes) -> str:
        """
        Process raw PDF data using the HuggingFace model.

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
        Process a PDF file using the HuggingFace model.

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
                dpi=self.dpi,
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

        # Process the images in batches
        all_text = []
        batch_size = self.batch_size

        for i in range(0, len(images), batch_size):
            batch = images[i:i+batch_size]
            logger.info(f"Processing pages {i+1}-{i+len(batch)} of {len(images)}")

            batch_text = self._process_images(batch)
            all_text.append(batch_text)

        # Combine all extracted text
        return "\n\n".join(all_text)

    def _process_images(self, images: List[Image.Image]) -> str:
        """
        Process a list of images using the HuggingFace model.

        Args:
            images: List of PIL images

        Returns:
            Extracted text from the images
        """
        for attempt in range(self.max_retries + 1):
            try:
                # Prepare a system prompt for OCR
                prompt = "Extract all the text from these PDF images. Maintain the structure and formatting as much as possible."

                # Process the images with the model
                inputs = self.processor(
                    images=images,
                    text=prompt,
                    return_tensors="pt"
                ).to(self.device)

                # Generate text
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_length=self.max_length,
                        do_sample=False,
                        **self.kwargs
                    )

                # Decode the outputs
                decoded_text = self.processor.decode(outputs[0], skip_special_tokens=True)

                # Clean up the text (remove the prompt if it's included in the output)
                if prompt in decoded_text:
                    decoded_text = decoded_text.replace(prompt, "").strip()

                return decoded_text

            except Exception as e:
                logger.warning(f"Error on attempt {attempt+1}: {e}")
                if attempt < self.max_retries:
                    # Wait before retrying
                    time.sleep(2)
                else:
                    logger.error(f"Failed to process images after {self.max_retries + 1} attempts")
                    raise RuntimeError(f"Failed to process images: {e}")

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
