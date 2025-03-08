import logging
from typing import Optional, Union, List
from pathlib import Path
from tqdm import tqdm
import torch
from transformers import AutoProcessor, AutoModelForVision2Seq
from huggingface_hub import hf_hub_download
from PIL import Image
from .base import BaseOCR

# Get logger
logger = logging.getLogger(__name__)

class HuggingFaceOCR(BaseOCR):
    """OCR implementation using Hugging Face vision-language models."""

    def __init__(self, model_name: str = "ibm-granite/granite-vision-3.2-2b"):
        """
        Initialize the Hugging Face OCR processor.

        Args:
            model_name: Name of the Hugging Face model to use
        """
        # Check if CUDA is available
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load the model and processor
        self.model_name = model_name
        self.processor = AutoProcessor.from_pretrained(model_name, use_fast=True)
        self.model = AutoModelForVision2Seq.from_pretrained(model_name).to(self.device)

        logger.debug(f"HuggingFaceOCR initialized with model={model_name}, device={self.device}")

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
        Process a PDF file using Hugging Face model.

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

    def _process_image_chunk(self, images: List[Image.Image], start_page: int) -> str:
        """
        Process a chunk of images using the Hugging Face model.

        Args:
            images: List of PIL images
            start_page: Starting page number for this chunk

        Returns:
            Extracted text from the images
        """
        # Create a combined prompt with all images
        prompt = f"Extract all text from these {len(images)} PDF pages. Return only the extracted text content."

        # Process each image individually with progress
        progress_desc = f"Processing {len(images)} images"
        responses = []
        for idx, img in enumerate(tqdm(images, desc=progress_desc, leave=False)):
            # Prepare the conversation
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": img},
                        {"type": "text", "text": prompt},
                    ],
                },
            ]

            # Tokenize the conversation
            inputs = self.processor.apply_chat_template(
                conversation,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt"
            ).to(self.device)

            # Generate the response
            output = self.model.generate(**inputs, max_new_tokens=100)
            response = self.processor.decode(output[0], skip_special_tokens=True)
            responses.append(response)

        # Combine responses
        combined_text = "\n\n".join(responses)
        return combined_text
