"""
Example usage of the PaperTuner library.
"""

from papertuner.ocr import GeminiOCR
from papertuner.sources import ArxivClient
from papertuner.data import HGDataset
import os
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def main():
    # Get API keys from environment variables
    gemini_api_key = os.environ.get("GEMINI_API_KEY")
    hf_token = os.environ.get("HF_TOKEN")

    if not gemini_api_key:
        raise ValueError("GEMINI_API_KEY environment variable is required")

    if not hf_token:
        logging.warning("HF_TOKEN environment variable is not set. Dataset upload will not be available.")

    # Initialize OCR
    gemini_ocr = GeminiOCR(gemini_api_key, hf_token)

    # Initialize ArXiv client
    client = ArxivClient(
        max_papers=100,
        queries=[
            "deep learning",
            "natural language processing",
            "unsupervised learning",
            "reinforcement learning"
        ]
    )

    # Initialize dataset
    dataset = HGDataset(
        name="Quantum mechanics",
        remote_username="densud2",
        hg_token=hf_token,
    )

    # Generate and upload dataset
    dataset.generate(
        upload=True,
        output_path="temp",
        save_to_disk=True,
        client=client,
        ocr=gemini_ocr
    )

if __name__ == "__main__":
    main()
