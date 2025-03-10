"""
Example usage of the PaperTuner library.
"""

from papertuner import setup_logging
from papertuner.ocr import GeminiOCR, PyMuPDFOCR
from papertuner.sources import ArxivClient
from papertuner.data import HGDataset
import os
import logging

def main():
    # Configure logging
    setup_logging(logging.INFO)

    # Get API keys from environment variables
    hf_token = os.environ.get("HF_TOKEN")

    if not hf_token:
        logging.warning("HF_TOKEN environment variable is not set. Dataset upload will not be available.")

    # Initialize OCR
    ocr = PyMuPDFOCR()

    # Initialize ArXiv client
    client = ArxivClient(
        max_papers=500,
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
        ocr=ocr
    )

if __name__ == "__main__":
    main()
