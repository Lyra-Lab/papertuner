"""
Example usage of the PaperTuner library.
"""

from papertuner import setup_logging
from papertuner.ocr import GeminiOCR, PyMuPDFOCR
from papertuner.sources import ArxivClient
from papertuner.data import HGDataset
import os
import logging
import argparse

def main():
    parser = argparse.ArgumentParser(description="Generate a dataset from research papers")
    parser.add_argument("--resume", action="store_true", help="Resume from previous run")
    parser.add_argument("--output", type=str, default="dataset", help="Output directory")
    parser.add_argument("--max-papers", type=int, default=500, help="Maximum number of papers to fetch")
    parser.add_argument("--min-text", type=int, default=500, help="Minimum text length to include")
    parser.add_argument("--no-upload", action="store_true", help="Disable uploading to HuggingFace")
    parser.add_argument("--log-level", type=str, default="INFO", 
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"], 
                       help="Logging level")
    args = parser.parse_args()

    # Configure logging
    log_level = getattr(logging, args.log_level)
    setup_logging(log_level)

    # Get API keys from environment variables
    hf_token = os.environ.get("HF_TOKEN")

    if not hf_token and not args.no_upload:
        logging.warning("HF_TOKEN environment variable is not set. Dataset upload will not be available.")
        args.no_upload = True

    # Initialize OCR
    ocr = PyMuPDFOCR()

    # Initialize ArXiv client
    client = ArxivClient(
        max_papers=args.max_papers,
        queries=[
            "deep learning",
            "natural language processing",
            "unsupervised learning",
            "reinforcement learning"
        ]
    )

    # Initialize dataset
    dataset = HGDataset(
        name="ML Research Papers",
        remote_username="densud2",
        hg_token=hf_token,
    )

    # Generate and upload dataset
    try:
        result = dataset.generate(
            upload=not args.no_upload,
            output_path=args.output,
            save_to_disk=True,
            client=client,
            ocr=ocr,
            min_text_length=args.min_text,
            resume=args.resume
        )
        
        # Print dataset summary
        logging.info(dataset.describe())
        
    except Exception as e:
        logging.error(f"Error generating dataset: {e}", exc_info=True)
        return 1
        
    return 0

if __name__ == "__main__":
    exit(main())
