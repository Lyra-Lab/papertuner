import os
import sys
import logging
import argparse
import subprocess
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from papertuner import DatasetPipeline

# Set up argument parsing
parser = argparse.ArgumentParser(description="Generate a dataset using HuggingFace Transformers OCR")
parser.add_argument("--model-name", default="allenai/olmOCR-7B-0225-preview",
                   help="HuggingFace model name/path (default: allenai/olmOCR-7B-0225-preview)")
parser.add_argument("--device", default=None,
                   help="Device to use (cpu, cuda, mps). If None, will try to use CUDA if available.")
parser.add_argument("--torch-dtype", default="auto",
                   help="PyTorch dtype to use (auto, float16, bfloat16, float32)")
parser.add_argument("--max-length", type=int, default=4096,
                   help="Maximum length of generated text")
parser.add_argument("--query", default="quantum computing",
                   help="Search query for papers (default: 'quantum computing')")
parser.add_argument("--max-papers", type=int, default=2,
                   help="Maximum number of papers to process (default: 2)")
parser.add_argument("--output-path", default="datasets/quantum_papers_hf",
                   help="Path to save the dataset (default: datasets/quantum_papers_hf)")
parser.add_argument("--hub-dataset-name", default="",
                   help="HuggingFace Hub dataset name (default: empty, no push)")
args = parser.parse_args()

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Install necessary dependencies
try:
    import torch
except ImportError:
    logger.info("Installing PyTorch...")
    subprocess.check_call(["pip", "install", "torch"])
    import torch

try:
    import transformers
except ImportError:
    logger.info("Installing Transformers...")
    subprocess.check_call(["pip", "install", "transformers"])
    import transformers

try:
    import pdf2image
except ImportError:
    logger.info("Installing pdf2image package...")
    # Install poppler-utils if on Linux
    if os.name == 'posix':
        try:
            subprocess.check_call(["apt-get", "update"])
            subprocess.check_call(["apt-get", "install", "-y", "poppler-utils"])
        except Exception as e:
            logger.warning(f"Failed to install poppler-utils: {e}")
            logger.warning("You may need to install poppler-utils manually")

    subprocess.check_call(["pip", "install", "pdf2image"])
    import pdf2image

try:
    from PIL import Image
except ImportError:
    logger.info("Installing Pillow...")
    subprocess.check_call(["pip", "install", "Pillow"])
    from PIL import Image

try:
    import datasets
except ImportError:
    logger.info("Installing datasets package...")
    subprocess.check_call(["pip", "install", "datasets"])
    import datasets

# Get HuggingFace token from environment
huggingface_token = os.environ.get("HF_TOKEN")

# Determine if we should push to HuggingFace Hub
push_to_hub = bool(args.hub_dataset_name and huggingface_token)
if args.hub_dataset_name and not huggingface_token:
    logger.warning("HF_TOKEN environment variable not set but hub_dataset_name provided. Will not push to HuggingFace Hub.")

# Create a pipeline with the HuggingFace Transformers OCR
pipeline = DatasetPipeline(
    ocr_type="transformers",
    source_type="arxiv",
    formatter_type="huggingface",
    ocr_kwargs={
        "model_name": args.model_name,
        "device": args.device,
        "torch_dtype": args.torch_dtype,
        "max_length": args.max_length,
        "use_auth_token": huggingface_token  # Pass token for accessing gated models
    },
    formatter_kwargs={
        "save_locally": True,
        "push_to_hub": push_to_hub,
        "hub_token": huggingface_token,
        "hub_dataset_name": args.hub_dataset_name if args.hub_dataset_name else None
    },
    retry_failed=True,
    max_retries=2,
    parallel=False  # Processing is GPU-bound, so parallel wouldn't help much
)

# Print some device information
if torch.cuda.is_available():
    logger.info(f"CUDA is available: {torch.cuda.get_device_name(0)}")
    logger.info(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# Generate the dataset
try:
    entries = pipeline.generate(
        query=args.query,
        output_path=args.output_path,
        max_papers=args.max_papers,
        save_intermediate=True,
        fail_on_empty=False,
    )

    logger.info(f"Successfully processed {len(entries)} papers")

except Exception as e:
    logger.error(f"Error generating dataset: {e}")
    raise
