import sys
import os
import logging
import subprocess

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import argparse
from papertuner import DatasetPipeline

# Set up argument parsing
parser = argparse.ArgumentParser(description="Generate a dataset using Ollama OCR")
parser.add_argument("--ollama-host", default="http://127.0.0.1:11434",
                   help="Ollama API host URL (default: http://127.0.0.1:11434)")
parser.add_argument("--model-name", default="olmocr",
                   help="Name for the Ollama model (default: olmocr)")
parser.add_argument("--base-model", default="qwen2-vl-7b-instruct",
                   help="Base model to use (default: qwen2-vl-7b-instruct)")
parser.add_argument("--create-model", action="store_true",
                   help="Create the model if it doesn't exist")
parser.add_argument("--query", default="quantum computing",
                   help="Search query for papers (default: 'quantum computing')")
parser.add_argument("--max-papers", type=int, default=3,
                   help="Maximum number of papers to process (default: 3)")
parser.add_argument("--output-path", default="datasets/quantum_papers_ollama",
                   help="Path to save the dataset (default: datasets/quantum_papers_ollama)")
parser.add_argument("--hub-dataset-name", default="",
                   help="HuggingFace Hub dataset name (default: empty, no push)")
args = parser.parse_args()

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set environment variables for Ollama host
os.environ["OLLAMA_HOST"] = args.ollama_host
logger.info(f"Setting OLLAMA_HOST environment variable to {args.ollama_host}")

# Install necessary dependencies
try:
    import ollama
except ImportError:
    logger.info("Installing ollama package...")
    subprocess.check_call(["pip", "install", "ollama"])
    import ollama

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

# Check if Ollama is running
try:
    logger.info(f"Checking if Ollama is running at {args.ollama_host}...")
    models = ollama.list()
    logger.info(f"Ollama is running. Available models: {[model['name'] for model in models.get('models', [])]}")
except Exception as e:
    logger.error(f"Error connecting to Ollama at {args.ollama_host}: {e}")
    logger.error(f"Make sure Ollama is running with 'ollama serve' and accessible at {args.ollama_host}")
    raise RuntimeError(f"Ollama is not running or not accessible at {args.ollama_host}")

# Create a pipeline with the Ollama OCR
pipeline = DatasetPipeline(
    ocr_type="ollama",
    source_type="arxiv",
    formatter_type="huggingface",
    ocr_kwargs={
        "model_name": args.model_name,
        "create_model": args.create_model,
        "base_model": args.base_model,
        "temperature": 0.2,
        "max_tokens": 8192,
        # No need to specify ollama_host here as it's already set via env var
    },
    formatter_kwargs={
        "save_locally": True,
        "push_to_hub": push_to_hub,
        "hub_token": huggingface_token,
        "hub_dataset_name": args.hub_dataset_name if args.hub_dataset_name else None
    },
    retry_failed=True,
    max_retries=2,
    parallel=False  # Set to False to avoid overwhelming the local Ollama instance
)

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
