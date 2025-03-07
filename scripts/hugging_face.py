# papertuner/scripts/hugging_face.py
import sys
import os
import logging
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from papertuner import DatasetPipeline
import os
# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Get tokens from environment
google_api_key = os.environ.get("GOOGLE_API_KEY")
huggingface_token = os.environ.get("HF_TOKEN")

if not google_api_key:
    raise ValueError("GOOGLE_API_KEY environment variable is required")
    
if not huggingface_token:
    raise ValueError("HF_TOKEN environment variable is required")

# Install required dependencies
try:
    import PyPDF2
except ImportError:
    logger.info("Installing PyPDF2 for text extraction fallback...")
    import subprocess
    subprocess.check_call(["pip", "install", "PyPDF2"])
    import PyPDF2

try:
    import datasets
except ImportError:
    logger.info("Installing datasets package for HuggingFace integration...")
    import subprocess
    subprocess.check_call(["pip", "install", "datasets"])
    import datasets

# Create a pipeline with the HuggingFace formatter
pipeline = DatasetPipeline(
    ocr_type="gemini",
    source_type="arxiv",
    formatter_type="huggingface",
    api_key=google_api_key,
    ocr_kwargs={
        "model": "gemini-1.5-pro-latest",  # More capable model
        "max_retries": 3,
        "fallback_to_text_extraction": True,
    },
    formatter_kwargs={
        "save_locally": True, 
        "push_to_hub": True, 
        "hub_token": huggingface_token, 
        "hub_dataset_name": "densud2/quantum-papers"
    },
    retry_failed=True,
    max_retries=2,
)

# Generate the dataset
try:
    entries = pipeline.generate(
        query="quantum computing",
        output_path="datasets/quantum_papers",
        max_papers=5,
        save_intermediate=True,
        fail_on_empty=False,
    )
    
    logger.info(f"Successfully processed {len(entries)} papers")
    
except Exception as e:
    logger.error(f"Error generating dataset: {e}")
    raise
