"""Configuration management for PaperTuner."""

import os
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("papertuner")

# Base directories
DEFAULT_BASE_DIR = Path.home() / ".papertuner"
DATA_DIR = Path(os.getenv("PAPERTUNER_DATA_DIR", DEFAULT_BASE_DIR / "data"))
RAW_DIR = DATA_DIR / "raw_dataset"
PROCESSED_DIR = DATA_DIR / "processed_dataset"

# API configuration
API_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Hugging Face configuration
HF_TOKEN = os.getenv("HF_TOKEN")
HF_REPO_ID = os.getenv("HF_REPO_ID", "user/ml-papers-qa")

# Default training parameters
DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
DEFAULT_MAX_SEQ_LENGTH = 1024
DEFAULT_LORA_RANK = 64
DEFAULT_SYSTEM_PROMPT = """You are an AI assistant specialized in machine learning concepts. Follow this response format:
<think>
First, think through the question step-by-step in this section.
Consider what the user is asking, relevant concepts, and how to structure your answer.
This section should contain your analytical process and reasoning.
</think>

After the think section, provide your direct answer without any tags.
Your answer should be clear, concise, and directly address the question.
"""

def setup_dirs():
    """Create necessary directories for data storage."""
    try:
        RAW_DIR.mkdir(parents=True, exist_ok=True)
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        (PROCESSED_DIR / "papers").mkdir(parents=True, exist_ok=True)
        logger.info("Directories set up successfully.")
        return True
    except OSError as e:
        logger.error(f"Failed to setup directories: {e}")
        return False
