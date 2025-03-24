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
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Hugging Face configuration
HF_TOKEN = os.getenv("HF_TOKEN")
HF_REPO_ID = os.getenv("HF_REPO_ID", "user/ml-papers-qa")

# Default training parameters
DEFAULT_MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"  # Updated to DeepSeek model
DEFAULT_MAX_SEQ_LENGTH = 1024  # Can increase for longer reasoning traces
DEFAULT_LORA_RANK = 64  # Larger rank = smarter, but slower
DEFAULT_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]
DEFAULT_SYSTEM_PROMPT = """You are an AI assistant. Follow this response format:
<think>
First, think through the question step-by-step in this section.
Consider what the user is asking, relevant concepts, and how to structure your answer.
This section should contain your analytical process and reasoning.
</think>

After the think section, provide your direct answer without any tags.
Your answer should be clear, concise, and directly address the question.
"""

# Default training hyperparameters
DEFAULT_TRAINING_ARGS = {
    "use_vllm": True,  # Use vLLM for fast inference
    "learning_rate": 5e-6,
    "adam_beta1": 0.9,
    "adam_beta2": 0.99,
    "weight_decay": 0.1,
    "warmup_ratio": 0.1,
    "lr_scheduler_type": "cosine",
    "optim": "adamw_8bit",
    "logging_steps": 1,
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 1,  # Increase to 4 for smoother training
    "num_generations": 4,  # Decrease if out of memory
    "max_prompt_length": 256,
    "max_steps": 2000,
    "save_steps": 50,
    "max_grad_norm": 0.1,
    "report_to": "none",  # Can use Weights & Biases
    "output_dir": "outputs",
}

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
