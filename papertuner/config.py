"""Configuration management for PaperTuner."""

import os
from pathlib import Path
import logging
import json
from typing import Dict, List, Any, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("papertuner")

# Base directories
DEFAULT_BASE_DIR = Path.home() / ".papertuner"
DATA_DIR = Path(os.getenv("PAPERTUNER_DATA_DIR", DEFAULT_BASE_DIR / "data"))
RAW_DIR = DATA_DIR / "raw_dataset"
PROCESSED_DIR = DATA_DIR / "processed_dataset"
CACHE_DIR = DATA_DIR / "cache"
MODEL_DIR = DATA_DIR / "models"

# API configuration
API_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

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
    "bf16": True,  # Use bfloat16 precision if available
    "fp16": False,  # Fallback to fp16 if bf16 not available
    "mixed_precision": "bf16",  # Default to BF16
}

# Data sources configuration
DEFAULT_SOURCES = {
    "arxiv": {
        "enabled": True,
        "max_results": 100,
        "sort_by": "relevance",
        "sort_order": "descending"
    },
    "pubmed": {
        "enabled": False,
        "max_results": 50,
        "sort_by": "relevance",
        "api_key": os.getenv("PUBMED_API_KEY", "")
    },
    "semantic_scholar": {
        "enabled": False,
        "max_results": 50,
        "fields": ["title", "abstract", "year", "authors", "citationCount"]
    },
    "ieee": {
        "enabled": False,
        "max_results": 30,
        "api_key": os.getenv("IEEE_API_KEY", "")
    },
    "acm": {
        "enabled": False,
        "max_results": 30,
        "api_key": os.getenv("ACM_API_KEY", "")
    },
    "local": {
        "enabled": False,
        "paths": [],  # List of local paths to PDF files
    }
}

# LLM configuration for data processing
DEFAULT_LLM_CONFIG = {
    "provider": "gemini",  # Options: gemini, openai, anthropic
    "model": "gemini-2.0-flash",  # Default model
    "fallback_provider": "openai",  # Fallback if primary fails
    "fallback_model": "gpt-4o-mini",
    "temperature": 0.3,
    "top_p": 0.9,
    "max_tokens": 1500,
    "timeout": 30,  # Request timeout in seconds
    "max_retries": 3,  # Max retry attempts
}

# Domain-specific configurations for paper processing
DEFAULT_DOMAIN_CONFIGS = {
    "machine_learning": {
        "sections_of_interest": ["methodology", "approach", "experiments", "results", "ablation studies"],
        "domain_keywords": ["neural network", "deep learning", "transformer", "attention", "embedding"],
        "citation_importance": 0.7,
    },
    "natural_language_processing": {
        "sections_of_interest": ["approach", "model", "experiments", "results", "discussion"],
        "domain_keywords": ["language model", "text", "embeddings", "corpus", "tokenizer"],
        "citation_importance": 0.6,
    },
    "computer_vision": {
        "sections_of_interest": ["methodology", "architecture", "experiments", "results", "ablation studies"],
        "domain_keywords": ["image", "vision", "object detection", "segmentation", "recognition"],
        "citation_importance": 0.6,
    },
    "reinforcement_learning": {
        "sections_of_interest": ["method", "algorithm", "experiments", "results", "discussion"],
        "domain_keywords": ["policy", "reward", "agent", "environment", "q-learning"],
        "citation_importance": 0.5,
    },
}

class Config:
    """Configuration manager for PaperTuner."""
    
    def __init__(self, config_file: Optional[str] = None):
        """Initialize configuration, optionally from a file."""
        self.config_file = config_file
        self.data = self._load_config()
        
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file or use defaults."""
        if self.config_file and os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    config = json.load(f)
                logger.info(f"Loaded configuration from {self.config_file}")
                return config
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"Failed to load config file: {e}, using defaults")
        
        # Return default configuration
        return {
            "data_dir": str(DATA_DIR),
            "training_args": DEFAULT_TRAINING_ARGS,
            "llm_config": DEFAULT_LLM_CONFIG,
            "sources": DEFAULT_SOURCES,
            "domain_configs": DEFAULT_DOMAIN_CONFIGS,
            "system_prompt": DEFAULT_SYSTEM_PROMPT,
            "model_name": DEFAULT_MODEL_NAME,
            "max_seq_length": DEFAULT_MAX_SEQ_LENGTH,
            "lora_rank": DEFAULT_LORA_RANK,
            "target_modules": DEFAULT_TARGET_MODULES,
        }
    
    def save(self, path: Optional[str] = None) -> bool:
        """Save configuration to file."""
        save_path = path or self.config_file
        if not save_path:
            logger.warning("No path provided for saving configuration")
            return False
        
        try:
            with open(save_path, 'w') as f:
                json.dump(self.data, f, indent=2)
            logger.info(f"Configuration saved to {save_path}")
            return True
        except IOError as e:
            logger.error(f"Failed to save configuration: {e}")
            return False
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value."""
        return self.data.get(key, default)
    
    def set(self, key: str, value: Any) -> None:
        """Set a configuration value."""
        self.data[key] = value
    
    def update(self, updates: Dict[str, Any]) -> None:
        """Update multiple configuration values."""
        self.data.update(updates)

def setup_dirs():
    """Create necessary directories for data storage."""
    try:
        RAW_DIR.mkdir(parents=True, exist_ok=True)
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        (PROCESSED_DIR / "papers").mkdir(parents=True, exist_ok=True)
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        logger.info("Directories set up successfully.")
        return True
    except OSError as e:
        logger.error(f"Failed to setup directories: {e}")
        return False
