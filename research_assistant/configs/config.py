"""
Configuration settings for the research assistant project.
"""

import os
from pathlib import Path

# Project structure
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

# Create directories if they don't exist
os.makedirs(RAW_DATA_DIR, exist_ok=True)
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Model settings
BASE_MODEL = "meta-llama/Llama-2-7b-hf"  # Example, adjust as needed
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"

# Training settings
BATCH_SIZE = 8
LEARNING_RATE = 5e-5
NUM_EPOCHS = 3
GRADIENT_ACCUMULATION_STEPS = 4
WARMUP_STEPS = 100
LOGGING_STEPS = 100
EVAL_STEPS = 500
MAX_SEQ_LENGTH = 1024

# Hardware settings
DEVICE = "cuda"  # or "cpu"
MIXED_PRECISION = "fp16"  # or "no" for full precision

# Weights & Biases settings
WANDB_PROJECT = "research-assistant"
WANDB_ENTITY = "denissud"  # Set to your username or organization 