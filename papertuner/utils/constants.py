"""
Constants used throughout the papertuner package.
"""
import os
from pathlib import Path

# System prompt for model training and inference
SYSTEM_PROMPT = """
Respond in the following format:
<reasoning>
...
</reasoning>
<answer>
...
</answer>
"""

# XML format template for Chain of Thought responses
XML_COT_FORMAT = """\
<reasoning>
{reasoning}
</reasoning>
<answer>
{answer}
</answer>
"""

# Alternative format with <think> tags
THINK_FORMAT = """\
<think>
{thinking}
</think>
{answer}
"""

# Default data directories
DATA_DIR = Path(os.getenv("PAPERTUNER_DATA_DIR", "data"))
RAW_DIR = DATA_DIR / "raw_dataset"
PROCESSED_DIR = DATA_DIR / "processed_dataset"

# Default model settings
DEFAULT_MODEL_NAME = "unsloth/DeepSeek-R1-Distill-Qwen-1.5B-unsloth-bnb-4bit"
DEFAULT_MAX_SEQ_LENGTH = 8192
DEFAULT_LORA_RANK = 64
DEFAULT_MAX_STEPS = 250
DEFAULT_SAVE_STEPS = 250
DEFAULT_GPU_MEMORY_UTILIZATION = 0.5 