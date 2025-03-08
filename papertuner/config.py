"""
Configuration management for PaperTuner.
"""

import os
import yaml
from pathlib import Path

DEFAULT_CONFIG = {
    "max_papers": 100,
    "min_text_length": 500,
    "output_path": "dataset",
    "save_to_disk": True,
    "upload": False,
}

def load_config(config_path=None):
    """Load configuration from a YAML file."""
    config = DEFAULT_CONFIG.copy()
    
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r') as f:
            user_config = yaml.safe_load(f)
            if user_config:
                config.update(user_config)
    
    return config 