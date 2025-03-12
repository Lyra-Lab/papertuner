"""
Helper script to initialize Weights & Biases tracking.
"""

import os
from datetime import datetime

import wandb

from configs.config import WANDB_PROJECT, WANDB_ENTITY

def init_wandb(run_name=None, config=None):
    """
    Initialize a Weights & Biases run.
    
    Args:
        run_name (str, optional): Name for this run. If None, generates a timestamp-based name.
        config (dict, optional): Configuration dictionary to log.
        
    Returns:
        wandb.Run: The initialized run object.
    """
    if run_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"run_{timestamp}"
    
    return wandb.init(
        project=WANDB_PROJECT,
        entity=WANDB_ENTITY,
        name=run_name,
        config=config,
    )

def log_model_metadata(model_name, model_parameters=None, model_config=None):
    """
    Log model metadata to the current W&B run.
    
    Args:
        model_name (str): The name of the model.
        model_parameters (int, optional): Number of parameters in the model.
        model_config (dict, optional): Model configuration.
    """
    metadata = {
        "model_name": model_name,
    }
    
    if model_parameters is not None:
        metadata["model_parameters"] = model_parameters
    
    if model_config is not None:
        metadata.update(model_config)
    
    wandb.config.update(metadata) 