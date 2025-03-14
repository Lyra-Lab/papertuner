"""
Phi-4 (14B) GRPO Training Script

Simplified and organized version maintaining core functionality.
"""

import os
import re
import logging
from typing import List, Optional
import torch
import wandb
from datasets import Dataset
from trl import GRPOConfig, GRPOTrainer
from vllm import SamplingParams
import unsloth
from unsloth import FastLanguageModel, is_bfloat16_supported
from config import Config, default_config

#Configuration 
class TrainingConfig:
    """Container for training hyperparameters"""
    def __init__(self):
        self.model_name = "Qwen/Qwen2.5-3B-Instruct"
        self.max_seq_length = 2048
        self.lora_rank = 8
        self.learning_rate = 5e-5
        self.num_epochs = 3
        self.max_steps = 1000
        self.batch_size = 4
        self.grad_accum_steps = 2
        self.num_generations = 5
        self.output_dir = "output_dir"
        self.dataset_name = "gsm8k"
        self.dataset_config = "main"
        self.split = "train"

TRAIN_CONFIG = TrainingConfig()
WANDB_CONFIG = {"project": "papertuner", "name": "first run", "enabled": True}

#Helper functions 
def configure_logging() -> logging.Logger:
    """Set up standardized logging"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

logger = configure_logging()

XML_TEMPLATE = """<reasoning>{reasoning}</reasoning>\n<answer>{answer}</answer>"""

def extract_answer(text: str, mode: str = "xml") -> Optional[str]:
    """Unified answer extraction"""
    if mode == "xml":
        parts = text.split("</answer>")
        return parts[0].split("<answer>")[-1].strip() if len(parts) > 1 else None
    return text.split("####")[1].strip() if "####" in text else None

#Core components 
class RewardSystem:
    """Container for reward calculation functions"""
    
    @staticmethod
    def correctness(prompts, completions, answers, **_) -> List[float]:
        """Primary correctness reward (2.0 for correct answers)"""
        responses = [extract_answer(c[0]['content']) for c in completions]
        return [2.0 if r == a else 0.0 for r, a in zip(responses, answers)]

    @staticmethod
    def format_strict(completions, **_) -> List[float]:
        """Strict XML format validation reward"""
        pattern = r"^<reasoning>\n.*?\n</reasoning>\n<answer>\n.*?\n</answer>\n$"
        return [0.5 if re.match(pattern, c[0]["content"], re.DOTALL) else 0.0 for c in completions]

class ModelManager:
    """Handle model loading and saving operations"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.model, self.tokenizer = self._load_base_model()
        
    def _load_base_model(self):
        """Load base model with 4-bit quantization"""
        return FastLanguageModel.from_pretrained(
            model_name=self.config.model_name,
            max_seq_length=self.config.max_seq_length,
            load_in_4bit=True,
            max_lora_rank=self.config.lora_rank
        )

    def save_lora(self, path: str):
        """Save LoRA adapters"""
        self.model.save_lora(path)
        logger.info(f"Saved LoRA weights to {path}")

def initialize_wandb(config: TrainingConfig):
    """Initialize Weights & Biases logging"""
    if not WANDB_CONFIG["enabled"]:
        return

    wandb.init(
        project=WANDB_CONFIG["project"],
        name=WANDB_CONFIG["name"],
        config=vars(config)
    )
    logger.info(f"Initialized W&B: {wandb.run.name}")

# Main pipeline
def main_training_flow():
    """Orchestrate the full training process"""
    logger.info("Initializing training pipeline")
    
    # Configuration
    manager = ModelManager(TRAIN_CONFIG)
    initialize_wandb(TRAIN_CONFIG)
    
    # Dataset setup
    dataset = load_dataset(TRAIN_CONFIG.dataset_name, TRAIN_CONFIG.dataset_config)[TRAIN_CONFIG.split]
    logger.info(f"Loaded dataset with {len(dataset)} examples")

    # Training setup
    trainer = GRPOTrainer(
        model=manager.model,
        processing_class=manager.tokenizer,
        reward_funcs=[RewardSystem.correctness, RewardSystem.format_strict],
        args=GRPOConfig(
            learning_rate=TRAIN_CONFIG.learning_rate,
            per_device_train_batch_size=TRAIN_CONFIG.batch_size,
            gradient_accumulation_steps=TRAIN_CONFIG.grad_accum_steps,
            num_train_epochs=TRAIN_CONFIG.num_epochs,
            max_steps=TRAIN_CONFIG.max_steps,
            output_dir=TRAIN_CONFIG.output_dir,
            report_to="wandb" if WANDB_CONFIG["enabled"] else "none",
            bf16=is_bfloat16_supported(),
        ),
        train_dataset=dataset,
    )

    # Execute training
    logger.info("Starting model training")
    trainer.train()
    
    # Save artifacts
    manager.save_lora(os.path.join(TRAIN_CONFIG.output_dir, "grpo_lora"))
    
    logger.info("Training completed successfully")

if __name__ == "__main__":
    main_training_flow()
