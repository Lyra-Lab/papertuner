"""
Configuration for Phi-4 GRPO training
"""

import os
from dataclasses import dataclass, field
from typing import List, Optional, Union

@dataclass
class ModelConfig:
    model_name: str = "unsloth/Phi-4"
    max_seq_length: int = 512
    lora_rank: int = 16
    load_in_4bit: bool = True
    fast_inference: bool = True
    gpu_memory_utilization: float = 0.7
    target_modules: List[str] = field(default_factory=lambda: ["gate_proj", "up_proj", "down_proj"])
    use_gradient_checkpointing: str = "unsloth"
    random_state: int = 3407

@dataclass
class TrainingConfig:
    use_vllm: bool = True
    learning_rate: float = 5e-6
    adam_beta1: float = 0.9
    adam_beta2: float = 0.99
    weight_decay: float = 0.1
    warmup_ratio: float = 0.1
    lr_scheduler_type: str = "cosine"
    optim: str = "paged_adamw_8bit"
    logging_steps: int = 1
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 1
    num_generations: int = 6
    max_prompt_length: int = 256
    max_completion_length: int = 200
    num_train_epochs: Optional[int] = 1
    max_steps: int = 100
    save_steps: int = 250
    max_grad_norm: float = 0.1
    output_dir: str = "outputs"
    
@dataclass
class DataConfig:
    dataset_name: str = "openai/gsm8k"
    dataset_config: str = "main"
    split: str = "train"
    
@dataclass
class InferenceConfig:
    temperature: float = 0.8
    top_p: float = 0.95
    max_tokens: int = 1024
    
@dataclass
class WandbConfig:
    enabled: bool = True
    project: str = "phi4-grpo"
    name: Optional[str] = None
    tags: List[str] = field(default_factory=lambda: ["reasoning", "grpo"])
    
@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)
    
    # System prompt for reasoning format
    system_prompt: str = """
Respond in the following format:
<reasoning>
...
</reasoning>
<answer>
...
</answer>
"""

# Default configuration
default_config = Config() 