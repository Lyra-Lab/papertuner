#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Phi-4 (14B) GRPO Training Script

This script trains the Phi-4 model using GRPO (Generative Reward-Prompted Optimization)
for mathematical reasoning tasks.

Original notebook: https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Phi_4_(14B)-GRPO.ipynb
"""

import os
import re
import argparse
import logging
import unsloth
from typing import List, Dict, Any, Optional, Union

import torch
import wandb
from datasets import load_dataset, Dataset
from trl import GRPOConfig, GRPOTrainer
from vllm import SamplingParams

from unsloth import FastLanguageModel, is_bfloat16_supported

from config import Config, default_config

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# XML format constants
XML_COT_FORMAT = """\
<reasoning>
{reasoning}
</reasoning>
<answer>
{answer}
</answer>
"""

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train Phi-4 with GRPO for reasoning")
    
    # Model arguments
    parser.add_argument("--model_name", type=str, default=default_config.model.model_name,
                        help="Model name or path")
    parser.add_argument("--max_seq_length", type=int, default=default_config.model.max_seq_length,
                        help="Maximum sequence length")
    parser.add_argument("--lora_rank", type=int, default=default_config.model.lora_rank,
                        help="LoRA rank")
    
    # Training arguments
    parser.add_argument("--learning_rate", type=float, default=default_config.training.learning_rate,
                        help="Learning rate")
    parser.add_argument("--num_train_epochs", type=int, default=default_config.training.num_train_epochs,
                        help="Number of training epochs")
    parser.add_argument("--max_steps", type=int, default=default_config.training.max_steps,
                        help="Maximum number of training steps")
    parser.add_argument("--batch_size", type=int, default=default_config.training.per_device_train_batch_size,
                        help="Batch size per device")
    parser.add_argument("--gradient_accumulation_steps", type=int, 
                        default=default_config.training.gradient_accumulation_steps,
                        help="Number of gradient accumulation steps")
    parser.add_argument("--num_generations", type=int, default=default_config.training.num_generations,
                        help="Number of generations per prompt")
    parser.add_argument("--output_dir", type=str, default=default_config.training.output_dir,
                        help="Output directory")
    
    # Dataset arguments
    parser.add_argument("--dataset_name", type=str, default=default_config.data.dataset_name,
                        help="Dataset name")
    parser.add_argument("--dataset_config", type=str, default=default_config.data.dataset_config,
                        help="Dataset configuration")
    parser.add_argument("--split", type=str, default=default_config.data.split,
                        help="Dataset split")
    
    # Weights & Biases arguments
    parser.add_argument("--wandb_project", type=str, default=default_config.wandb.project,
                        help="Weights & Biases project name")
    parser.add_argument("--wandb_name", type=str, default=default_config.wandb.name,
                        help="Weights & Biases run name")
    parser.add_argument("--no_wandb", action="store_true",
                        help="Disable Weights & Biases logging")
    
    return parser.parse_args()

def update_config_from_args(config: Config, args) -> Config:
    """Update configuration from command line arguments."""
    # Update model config
    config.model.model_name = args.model_name
    config.model.max_seq_length = args.max_seq_length
    config.model.lora_rank = args.lora_rank
    
    # Update training config
    config.training.learning_rate = args.learning_rate
    config.training.num_train_epochs = args.num_train_epochs
    config.training.max_steps = args.max_steps
    config.training.per_device_train_batch_size = args.batch_size
    config.training.gradient_accumulation_steps = args.gradient_accumulation_steps
    config.training.num_generations = args.num_generations
    config.training.output_dir = args.output_dir
    
    # Update data config
    config.data.dataset_name = args.dataset_name
    config.data.dataset_config = args.dataset_config
    config.data.split = args.split
    
    # Update wandb config
    config.wandb.enabled = not args.no_wandb
    config.wandb.project = args.wandb_project
    if args.wandb_name:
        config.wandb.name = args.wandb_name
    
    return config

def extract_xml_answer(text: str) -> str:
    """Extract answer from XML format."""
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return answer.strip()

def extract_hash_answer(text: str) -> Optional[str]:
    """Extract answer from hash format (GSM8K)."""
    if "####" not in text:
        return None
    return text.split("####")[1].strip()

def get_gsm8k_questions(config: Config) -> Dataset:
    """Load and prepare GSM8K dataset."""
    data = load_dataset(
        config.data.dataset_name, 
        config.data.dataset_config
    )[config.data.split]
    
    data = data.map(lambda x: {
        'prompt': [
            {'role': 'system', 'content': config.system_prompt},
            {'role': 'user', 'content': x['question']}
        ],
        'answer': extract_hash_answer(x['answer'])
    })
    
    return data

# Reward functions
def correctness_reward_func(prompts, completions, answer, **kwargs) -> List[float]:
    """Reward function for correctness of the answer."""
    responses = [completion[0]['content'] for completion in completions]
    q = prompts[0][-1]['content']
    extracted_responses = [extract_xml_answer(r) for r in responses]
    logger.info('-'*20)
    logger.info(f"Question:\n{q}")
    logger.info(f"Answer:\n{answer[0]}")
    logger.info(f"Response:\n{responses[0]}")
    logger.info(f"Extracted:\n{extracted_responses[0]}")
    return [2.0 if r == a else 0.0 for r, a in zip(extracted_responses, answer)]

def int_reward_func(completions, **kwargs) -> List[float]:
    """Reward function for integer answers."""
    responses = [completion[0]['content'] for completion in completions]
    extracted_responses = [extract_xml_answer(r) for r in responses]
    return [0.5 if r.isdigit() else 0.0 for r in extracted_responses]

def strict_format_reward_func(completions, **kwargs) -> List[float]:
    """Reward function that checks if the completion has a specific format."""
    pattern = r"^<reasoning>\n.*?\n</reasoning>\n<answer>\n.*?\n</answer>\n$"
    responses = [completion[0]["content"] for completion in completions]
    matches = [bool(re.match(pattern, r, re.DOTALL)) for r in responses]
    return [0.5 if match else 0.0 for match in matches]

def soft_format_reward_func(completions, **kwargs) -> List[float]:
    """Reward function that checks if the completion has a specific format (less strict)."""
    pattern = r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>"
    responses = [completion[0]["content"] for completion in completions]
    matches = [bool(re.search(pattern, r, re.DOTALL)) for r in responses]
    return [0.5 if match else 0.0 for match in matches]

def count_xml(text) -> float:
    """Count XML tags and calculate a reward."""
    count = 0.0
    if text.count("<reasoning>\n") == 1:
        count += 0.125
    if text.count("\n</reasoning>\n") == 1:
        count += 0.125
    if text.count("\n<answer>\n") == 1:
        count += 0.125
        count -= len(text.split("\n</answer>\n")[-1])*0.001
    if text.count("\n</answer>") == 1:
        count += 0.125
        count -= (len(text.split("\n</answer>")[-1]) - 1)*0.001
    return count

def xmlcount_reward_func(completions, **kwargs) -> List[float]:
    """Reward function based on XML tag counting."""
    contents = [completion[0]["content"] for completion in completions]
    return [count_xml(c) for c in contents]

def load_model(config: Config):
    """Load and prepare the model with LoRA."""
    logger.info(f"Loading model {config.model.model_name}")
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config.model.model_name,
        max_seq_length=config.model.max_seq_length,
        load_in_4bit=config.model.load_in_4bit,
        fast_inference=config.model.fast_inference,
        max_lora_rank=config.model.lora_rank,
        gpu_memory_utilization=config.model.gpu_memory_utilization,
    )
    
    model = FastLanguageModel.get_peft_model(
        model,
        r=config.model.lora_rank,
        target_modules=config.model.target_modules,
        lora_alpha=config.model.lora_rank,
        use_gradient_checkpointing=config.model.use_gradient_checkpointing,
        random_state=config.model.random_state,
    )
    
    return model, tokenizer

def setup_training(config: Config, model, tokenizer, dataset):
    """Set up the GRPO trainer."""
    training_args = GRPOConfig(
        use_vllm=config.training.use_vllm,
        learning_rate=config.training.learning_rate,
        adam_beta1=config.training.adam_beta1,
        adam_beta2=config.training.adam_beta2,
        weight_decay=config.training.weight_decay,
        warmup_ratio=config.training.warmup_ratio,
        lr_scheduler_type=config.training.lr_scheduler_type,
        optim=config.training.optim,
        logging_steps=config.training.logging_steps,
        bf16=is_bfloat16_supported(),
        fp16=not is_bfloat16_supported(),
        per_device_train_batch_size=config.training.per_device_train_batch_size,
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        num_generations=config.training.num_generations,
        max_prompt_length=config.training.max_prompt_length,
        max_completion_length=config.training.max_completion_length,
        num_train_epochs=config.training.num_train_epochs,
        max_steps=config.training.max_steps,
        save_steps=config.training.save_steps,
        max_grad_norm=config.training.max_grad_norm,
        report_to="wandb" if config.wandb.enabled else "none",
        output_dir=config.training.output_dir,
    )
    
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[
            xmlcount_reward_func,
            soft_format_reward_func,
            strict_format_reward_func,
            int_reward_func,
            correctness_reward_func,
        ],
        args=training_args,
        train_dataset=dataset,
    )
    
    return trainer

def run_inference(config: Config, model, tokenizer, lora_path=None, system_prompt=True):
    """Run inference with the model."""
    if system_prompt:
        messages = [
            {"role": "system", "content": config.system_prompt},
            {"role": "user", "content": "Which is bigger? 9.11 or 9.9?"},
        ]
    else:
        messages = [
            {"role": "user", "content": "Which is bigger? 9.11 or 9.9?"},
        ]
    
    text = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )
    
    sampling_params = SamplingParams(
        temperature=config.inference.temperature,
        top_p=config.inference.top_p,
        max_tokens=config.inference.max_tokens,
    )
    
    lora_request = None
    if lora_path:
        lora_request = model.load_lora(lora_path)
    
    output = model.fast_generate(
        [text],
        sampling_params=sampling_params,
        lora_request=lora_request,
    )[0].outputs[0].text
    
    return output

def main():
    """Main function."""
    args = parse_args()
    config = update_config_from_args(default_config, args)
    
    # Initialize Weights & Biases
    if config.wandb.enabled:
        wandb.init(
            project=config.wandb.project,
            name=config.wandb.name,
            tags=config.wandb.tags,
            config=vars(args)
        )
        logger.info(f"Initialized Weights & Biases: {wandb.run.name}")
    
    # Create output directory
    os.makedirs(config.training.output_dir, exist_ok=True)
    
    # Load model and tokenizer
    model, tokenizer = load_model(config)
    
    # Load dataset
    dataset = get_gsm8k_questions(config)
    logger.info(f"Loaded dataset with {len(dataset)} examples")
    
    # Set up trainer
    trainer = setup_training(config, model, tokenizer, dataset)
    
    # Train model
    logger.info("Starting training")
    trainer.train()
    
    # Save LoRA weights
    lora_path = os.path.join(config.training.output_dir, "grpo_saved_lora")
    model.save_lora(lora_path)
    logger.info(f"Saved LoRA weights to {lora_path}")
    
    # Run inference without LoRA
    logger.info("Running inference without LoRA")
    output_base = run_inference(config, model, tokenizer, lora_path=None, system_prompt=False)
    logger.info(f"Base model output:\n{output_base}")
    
    # Run inference with LoRA
    logger.info("Running inference with LoRA")
    output_lora = run_inference(config, model, tokenizer, lora_path=lora_path, system_prompt=True)
    logger.info(f"LoRA model output:\n{output_lora}")
    
    # Log to Weights & Biases
    if config.wandb.enabled:
        wandb.log({
            "base_model_output": wandb.Html(f"<pre>{output_base}</pre>"),
            "lora_model_output": wandb.Html(f"<pre>{output_lora}</pre>"),
        })
        wandb.finish()

if __name__ == "__main__":
    main() 