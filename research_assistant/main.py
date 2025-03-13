"""
Main entry point for the research assistant project.
"""

import argparse
import os
from pathlib import Path
import logging
import gc
import signal
import psutil
import threading
import time
import sys

from datasets import load_dataset
import torch
from transformers import set_seed
import wandb

from configs.config import MODELS_DIR, OUTPUT_DIR
from data.processor import prepare_research_dataset
from training.grpo_trainer import ResearchGRPOTrainer
from training.wandb_setup import init_wandb, log_model_metadata

def parse_args():
    parser = argparse.ArgumentParser(description="Research Assistant Training Script")
    parser.add_argument("--model", type=str, default="unsloth/Phi-4", help="Base model to use")
    parser.add_argument("--dataset", type=str, required=True, help="Path to the dataset")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory")
    parser.add_argument("--run_name", type=str, default=None, help="Name for the run")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--max_seq_length", type=int, default=512, help="Maximum sequence length")
    parser.add_argument("--lora_rank", type=int, default=16, help="LoRA rank for training")
    parser.add_argument("--num_epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=5e-6, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=1, help="Training batch size")
    parser.add_argument("--grad_accum", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--max_steps", type=int, default=None, help="Maximum training steps")
    parser.add_argument("--save_method", type=str, default="lora", 
                       choices=["lora", "merged_16bit", "merged_4bit", "gguf_8bit", "gguf_16bit", "gguf_q4_k_m"],
                       help="Method to save the model")
    parser.add_argument("--push_to_hub", action="store_true", help="Push model to Hugging Face Hub")
    parser.add_argument("--hub_path", type=str, default=None, help="Path on Hugging Face Hub")
    parser.add_argument("--hub_token", type=str, default=None, help="Hugging Face Hub token")
    return parser.parse_args()

# Memory monitoring thread
def monitor_memory(stop_event, interval=5):
    while not stop_event.is_set():
        process = psutil.Process(os.getpid())
        cpu_percent = process.cpu_percent()
        ram_percent = process.memory_percent()
        
        if torch.cuda.is_available():
            gpu_memory_allocated = torch.cuda.memory_allocated() / (1024**3)
            gpu_memory_reserved = torch.cuda.memory_reserved() / (1024**3)
            gpu_max_memory = torch.cuda.max_memory_allocated() / (1024**3)
            
            logging.info(f"Memory Monitor - CPU: {cpu_percent:.1f}%, RAM: {ram_percent:.1f}%, "
                        f"GPU Allocated: {gpu_memory_allocated:.2f}GB, "
                        f"GPU Reserved: {gpu_memory_reserved:.2f}GB, "
                        f"GPU Max: {gpu_max_memory:.2f}GB")
        else:
            logging.info(f"Memory Monitor - CPU: {cpu_percent:.1f}%, RAM: {ram_percent:.1f}%")
        
        time.sleep(interval)

def main():
    args = parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("training.log"),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    
    logger.info("Starting training process")
    logger.info(f"Arguments: {args}")
    
    # Set random seed
    set_seed(args.seed)
    logger.info(f"Random seed set to {args.seed}")
    
    # Initialize W&B
    logger.info("Initializing W&B")
    init_wandb(args.run_name)
    logger.info("W&B initialized successfully")
    
    # Load and prepare the dataset
    try:
        logger.info(f"Loading dataset from {args.dataset}")
        raw_dataset = load_dataset(args.dataset, "main")
        logger.info(f"Dataset loaded successfully with {len(raw_dataset['train'])} training examples")
    except Exception as e:
        logger.error(f"Error loading dataset: {str(e)}", exc_info=True)
        raise
    
    # Process dataset if needed
    if all(col in raw_dataset["train"].features for col in ["problem", "literature", "hypothesis", "methodology"]):
        dataset = prepare_research_dataset(raw_dataset["train"].to_pandas())
    else:
        dataset = raw_dataset
    
    # Initialize the GRPO trainer
    output_dir = args.output_dir or Path(OUTPUT_DIR) / (args.run_name or "default_run")
    try:
        logger.info("Initializing GRPO trainer")
        trainer = ResearchGRPOTrainer(
            model_name=args.model,
            max_seq_length=args.max_seq_length,
            lora_rank=args.lora_rank,
            output_dir=output_dir,
            gpu_memory_utilization=0.6,
        )
        logger.info("GRPO trainer initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing trainer: {str(e)}", exc_info=True)
        raise
    
    # Log model configuration
    log_model_metadata(
        model_name=args.model,
        model_config={
            "max_seq_length": args.max_seq_length,
            "lora_rank": args.lora_rank,
            "learning_rate": args.learning_rate,
            "batch_size": args.batch_size,
            "grad_accum": args.grad_accum,
        }
    )
    
    # Prepare dataset for GRPO training
    train_dataset = trainer.prepare_dataset(dataset["train"])
    eval_dataset = trainer.prepare_dataset(dataset["validation"]) if "validation" in dataset else None
    
    # Start memory monitoring in main()
    stop_monitor = threading.Event()
    monitor_thread = threading.Thread(target=monitor_memory, args=(stop_monitor,))
    monitor_thread.daemon = True
    monitor_thread.start()

    # Register signal handlers for graceful shutdown
    def signal_handler(sig, frame):
        logging.warning(f"Received signal {sig}, shutting down gracefully")
        stop_monitor.set()
        wandb.finish()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Train the model
    logger.info(f"Training model: {args.model}")
    model, results = trainer.train(
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        num_train_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        max_steps=args.max_steps,
    )
    
    # Save the model
    save_path = trainer.save_model(
        save_method=args.save_method,
        push_to_hub=args.push_to_hub,
        hub_path=args.hub_path,
        token=args.hub_token
    )
    logger.info(f"Model saved to {save_path}")
    
    # Test the model
    test_prompt = "What methodology should be used to study the impact of social media on mental health?"
    response = trainer.generate(
        prompt=test_prompt,
        lora_path=save_path if args.save_method == "lora" else None,
    )
    logger.info("\nTest Generation:")
    logger.info(f"Prompt: {test_prompt}")
    logger.info(f"Response: {response}")
    
    # Close W&B
    wandb.finish()

if __name__ == "__main__":
    main() 