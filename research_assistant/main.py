"""
Main entry point for the research assistant project.
"""

import argparse
import os
from pathlib import Path

from datasets import load_dataset
import torch
from transformers import set_seed

from configs.config import MODELS_DIR, OUTPUT_DIR
from data.processor import prepare_research_dataset
from training.trainer import ResearchAssistantTrainer

def parse_args():
    parser = argparse.ArgumentParser(description="Research Assistant Training Script")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-2-7b-hf", help="Base model to use")
    parser.add_argument("--dataset", type=str, required=True, help="Path to the dataset")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory")
    parser.add_argument("--run_name", type=str, default=None, help="Name for the run")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Load and prepare the dataset
    print(f"Loading dataset from {args.dataset}")
    dataset = load_dataset(args.dataset)
    
    # If the dataset is not already in the required format
    # dataset = prepare_research_dataset(dataset)
    
    # Initialize the trainer
    output_dir = args.output_dir or Path(OUTPUT_DIR) / (args.run_name or "default_run")
    trainer = ResearchAssistantTrainer(
        model_name=args.model,
        output_dir=output_dir,
        run_name=args.run_name,
    )
    
    # Train the model
    print(f"Training model: {args.model}")
    model, results = trainer.train(
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
    )
    
    # Close the W&B connection
    trainer.close()
    
    print(f"Training complete. Model saved to {output_dir}/final_model")

if __name__ == "__main__":
    main() 