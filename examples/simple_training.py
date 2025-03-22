"""
Example of how to use PaperTuner for training a model.
"""
import os
import logging
from papertuner.train.trainer import MLAssistantTrainer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def run_training_example():
    """Run a simple training example using the default ML QA dataset."""
    
    # Initialize the trainer with custom settings
    trainer = MLAssistantTrainer(
        model_name="unsloth/DeepSeek-R1-Distill-Qwen-1.5B-unsloth-bnb-4bit",
        output_dir="example_outputs",
        max_seq_length=8192,
        lora_rank=64,
        max_steps=100,  # Using a smaller value for this example
        save_steps=50,
        gpu_memory_utilization=0.5
    )
    
    # Load the model
    logging.info("Loading model...")
    trainer.load_model()
    
    # Load the dataset
    logging.info("Loading dataset...")
    trainer.load_dataset()
    
    # Setup reward functions
    logging.info("Setting up reward functions...")
    trainer.setup_reward_functions()
    
    # Train the model
    logging.info("Starting training...")
    trainer.train(report_to="none")  # Set to "wandb" to use Weights & Biases for logging
    
    # Save the trained model
    logging.info("Training complete. Model saved to example_outputs/grpo_saved_lora")

def run_inference_example():
    """Run a simple inference example using a trained model."""
    
    # Check if a trained model exists
    if not os.path.exists("example_outputs/grpo_saved_lora"):
        logging.error("No trained model found. Please run training first.")
        return
    
    # Initialize the trainer with the same model
    trainer = MLAssistantTrainer(
        model_name="unsloth/DeepSeek-R1-Distill-Qwen-1.5B-unsloth-bnb-4bit"
    )
    
    # Load the model
    logging.info("Loading model for inference...")
    trainer.load_model()
    
    # Run inference with the trained model
    query = "How should I approach implementing a transformer model from scratch?"
    
    logging.info(f"Running inference with query: {query}")
    output = trainer.run_inference(
        query=query,
        lora_path="example_outputs/grpo_saved_lora",
        temperature=0.7,
        top_p=0.9,
        max_tokens=1000
    )
    
    print("\n" + "="*40 + " QUERY " + "="*40)
    print(query)
    print("\n" + "="*40 + " OUTPUT " + "="*41)
    print(output)
    print("\n" + "="*88)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="PaperTuner simple example")
    parser.add_argument("--mode", choices=["train", "inference", "both"], default="both",
                        help="Mode to run: train, inference, or both (default: both)")
    
    args = parser.parse_args()
    
    if args.mode in ["train", "both"]:
        run_training_example()
    
    if args.mode in ["inference", "both"]:
        run_inference_example() 