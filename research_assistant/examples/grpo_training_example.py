"""
Example script for training a research assistant model using GRPO.
"""

import os
from datasets import load_dataset
from research_assistant.training.grpo_trainer import ResearchGRPOTrainer
from research_assistant.configs.config import MODELS_DIR, OUTPUT_DIR

def main():
    # Initialize the GRPO trainer
    trainer = ResearchGRPOTrainer(
        model_name="unsloth/Phi-4",  # You can use other models like "meta-llama/Llama-2-7b-hf"
        max_seq_length=512,
        lora_rank=16,
        output_dir=os.path.join(OUTPUT_DIR, "grpo_research_assistant"),
    )
    
    # Load a dataset (example with GSM8K for demonstration)
    # In a real scenario, you would use your research methodology dataset
    dataset = load_dataset('openai/gsm8k', 'main')
    
    # Prepare the dataset for training
    train_dataset = trainer.prepare_dataset(dataset["train"])
    
    # Train the model
    model, results = trainer.train(
        train_dataset=train_dataset,
        num_train_epochs=1,
        learning_rate=5e-6,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        max_steps=100,  # For quick testing, remove for full training
    )
    
    # Save the model
    save_path = trainer.save_model()
    print(f"Model saved to {save_path}")
    
    # Test the model
    test_prompt = "How do sleep patterns affect academic performance in college students?"
    response = trainer.generate(
        prompt=test_prompt,
        lora_path=save_path,
    )
    
    print(f"Test prompt: {test_prompt}")
    print(f"Generated response: {response}")

if __name__ == "__main__":
    main() 