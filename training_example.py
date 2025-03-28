from papertuner.train import ResearchAssistantTrainer
import os
from pathlib import Path

if __name__ == "__main__":
    trainer = ResearchAssistantTrainer(
        model_name="unsloth/DeepSeek-R1-Distill-Qwen-1.5B-GGUF",
        max_seq_length=1024,
        lora_rank=64,
        output_dir="outputs",
        batch_size=1,
        gradient_accumulation_steps=1,
        learning_rate=5e-6,
        max_steps=2000,
        save_steps=100
    )

    # Run training with the new train method
    training_results = trainer.train(
        dataset_name="densud2/ml_qa_dataset",
        push_to_hf=True,  # Set to True if you want to push to Hugging Face
        hf_username=os.getenv('HF_USERNAME'),
        hf_model_name="your_model_name",  # Replace with your model name
        hf_token="your_hf_token",  # Replace with your Hugging Face token
        bespoke_api_token="your_bespoke_api_token"  # Replace with your Bespoke API token
    )

    # Example inference
    question = "What are the key considerations for designing a neural network architecture for image classification?"
    response = trainer.run_inference(training_results["model"], training_results["tokenizer"], question, lora_path=training_results["lora_path"])
    print("Model Response:", response)
