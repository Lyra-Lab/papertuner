from papertuner.train import ResearchAssistantTrainer

# Example usage of the ResearchAssistantTrainer with fact-checking reward function
if __name__ == "__main__":
    trainer = ResearchAssistantTrainer(
        model_name="unsloth/Phi-4-mini-instruct",
        max_seq_length=1024,
        lora_rank=64,
        output_dir="outputs",
        batch_size=1,
        gradient_accumulation_steps=1,
        learning_rate=5e-6,
        max_steps=2000,
        save_steps=100
    )

    # Load dataset
    dataset = trainer.load_dataset("densud2/ml_qa_dataset")

    # Load model
    model, tokenizer, peft_model = trainer.load_model()

    # Initialize trainer
    trainer_instance = trainer.get_trainer(peft_model, tokenizer, dataset)

    # Run training
    trainer_instance.train()

    # Save trained LoRA weights
    lora_path = Path("outputs") / "final_lora"
    peft_model.save_lora(str(lora_path))

    # Example inference
    question = "What are the key considerations for designing a neural network architecture for image classification?"
    response = trainer.run_inference(model, tokenizer, question, lora_path=lora_path)
    print("Model Response:", response)
