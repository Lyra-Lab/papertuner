import os
from papertuner.dataset import ResearchPaperProcessor
# 1. Create a biology-focused dataset
processor = ResearchPaperProcessor(
    api_key=os.getenv("GEMINI_API_KEY"),
    hf_token=os.getenv("HF_TOKEN"),
    hf_repo_id="densud2/biological-medical-dataset"  # Replace with your desired repo name
)

# Define a biology-focused search query
bio_query = " OR ".join([
    "biology",
    "molecular biology",
    "cell biology",
    "genetics",
    "biochemistry",
    "systems biology",
    "synthetic biology",
    "bioinformatics",
])

# Process biology papers and create dataset
# This will download papers, extract text, and generate QA pairs
print("Processing biology research papers...")
papers = processor.process_papers(
    max_papers=400,
    search_query=bio_query,
    clear_processed_data=True  # Start fresh
)

# Validate and push dataset to HuggingFace
validation_results = processor.validate_dataset()
print(f"Dataset validation - Valid entries: {validation_results['valid_entries']}/{validation_results['total_files']}")

# Push the dataset to HuggingFace
processor.push_to_hf()
print(f"Dataset uploaded to HuggingFace: {processor.hf_repo_id}")

from papertuner.train import ResearchAssistantTrainer
# 2. Train a specialized biology research assistant model using PEF with LoRA adapters
trainer = ResearchAssistantTrainer(
    model_name="unsloth/Phi-4-mini-instruct-GGUF",  # Updated to use the Phi-4 model
    max_seq_length=2048,  # Increased sequence length for better context
    lora_rank=64,
    output_dir="./bio_model",
    batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    max_steps=2000,
    system_prompt="""You are a biology research assistant. Follow this format:
<think>
Analyze the biological research question step-by-step, considering:
- Relevant biological mechanisms
- Experimental approaches
- Key methodological considerations
- Potential limitations
- Ethical considerations
</think>

Provide a clear, scientifically-grounded answer that explains both the 'how' and 'why'
of the biological approach or method."""
)

# Train the model using the new PEF approach
print("Starting model training with LoRA adapters...")
model, tokenizer, output_path = trainer.train("densud2/bio-research-qa")  # Use the dataset we created
print(f"Model trained and saved to {output_path}")

# 3. Test the model with biology research questions
test_questions = [
    "How would you design a CRISPR experiment to study gene function in mammalian cells?",
    "What approaches can be used to study protein-protein interactions in vivo?",
    "How would you analyze single-cell RNA sequencing data to identify cell types?"
]

print("\nTesting the trained model with biology questions:")
for question in test_questions:
    response = trainer.run_inference(
        model,
        tokenizer,
        question,
        output_path
    )
    print(f"\nQ: {question}")
    print(f"A: {response}\n")
    print("-" * 80)

# Run a comparison demo with sample questions from the dataset
print("\nRunning comparison demo with examples from the dataset:")
trainer.demo_comparison(model, tokenizer, output_path, "densud2/bio-research-qa")

# Optional: Push the trained model to HuggingFace Hub
repo_id = "densud2/bio-research-assistant"  # Replace with your desired model repo name
trainer.push_to_hf(
    model,
    tokenizer,
    repo_id,
    os.getenv("HF_TOKEN")
)
print(f"Model pushed to HuggingFace Hub: {repo_id}")
