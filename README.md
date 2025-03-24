# PaperTuner

PaperTuner is a Python package for creating research assistant models by processing academic papers and fine-tuning language models to provide methodology guidance and research approaches.

## Features

- Automated extraction of research papers from arXiv
- Section extraction to identify problem statements, methodologies, and results
- Generation of high-quality question-answer pairs for research methodology
- Fine-tuning of language models with GRPO (Growing Rank Pruned Optimization)
- Integration with Hugging Face for dataset and model sharing

## Installation

```bash
pip install papertuner
```

## Basic Usage

### As a Command-Line Tool

#### 1. Create a dataset from research papers

```bash
# Set up your environment variables
export GEMINI_API_KEY="your-api-key"
export HF_TOKEN="your-huggingface-token"  # Optional, for uploading to HF

# Run the dataset creation
papertuner-dataset --max-papers 100
```

#### 2. Train a model

```bash
# Train using the created or an existing dataset
papertuner-train --model "Qwen/Qwen2.5-3B-Instruct" --dataset "densud2/ml_qa_dataset"
```

### As a Python Library

Here's a complete example of creating a specialized biology research model:

```python
from papertuner import ResearchPaperProcessor, ResearchAssistantTrainer

# 1. Create a dataset from biology papers
processor = ResearchPaperProcessor(
    api_key="your-gemini-api-key",
    hf_repo_id="your-username/bio-research-qa"
)

# Use a biology-focused search query
bio_query = " OR ".join([
    "molecular biology",
    "cell biology",
    "genetics",
    "biochemistry",
    "systems biology",
    "synthetic biology",
    "bioinformatics",
    "genomics",
    "proteomics",
    "metabolomics"
])

# Process papers and create dataset
papers = processor.process_papers(
    max_papers=100,
    search_query=bio_query,
    clear_processed_data=True  # Start fresh
)

# 2. Train a specialized model
trainer = ResearchAssistantTrainer(
    model_name="Qwen/Qwen2.5-3B-Instruct",  # Base model
    lora_rank=64,
    output_dir="./bio_model",
    system_prompt="""You are a biology research assistant. Follow this format:
<think>
Analyze the biological research question step-by-step, considering:
- Relevant biological mechanisms
- Experimental approaches
- Key methodological considerations
- Potential limitations
</think>

Provide a clear, scientifically-grounded answer that explains both the 'how' and 'why'
of the biological approach or method."""
)

# Train the model
results = trainer.train("your-username/bio-research-qa")

# 3. Test the model with biology questions
questions = [
    "How would you design a CRISPR experiment to study gene function in mammalian cells?",
    "What approaches can be used to study protein-protein interactions in vivo?",
    "How would you analyze single-cell RNA sequencing data to identify cell types?"
]

for question in questions:
    response = trainer.run_inference(
        results["model"],
        results["tokenizer"],
        question,
        results["lora_path"]
    )
    print(f"\nQ: {question}")
    print(f"A: {response}\n")
```

## Configuration

You can configure the tool using environment variables or when initializing the classes:

- `GEMINI_API_KEY`: API key for generating QA pairs
- `HF_TOKEN`: Hugging Face token for uploading datasets and models
- `HF_REPO_ID`: Hugging Face repository ID for the dataset
- `PAPERTUNER_DATA_DIR`: Custom directory for storing data (default: ~/.papertuner/data)

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
