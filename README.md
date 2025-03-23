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

```python
from papertuner import ResearchPaperProcessor, ResearchAssistantTrainer

# Create a dataset
processor = ResearchPaperProcessor(
    api_key="your-api-key",
    hf_repo_id="your-username/dataset-name"
)
papers = processor.process_papers(max_papers=10)

# Train a model
trainer = ResearchAssistantTrainer(
    model_name="Qwen/Qwen2.5-3B-Instruct",
    lora_rank=64,
    output_dir="./model_output"
)
results = trainer.train("your-username/dataset-name")

# Test the model
question = "How would you design a transformer model for time series forecasting?"
response = trainer.run_inference(
    results["model"],
    results["tokenizer"],
    question,
    results["lora_path"]
)
print(response)
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
