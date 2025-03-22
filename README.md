# PaperTuner

A Python package for fine-tuning language models on research papers using GRPO (Guided Reward Preference Optimization).

## Features

- Research paper dataset creation from arXiv
- QA pair generation from research papers
- Fine-tuning language models using GRPO
- Inference with trained models

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/papertuner.git
cd papertuner

# Install the package
pip install -e .
```

## Usage

### Command Line Interface

PaperTuner provides command-line tools for dataset processing and model training:

#### Dataset Processing

```bash
# Process papers from arXiv (requires API key in environment variable)
papertuner-dataset process --max-papers 100

# Create dataset from processed papers
papertuner-dataset create-dataset --output-path my_dataset

# Upload dataset to HuggingFace Hub
papertuner-dataset upload --hf-repo-id yourusername/my-dataset
```

#### Model Training

```bash
# Train a model
papertuner-train train --model-name unsloth/DeepSeek-R1-Distill-Qwen-1.5B-unsloth-bnb-4bit --output-dir my_model

# Run inference with a trained model
papertuner-train inference --lora-path my_model/grpo_saved_lora --query "How do transformer models work?"

# Export a trained model (e.g., to GGUF format for llama.cpp)
papertuner-train export --lora-path my_model/grpo_saved_lora --format gguf
```

### Python API

You can also use PaperTuner as a Python library:

```python
from papertuner.data.processor import PaperProcessor
from papertuner.train.trainer import MLAssistantTrainer

# Process papers and create a dataset
processor = PaperProcessor()
processor.process_papers(max_papers=100)
processor.create_dataset_from_processed_papers(output_path="my_dataset")

# Train a model
trainer = MLAssistantTrainer(
    model_name="unsloth/DeepSeek-R1-Distill-Qwen-1.5B-unsloth-bnb-4bit",
    output_dir="my_model"
)
trainer.load_model()
trainer.load_dataset()
trainer.setup_reward_functions()
trainer.train()

# Run inference
response = trainer.run_inference(
    query="How should I approach implementing a transformer model from scratch?",
    lora_path="my_model/grpo_saved_lora"
)
print(response)
```

## Examples

Check out the `examples` directory for complete examples of how to use PaperTuner:

- `dataset_processing.py`: Example of processing papers and creating a dataset
- `simple_training.py`: Example of training a model and running inference

## Environment Variables

PaperTuner uses the following environment variables:

- `OPENAI_API_KEY` or `GEMINI_API_KEY`: API key for generating QA pairs from papers
- `API_BASE_URL`: Base URL for the API (if using a custom API endpoint)
- `HF_TOKEN`: HuggingFace token for uploading datasets and models
- `HF_REPO_ID`: Default HuggingFace repository ID for uploads
- `PAPERTUNER_DATA_DIR`: Custom data directory (default: `data`)

## Requirements

- Python 3.8+
- PyTorch
- Unsloth
- Hugging Face libraries (transformers, datasets)
- trl
- vllm
- arxiv
- PyMuPDF

## License

Apache 2.0
