# PaperTuner

PaperTuner is a Python package for creating research assistant models by processing academic papers and fine-tuning language models to provide methodology guidance and research approaches.
(quick start is available in [colab](https://colab.research.google.com/drive/1I96xJAveEYoW-xjS4WcAeSbxuB7I169q?usp=sharing) now)

## Features

- Automated extraction of research papers from multiple sources (arXiv, PubMed, Semantic Scholar, IEEE, ACM)
- Section extraction to identify problem statements, methodologies, and results
- Generation of high-quality question-answer pairs for research methodology
- Fine-tuning of language models with GRPO (Growing Rank Pruned Optimization)
- Multiple evaluation metrics (semantic similarity, ROUGE, BERTScore)
- Incremental model fine-tuning and checkpointing
- Integration with Hugging Face for dataset and model sharing
- Interactive demo creation with Gradio
- Support for multiple LLM providers (OpenAI, Anthropic, Gemini)
- Domain-specific configurations for different research fields
- Beautiful command-line interface with rich progress indicators

## Installation

```bash
pip install papertuner
```

## Quick Start

```bash
# Set up PaperTuner with your API key
papertuner setup --api-key "your-api-key" --api-type "openai" --hf-token "your-huggingface-token"

# Create a dataset from papers on transformers
papertuner dataset create "transformer architecture deep learning" --max-papers 20 --domain "machine_learning"

# Train a model using the created dataset
papertuner train model "your-username/dataset-name" --model-name "Qwen/Qwen2.5-3B-Instruct" --lora-rank 64

# Create a demo to interact with your model
papertuner demo "path/to/your/model" --share
```

## Basic Usage

### Command-Line Interface

PaperTuner provides an easy-to-use CLI with rich status indicators and helpful information.

#### Creating a Dataset

```bash
# Set up your environment variables
export OPENAI_API_KEY="your-api-key"
export HF_TOKEN="your-huggingface-token"

# Create a dataset from research papers on a specific topic
papertuner dataset create "graph neural networks applications" \
  --max-papers 30 \
  --output-name "gnn_dataset" \
  --push-to-hub \
  --hub-repo-id "your-username/gnn-dataset" \
  --domain "machine_learning" \
  --qa-pairs-per-paper 5
```

#### Training a Model

```bash
# Train using the created dataset
papertuner train model "your-username/gnn-dataset" \
  --model-name "Qwen/Qwen2.5-3B-Instruct" \
  --lora-rank 64 \
  --output-dir "./my_gnn_assistant" \
  --push-to-hub \
  --hub-model-id "your-username/gnn-assistant" \
  --create-demo
```

#### Creating a Demo

```bash
# Create an interactive demo for your trained model
papertuner demo "./my_gnn_assistant/final" --share
```

### As a Python Library

```python
from papertuner import ResearchPaperProcessor, ResearchAssistantTrainer, Config, LLMClient, DatasetManager
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)

# 1. Configure PaperTuner with your preferred settings
config = Config()
config.set("llm_config", {
    "provider": "openai",  # Options: "openai", "anthropic", "gemini"
    "model": "gpt-4o-mini",
    "temperature": 0.3,
    "fallback_provider": "anthropic",  # Fallback provider if primary fails
    "fallback_model": "claude-3-haiku-20240307"
})

# Configure domain-specific settings for your research field
config.set("domain_configs", {
    "computer_vision": {
        "sections_of_interest": ["methodology", "architecture", "experiments", "ablation studies"],
        "domain_keywords": ["CNN", "object detection", "segmentation", "feature extraction", "attention"],
        "citation_importance": 0.6
    }
})

# Save the configuration for future use
config.save("cv_config.json")

# 2. Create a dataset from research papers
processor = ResearchPaperProcessor(
    api_key=os.environ.get("OPENAI_API_KEY"),
    hf_token=os.environ.get("HF_TOKEN"),
    hf_repo_id="your-username/cv-papers-dataset",
    domain_config=config.get("domain_configs").get("computer_vision")
)

try:
    # Process papers on a specific research topic
    results = processor.process_papers(
        max_papers=20,
        search_query="vision transformers object detection",
        num_qa_pairs=5
    )
    
    # Generate statistics about your dataset
    stats = processor.generate_statistics()
    print(f"Processed {stats['total_papers']} papers with {stats['total_qa_pairs']} QA pairs")
    
    # Push the dataset to Hugging Face Hub
    processor.push_to_hf(split_ratios=(0.8, 0.1, 0.1))  # train/validation/test splits
    print(f"Dataset pushed to: {processor.hf_repo_id}")
    
except Exception as e:
    logging.error(f"Error creating dataset: {e}")

# 3. Manage and analyze datasets with DatasetManager
dataset_manager = DatasetManager()

# Load and analyze an existing dataset
dataset_stats = dataset_manager.analyze_dataset("your-username/cv-papers-dataset")
print(f"Average question length: {dataset_stats['avg_question_length']}")
print(f"Average answer length: {dataset_stats['avg_answer_length']}")

# Merge multiple datasets
merged_dataset = dataset_manager.merge_datasets([
    "your-username/cv-papers-dataset",
    "your-username/additional-cv-dataset"
], output_path="./merged_dataset.json")

# Filter dataset by category
filtered_dataset = dataset_manager.filter_dataset(
    "your-username/cv-papers-dataset",
    output_path="./filtered_dataset.json",
    category="Architecture & Design"
)

# 4. Train a custom research assistant model
trainer = ResearchAssistantTrainer(
    model_name="Qwen/Qwen2.5-3B-Instruct",
    lora_rank=64,
    output_dir="./my_cv_assistant",
    max_steps=1000,
    num_generations=4,
    batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=5e-6,
    mixed_precision="bf16"  # Use "fp16" if bf16 isn't available on your GPU
)

# Train on your dataset
training_result = trainer.train("your-username/cv-papers-dataset")

# Evaluate the model on test data
evaluation = trainer.evaluate_model(
    model=training_result["model"],
    tokenizer=training_result["tokenizer"],
    dataset=dataset_manager.load_dataset("your-username/cv-papers-dataset", split="test"),
    lora_path=training_result["lora_path"]
)

# Print evaluation metrics
print(f"Semantic similarity: {evaluation['semantic_similarity']['mean']:.4f}")
print(f"ROUGE-L score: {evaluation['rouge']['rougeL']:.4f}")

# 5. Run inference with your trained model
questions = [
    "How would you implement a vision transformer for small object detection?",
    "What are the key architectural differences between YOLO and Faster R-CNN?",
    "Explain the role of the attention mechanism in vision transformers for dense prediction tasks."
]

# Create an LLM client for comparison with existing models
llm_client = LLMClient(config.get("llm_config"))

for question in questions:
    # Get response from your fine-tuned model
    ft_response = trainer.run_inference(
        training_result["model"],
        training_result["tokenizer"],
        question,
        training_result["lora_path"]
    )
    
    # Compare with response from base LLM
    base_response = llm_client.generate(question)
    
    print(f"\nQuestion: {question}")
    print(f"Fine-tuned model: {ft_response[:300]}...")
    print(f"Base LLM: {base_response[:300]}...")

# 6. Create and share an interactive demo
trainer.create_gradio_demo(
    model=training_result["model"],
    tokenizer=training_result["tokenizer"],
    lora_path=training_result["lora_path"],
    share=True
)

# 7. Push your model to Hugging Face Hub
model_url = trainer.push_to_hf(
    model=training_result["model"],
    tokenizer=training_result["tokenizer"],
    repo_id="your-username/cv-research-assistant",
    lora_path=training_result["lora_path"]
)
print(f"Model published at: {model_url}")

## Advanced Features

### Domain-Specific Configuration

PaperTuner supports domain-specific configurations for different research fields:

```python
# Create a custom configuration file
config = {
    "domain_configs": {
        "reinforcement_learning": {
            "sections_of_interest": ["method", "algorithm", "experiments", "results", "discussion"],
            "domain_keywords": ["policy", "reward", "agent", "environment", "q-learning"],
            "citation_importance": 0.5,
        }
    }
}

# Save the configuration
with open("rl_config.json", "w") as f:
    import json
    json.dump(config, f)

# Use the configuration when creating a dataset
papertuner dataset create "reinforcement learning robotics" --config-file "rl_config.json" --domain "reinforcement_learning"
```

### Incremental Fine-Tuning

```bash
# Train a model for a few steps
papertuner train model "your-dataset" --max-steps 200 --output-dir "./checkpoint1"

# Continue training from the checkpoint
papertuner train model "your-dataset" --checkpoint "./checkpoint1/final" --max-steps 500 --output-dir "./final_model"
```

### Model Comparison

```python
from papertuner import ResearchAssistantTrainer

trainer = ResearchAssistantTrainer()
comparison = trainer.demo_comparison(
    model=model,
    tokenizer=tokenizer,
    dataset_name="your-username/dataset-name",
    lora_path="path/to/adapter",
    num_examples=5
)

# Print the comparison
for example in comparison["examples"]:
    print(f"Question: {example['question']}")
    print(f"Reference Answer: {example['reference'][:100]}...")
    print(f"Base Model: {example['base_model_response'][:100]}...")
    print(f"Fine-tuned Model: {example['fine_tuned_response'][:100]}...")
    print("")
```

## Advanced Use Cases

#### Working with Specific Papers

```python
from papertuner import ResearchPaperProcessor, Config
from pathlib import Path

# Configure for processing specific papers
config = Config()
config.set("sources", {
    "local": {
        "enabled": True,
        "paths": [
            "./papers/paper1.pdf", 
            "./papers/paper2.pdf"
        ],
    },
    "arxiv": {"enabled": False},  # Disable arxiv search
})

processor = ResearchPaperProcessor(domain_config=config.get("domain_configs").get("machine_learning"))

# Process local papers
results = processor.process_papers(max_papers=10)
```

#### Incremental Training with Checkpoints

```python
from papertuner import ResearchAssistantTrainer

# Initial training phase - train for 200 steps
trainer = ResearchAssistantTrainer(
    model_name="Qwen/Qwen2.5-3B-Instruct",
    output_dir="./checkpoint_training",
    max_steps=200
)
result1 = trainer.train("your-username/dataset")

# Continue training from checkpoint - train for 300 more steps
trainer = ResearchAssistantTrainer(
    model_name="Qwen/Qwen2.5-3B-Instruct",
    output_dir="./final_model",
    max_steps=300
)
result2 = trainer.train(
    "your-username/dataset", 
    checkpoint_path="./checkpoint_training/final"
)
```

#### Multi-Provider LLM Client Usage

```python
from papertuner import LLMClient
import os

# Set up API keys for multiple providers
os.environ["OPENAI_API_KEY"] = "your-openai-key"
os.environ["ANTHROPIC_API_KEY"] = "your-anthropic-key"
os.environ["GEMINI_API_KEY"] = "your-gemini-key"

# Create client that can use any available provider
client = LLMClient({
    "provider": "anthropic",
    "model": "claude-3-opus-20240229",
    "fallback_provider": "openai",
    "fallback_model": "gpt-4-turbo",
    "temperature": 0.2,
    "max_tokens": 2000
})

# Use different providers for different tasks
system_design_response = client.generate(
    "Design a system for real-time object detection on edge devices",
    provider="anthropic"  # Use Anthropic for this query
)

implementation_response = client.generate(
    "Write pseudocode for non-maximum suppression in object detection",
    provider="openai"  # Use OpenAI for this query
)

# Automatic fallback if provider is unavailable
response = client.generate(
    "Explain the differences between YOLO and SSD architectures"
    # Uses default provider (anthropic) with fallback to openai if unavailable
)
```

## Configuration

You can configure the tool using environment variables, a configuration file, or when initializing the classes:

- `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GEMINI_API_KEY`: API keys for generating QA pairs
- `HF_TOKEN`: Hugging Face token for uploading datasets and models
- `HF_REPO_ID`: Hugging Face repository ID for the dataset
- `PAPERTUNER_DATA_DIR`: Custom directory for storing data (default: ~/.papertuner/data)

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Roadmap

- [x] Support for multiple data sources
- [x] Domain-specific configurations
- [x] Multi-provider LLM support
- [x] Model evaluation metrics
- [x] Interactive demos
- [ ] Web interface for dataset visualization
- [ ] Integration with vector databases for retrieval
- [ ] Support for multi-modal models
- [ ] Cloud deployment options
