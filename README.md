# PaperTuner

A streamlined tool for creating fine-tuning datasets from scientific papers, focusing on extracting text with Ollama and organizing data in HuggingFace format.

## Overview

PaperTuner helps researchers and ML engineers create high-quality training datasets from scientific literature. It:

1. Searches for relevant papers on ArXiv using customizable queries
2. Extracts full text from PDFs using Ollama's vision-capable models
3. Formats the data as HuggingFace datasets for direct use in fine-tuning pipelines

Perfect for creating domain-specific datasets to fine-tune LLMs on scientific literature, especially for specialized fields where quality data is scarce.

## Features

- **ArXiv Integration**: Search papers by keyword, category, date range, and more
- **Ollama OCR**: Extract text from PDFs using locally-running vision models
- **HuggingFace Format**: Output datasets ready for use with the HuggingFace ecosystem
- **Simple Pipeline**: All steps integrated in one cohesive workflow

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/papertuner.git
cd papertuner

# Install the package
pip install -e .
```

## Requirements

- Python 3.8+
- [Ollama](https://ollama.ai/) running locally or on a remote server
- A vision-capable model in Ollama (e.g., llava:latest)
- pdf2image dependencies (poppler)
  - Ubuntu/Debian: `apt-get install poppler-utils`
  - macOS: `brew install poppler`
  - Windows: [poppler for Windows](https://github.com/oschwartz10612/poppler-windows/releases/)

## Quick Start

```python
from papertuner import DatasetPipeline

# Initialize the pipeline
pipeline = DatasetPipeline(
    ocr_type="ollama",
    source_type="arxiv",
    formatter_type="huggingface",
    ocr_kwargs={"model_name": "llava:latest"}
)

# Generate a dataset on quantum computing papers
entries = pipeline.generate(
    query="cat:quant-ph AND ti:\"quantum computing\"",
    output_path="quantum_dataset",
    max_papers=5
)
```

## Example Script

Here's a complete example script that creates a dataset of machine learning papers:

```python
#!/usr/bin/env python3
import logging
from papertuner import DatasetPipeline

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize the pipeline
pipeline = DatasetPipeline(
    ocr_type="ollama",
    source_type="arxiv",
    formatter_type="huggingface",
    ocr_kwargs={
        "model_name": "llava:latest",
        "ollama_host": "http://localhost:11434",
        "temperature": 0.1,
    }
)

# Generate the dataset
query = "cat:cs.LG AND cat:cs.AI AND ti:\"machine learning\""
output_path = "machine_learning_dataset"
max_papers = 10

entries = pipeline.generate(
    query=query,
    output_path=output_path,
    max_papers=max_papers,
)

logger.info(f"Successfully processed {len(entries)} papers")
```

## Using the Dataset

After generating a dataset, you can load it with HuggingFace's `datasets` library:

```python
from datasets import load_from_disk

# Load your locally saved dataset
dataset = load_from_disk("machine_learning_dataset")

# Explore the dataset
print(dataset)
print(dataset.column_names)
print(dataset[0]["title"])
print(dataset[0]["summary"])
```

## Configuration Options

### OCR Options

```python
ocr_kwargs={
    "model_name": "llava:latest",  # Or any Ollama vision model
    "ollama_host": "http://localhost:11434",  # Change if needed
    "temperature": 0.1,  # Lower for more deterministic results
    "max_tokens": 8192,  # Adjust based on paper length
    "max_retries": 3,  # Number of retry attempts
}
```

### Source Options

```python
source_kwargs={
    "max_results": 100,  # Maximum papers to search
    "sort_by": "relevance",  # Sort order for ArXiv results
}
```

### Formatter Options

```python
formatter_kwargs={
    "save_locally": True,
    "push_to_hub": False,  # Set to True to push to HF Hub
    "hub_dataset_name": "username/dataset-name",  # If pushing to Hub
    "hub_token": "your_token",  # Or set HF_TOKEN env variable
}
```

## ArXiv Search Query Tips

- **Categories**: Use `cat:cs.AI` for AI papers, `cat:physics.comp-ph` for computational physics, etc.
- **Title/Abstract**: Use `ti:keyword` for title search, `abs:keyword` for abstract search
- **Date Range**: Use `submittedDate:[20220101 TO 20221231]` for papers from 2022
- **Authors**: Use `au:lastname` to filter by author
- **Combining**: Use `AND`, `OR`, and `NOT` for complex queries

## Limitations

- PDF extraction quality depends on the capabilities of the vision model used
- Some papers with complex formatting or equations may not extract perfectly
- Processing time depends on your local hardware and the complexity of papers

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

MIT

## Acknowledgments

- Built on ArXiv API for paper access
- Uses Ollama for local LLM inference
- Integrates with HuggingFace datasets library

---

*Note: This tool respects ArXiv's usage policies. Please ensure you comply with ArXiv's terms of service when using this tool, and always cite papers appropriately in your research.*
