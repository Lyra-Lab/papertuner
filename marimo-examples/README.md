# PaperTuner Marimo Examples

This directory contains interactive [marimo](https://marimo.io) notebooks that demonstrate how to use the PaperTuner package for processing research papers and training language models.

## Getting Started

1. Install marimo and PaperTuner:
   ```bash
   pip install marimo[recommended]
   pip install -e ..  # Install PaperTuner from parent directory
   ```

2. Run the examples:
   ```bash
   marimo edit dataset_explorer.py
   ```

## Available Examples

- **dataset_explorer.py**: Explore and visualize the research paper datasets
- **paper_processor.py**: Process papers from arXiv and generate QA pairs
- **model_trainer.py**: Train and fine-tune language models on research papers
- **inference_dashboard.py**: Run inference with trained models and visualize results

## Note on Environment Variables

Some examples require API keys to function properly. Make sure to set the following environment variables:

```bash
export OPENAI_API_KEY=your_openai_key_here
# or
export GEMINI_API_KEY=your_gemini_key_here

# For uploading to Hugging Face
export HF_TOKEN=your_huggingface_token_here
``` 