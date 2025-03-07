"""
Example script to generate a dataset on machine learning papers using PaperTuner.

This script:
1. Searches for machine learning papers on ArXiv
2. Extracts text using Ollama's vision-capable model
3. Creates a HuggingFace dataset
"""

import os
import logging
from papertuner import DatasetPipeline

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    # Initialize the pipeline
    pipeline = DatasetPipeline(
        ocr_type="ollama",
        source_type="arxiv",
        formatter_type="huggingface",
        ocr_kwargs={
            "model_name": "granite3.2-vision:latest",  # Using llava for vision capabilities
            # Change host if Ollama is running elsewhere
            "ollama_host": "http://localhost:11434",
            "temperature": 0.1,  # Low temperature for more deterministic outputs
            "max_tokens": 16384,  # Allow for longer outputs
        },
        source_kwargs={
            "max_results": 100,  # Maximum number of papers to search
        },
        formatter_kwargs={
            "save_locally": True,
            "push_to_hub": True,
            "hub_dataset_name": "densud2/machine-learning-papers",
            "hub_token": os.environ.get("HF_TOKEN"),
        }
    )

    # Generate the dataset
    # The search query uses ArXiv syntax
    query = "cat:cs.LG AND cat:cs.AI AND (ti:\"machine learning\" OR ti:\"deep learning\" OR ti:\"neural network\") AND submittedDate:[20230101 TO 20231231]"
    output_path = "machine_learning_dataset"

    # Process 10 papers (adjust as needed)
    max_papers = 10

    entries = pipeline.generate(
        query=query,
        output_path=output_path,
        max_papers=max_papers,
    )

    # Print summary
    logger.info(f"Successfully processed {len(entries)} out of {max_papers} papers")
    logger.info(f"Dataset saved to: {output_path}")

    # Print sample titles
    if entries:
        logger.info("\nSample papers processed:")
        for i, entry in enumerate(entries[:5]):
            logger.info(f"{i+1}. {entry['title']}")

if __name__ == "__main__":
    main()
