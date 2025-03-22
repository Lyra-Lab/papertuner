"""
Example of how to use PaperTuner for dataset processing.
"""
import os
import logging
from papertuner.data.processor import PaperProcessor

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def process_papers_example(max_papers=5):
    """
    Process papers from arXiv to create a dataset.
    
    Args:
        max_papers: Maximum number of papers to process
    """
    # Create a processor with API key from env var
    processor = PaperProcessor()
    
    # Check if API key is available
    if not processor.api_key:
        logging.warning("No API key found. Please set OPENAI_API_KEY or GEMINI_API_KEY environment variable.")
        logging.warning("Continuing without QA generation...")
    
    # Process papers (with a small number for this example)
    logging.info(f"Processing up to {max_papers} papers...")
    
    # Define a custom query for specific types of papers
    query = "\"fine-tuning language models\" OR \"instruction tuning\" OR \"RLHF\""
    
    # Process the papers
    new_papers = processor.process_papers(max_papers=max_papers, query=query)
    
    if new_papers:
        logging.info(f"Successfully processed {len(new_papers)} papers")
    else:
        logging.warning("No new papers were processed")

def create_dataset_example():
    """Create a dataset from processed papers."""
    processor = PaperProcessor()
    
    # Create dataset splits
    logging.info("Creating dataset from processed papers...")
    dataset_dict = processor.create_dataset_from_processed_papers(
        output_path="example_dataset",
        split_ratios=(0.8, 0.1, 0.1)
    )
    
    if dataset_dict:
        total_samples = sum(len(split) for split in dataset_dict.values())
        logging.info(f"Dataset created with {total_samples} total samples:")
        logging.info(f"  - Train: {len(dataset_dict['train'])} samples")
        logging.info(f"  - Validation: {len(dataset_dict['validation'])} samples")
        logging.info(f"  - Test: {len(dataset_dict['test'])} samples")
        
        # Show a sample from the training set
        if len(dataset_dict['train']) > 0:
            sample = dataset_dict['train'][0]
            logging.info("\nSample question:")
            print(sample['question'])
            logging.info("\nSample answer:")
            print(sample['answer'][:300] + "..." if len(sample['answer']) > 300 else sample['answer'])
    else:
        logging.error("Failed to create dataset. Make sure you have processed papers first.")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="PaperTuner dataset processing example")
    parser.add_argument("--mode", choices=["process", "dataset", "both"], default="both",
                       help="Mode to run: process papers, create dataset, or both (default: both)")
    parser.add_argument("--max-papers", type=int, default=5,
                       help="Maximum number of papers to process (default: 5)")
    
    args = parser.parse_args()
    
    if args.mode in ["process", "both"]:
        process_papers_example(max_papers=args.max_papers)
    
    if args.mode in ["dataset", "both"]:
        create_dataset_example() 