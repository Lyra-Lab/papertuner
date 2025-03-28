# example_dataset_usage.py
from papertuner.dataset import ResearchPaperProcessor
from papertuner.config import HF_REPO_ID  # Import HF_REPO_ID if needed for HF interaction

def main():
    """Example script to use ResearchPaperProcessor."""

    # 1. Initialize ResearchPaperProcessor
    # You might need to set GEMINI_API_KEY, HF_TOKEN, HF_REPO_ID as environment variables
    # or pass them directly during initialization if not using env vars.
    # Example assuming API keys and repo ID are in environment variables:
    processor = ResearchPaperProcessor(hf_repo_id=HF_REPO_ID)

    # If you want to override config values directly in code (less recommended for API keys):
    # processor = ResearchPaperProcessor(api_key="YOUR_GEMINI_API_KEY", hf_token="YOUR_HF_TOKEN", hf_repo_id="YOUR_HF_REPO_ID")


    # 2. Process papers
    # Example: Process a maximum of 3 papers related to "large language models"
    max_papers_to_process = 3
    search_query = "large language models"
    print(f"Processing up to {max_papers_to_process} papers related to '{search_query}'...")
    new_papers = processor.process_papers(max_papers=max_papers_to_process, search_query=search_query)

    if new_papers:
        print(f"\nSuccessfully processed {len(new_papers)} new papers. Check 'data/processed_dataset/' for results.")
    else:
        print("\nNo new papers processed.")

    # 3. (Optional) Validate the dataset after processing
    print("\nValidating the processed dataset...")
    validation_results = processor.validate_dataset()
    print(f"Validation Results:")
    print(f"- Total entries: {validation_results['total_files']}")
    print(f"- Valid QA pairs: {validation_results['valid_entries']}")
    print(f"- Issues found: {len(validation_results['validation_issues'])}")
    if validation_results['validation_issues']:
        print("Issues found during validation. Check validation_issues in results for details.")
    else:
        print("Dataset validation passed.")


if __name__ == "__main__":
    main()
