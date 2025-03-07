import os
import logging
from typing import List, Dict, Any, Optional
from .base import BaseFormatter

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    import datasets
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False

class HuggingFaceFormatter(BaseFormatter):
    """Formatter for HuggingFace datasets format."""

    def __init__(self, save_locally: bool = True, push_to_hub: bool = False,
                hub_dataset_name: Optional[str] = None, hub_token: Optional[str] = None):
        """
        Initialize the HuggingFace formatter.

        Args:
            save_locally: Whether to save the dataset locally
            push_to_hub: Whether to push the dataset to the HuggingFace Hub
            hub_dataset_name: Name of the dataset on the HuggingFace Hub (required if push_to_hub is True)
            hub_token: HuggingFace token for pushing to the Hub (or use HF_TOKEN env variable)
        """
        if not DATASETS_AVAILABLE:
            raise ImportError("The 'datasets' package is required for HuggingFaceFormatter. "
                             "Install it with 'pip install datasets'")

        self.save_locally = save_locally
        self.push_to_hub = push_to_hub
        self.hub_dataset_name = hub_dataset_name
        self.hub_token = hub_token or os.environ.get("HF_TOKEN")

        if self.push_to_hub:
            if not self.hub_dataset_name:
                raise ValueError("hub_dataset_name is required when push_to_hub is True")
            if not self.hub_token:
                raise ValueError("hub_token or HF_TOKEN environment variable is required when push_to_hub is True")

    def format_entry(self, paper_metadata: Dict[str, Any], text: str) -> Dict[str, Any]:
        """Format a single paper entry for the dataset."""
        # Format date as string if it's a datetime object
        published_date = paper_metadata["published"]
        if hasattr(published_date, 'isoformat'):
            published_date = published_date.isoformat()

        return {
            "paper_id": paper_metadata["id"],
            "title": paper_metadata["title"],
            "authors": paper_metadata["authors"],
            "published_date": published_date,
            "categories": paper_metadata["categories"],
            "summary": paper_metadata["summary"],
            "full_text": text,
        }

    def save(self, entries: List[Dict[str, Any]], output_path: str) -> None:
        """Save the formatted entries as a HuggingFace dataset."""
        if not entries:
            logger.warning("No entries to save to HuggingFace dataset")
            return

        # Convert to format expected by datasets
        dataset_dict = {}
        for key in entries[0].keys():
            dataset_dict[key] = [entry.get(key) for entry in entries]

        # Create the dataset
        dataset = datasets.Dataset.from_dict(dataset_dict)

        # Save locally if requested
        if self.save_locally:
            dataset.save_to_disk(output_path)
            logger.info(f"Dataset saved locally to {output_path}")

        # Push to HuggingFace Hub if requested
        if self.push_to_hub:
            dataset.push_to_hub(self.hub_dataset_name, token=self.hub_token)
            logger.info(f"Dataset pushed to HuggingFace Hub: {self.hub_dataset_name}")