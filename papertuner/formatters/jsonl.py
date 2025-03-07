import json
from typing import List, Dict, Any, Optional, IO
from .base import BaseFormatter


class JsonlFormatter(BaseFormatter):
    """Formatter for JSONL format, commonly used for LLM fine-tuning."""

    def __init__(self, **kwargs):
        """
        Initialize the JSONL formatter.

        Args:
            **kwargs: Optional additional configuration parameters (for forward compatibility)
        """
        # Store any future configuration parameters
        self.config = kwargs

    def format_entry(self, paper_metadata: Dict[str, Any], text: str) -> Dict[str, Any]:
        """
        Format a single paper entry for the dataset in JSONL format.

        Args:
            paper_metadata: Metadata about the paper
            text: Extracted text from the paper

        Returns:
            Formatted entry with messages field for fine-tuning
        """
        # Create a system message that instructs the model about the paper
        system_message = (
            f"You are an AI assistant that has read the research paper titled '{paper_metadata['title']}' "
            f"by {', '.join(paper_metadata['authors'])}. "
            f"The paper was published on {paper_metadata['published'].strftime('%Y-%m-%d')} "
            f"in the following categories: {', '.join(paper_metadata['categories'])}."
        )

        # Create a user message asking about the paper
        user_message = "Can you summarize the key findings and contributions of this paper?"

        # Create an assistant message with the paper summary
        assistant_message = paper_metadata['summary']

        # Format in the style expected by most LLM fine-tuning frameworks
        return {
            "messages": [
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message},
                {"role": "assistant", "content": assistant_message}
            ],
            "metadata": {
                "paper_id": paper_metadata["id"],
                "title": paper_metadata["title"],
                "authors": paper_metadata["authors"],
                "published": paper_metadata["published"].isoformat(),
                "categories": paper_metadata["categories"],
            }
        }

    def save(self, entries: List[Dict[str, Any]], output_path: str) -> None:
        """
        Save the formatted entries to a JSONL file.

        Args:
            entries: List of formatted entries
            output_path: Path to save the dataset to
        """
        with open(output_path, "w", encoding="utf-8") as f:
            for entry in entries:
                f.write(json.dumps(entry) + "\n")
