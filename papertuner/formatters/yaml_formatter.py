from typing import List, Dict, Any, Optional
import yaml
from .base import BaseFormatter


class YAMLFormatter(BaseFormatter):
    """Formatter for YAML format."""

    def __init__(self, include_text: bool = True, separate_files: bool = False):
        """
        Initialize the YAML formatter.

        Args:
            include_text: Whether to include the full text in the YAML
            separate_files: Whether to save each entry as a separate file
        """
        self.include_text = include_text
        self.separate_files = separate_files

    def format_entry(self, paper_metadata: Dict[str, Any], text: str) -> Dict[str, Any]:
        """Format a single paper entry for the dataset."""
        entry = {
            "paper_id": paper_metadata["id"],
            "title": paper_metadata["title"],
            "authors": paper_metadata["authors"],
            "published_date": paper_metadata["published"].isoformat(),
            "categories": paper_metadata["categories"],
            "summary": paper_metadata["summary"],
            "metadata": {
                key: value for key, value in paper_metadata.items()
                if key not in ["id", "title", "authors", "published", "categories", "summary"]
            }
        }

        if self.include_text:
            entry["full_text"] = text

        return entry

    def save(self, entries: List[Dict[str, Any]], output_path: str) -> None:
        """Save the formatted entries as YAML file(s)."""
        if not entries:
            raise ValueError("No entries to save")

        if self.separate_files:
            import os
            # Create directory if it doesn't exist
            os.makedirs(output_path, exist_ok=True)

            # Save each entry as a separate file
            for entry in entries:
                file_path = os.path.join(output_path, f"{entry['paper_id']}.yaml")
                with open(file_path, 'w', encoding='utf-8') as f:
                    yaml.dump(entry, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
        else:
            # Save all entries in a single file
            with open(output_path, 'w', encoding='utf-8') as f:
                yaml.dump(entries, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
