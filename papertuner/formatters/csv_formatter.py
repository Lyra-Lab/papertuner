import csv
from typing import List, Dict, Any, Optional
import json
from .base import BaseFormatter


class CSVFormatter(BaseFormatter):
    """Formatter for CSV format."""

    def __init__(self, include_text: bool = True, delimiter: str = ",",
                 quotechar: str = '"', flatten_metadata: bool = True):
        """
        Initialize the CSV formatter.

        Args:
            include_text: Whether to include the full text in the CSV
            delimiter: CSV delimiter character
            quotechar: CSV quote character
            flatten_metadata: Whether to flatten metadata fields into separate columns
        """
        self.include_text = include_text
        self.delimiter = delimiter
        self.quotechar = quotechar
        self.flatten_metadata = flatten_metadata

    def format_entry(self, paper_metadata: Dict[str, Any], text: str) -> Dict[str, Any]:
        """Format a single paper entry for the dataset."""
        entry = {
            "paper_id": paper_metadata["id"],
            "title": paper_metadata["title"],
            "authors": json.dumps(paper_metadata["authors"]),
            "published_date": paper_metadata["published"].isoformat(),
            "categories": json.dumps(paper_metadata["categories"]),
            "summary": paper_metadata["summary"],
        }

        if self.include_text:
            entry["full_text"] = text

        # Add any additional metadata fields
        if self.flatten_metadata:
            for key, value in paper_metadata.items():
                if key not in ["id", "title", "authors", "published", "categories", "summary"]:
                    if isinstance(value, (list, dict)):
                        entry[key] = json.dumps(value)
                    else:
                        entry[key] = value
        else:
            # Store metadata as a JSON string
            entry["metadata"] = json.dumps({
                key: value for key, value in paper_metadata.items()
                if key not in ["id", "title", "authors", "published", "categories", "summary"]
            })

        return entry

    def save(self, entries: List[Dict[str, Any]], output_path: str) -> None:
        """Save the formatted entries as a CSV file."""
        if not entries:
            raise ValueError("No entries to save")

        # Get fieldnames from the first entry
        fieldnames = list(entries[0].keys())

        with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(
                csvfile,
                fieldnames=fieldnames,
                delimiter=self.delimiter,
                quotechar=self.quotechar,
                quoting=csv.QUOTE_MINIMAL
            )

            writer.writeheader()
            for entry in entries:
                writer.writerow(entry)
