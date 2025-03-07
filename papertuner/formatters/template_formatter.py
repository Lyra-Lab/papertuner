from typing import List, Dict, Any, Optional, Callable
import json
import os
from string import Template
from .base import BaseFormatter


class TemplateFormatter(BaseFormatter):
    """Formatter that uses custom templates for different use cases."""

    def __init__(self,
                 template_string: Optional[str] = None,
                 template_file: Optional[str] = None,
                 output_format: str = "jsonl",
                 template_vars: Optional[Dict[str, str]] = None,
                 custom_processor: Optional[Callable[[Dict[str, Any], str], Dict[str, Any]]] = None):
        """
        Initialize the template formatter.

        Args:
            template_string: Template string to use
            template_file: Path to template file to load
            output_format: Format of the output (jsonl, json, txt)
            template_vars: Additional template variables to include
            custom_processor: Custom function to process entries before templating
        """
        self.output_format = output_format
        self.template_vars = template_vars or {}
        self.custom_processor = custom_processor

        # Load template from string or file
        if template_string:
            self.template = Template(template_string)
        elif template_file:
            with open(template_file, 'r', encoding='utf-8') as f:
                self.template = Template(f.read())
        else:
            # Default template for paper summarization
            self.template = Template('''
{
    "title": "$title",
    "authors": $authors_json,
    "published_date": "$published_date",
    "categories": $categories_json,
    "summary": "$summary",
    "text_sample": "$text_sample",
    "task": {
        "instruction": "Summarize the key findings of this research paper.",
        "response": "$summary"
    }
}''')

    def format_entry(self, paper_metadata: Dict[str, Any], text: str) -> Dict[str, Any]:
        """Format a single paper entry using the template."""
        # Apply custom processing if provided
        if self.custom_processor:
            processed_data = self.custom_processor(paper_metadata, text)
        else:
            processed_data = paper_metadata.copy()
            processed_data['text'] = text
            processed_data['text_sample'] = text[:500].replace('\n', ' ').replace('"', '\\"') + '...'

        # Prepare data for template substitution
        template_data = {
            'title': processed_data.get('title', '').replace('"', '\\"'),
            'authors_json': json.dumps(processed_data.get('authors', [])),
            'published_date': processed_data.get('published', '').isoformat() if hasattr(processed_data.get('published', ''), 'isoformat') else processed_data.get('published', ''),
            'categories_json': json.dumps(processed_data.get('categories', [])),
            'summary': processed_data.get('summary', '').replace('"', '\\"').replace('\n', ' '),
            'text_sample': processed_data.get('text_sample', '')
        }

        # Add any custom template variables
        template_data.update(self.template_vars)

        # Apply template
        formatted_str = self.template.substitute(template_data)

        # Return as string or parsed object based on format
        if self.output_format == 'txt':
            return {'formatted': formatted_str}
        else:
            return json.loads(formatted_str)

    def save(self, entries: List[Dict[str, Any]], output_path: str) -> None:
        """Save the formatted entries to a file."""
        if self.output_format == 'jsonl':
            with open(output_path, 'w', encoding='utf-8') as f:
                for entry in entries:
                    if 'formatted' in entry:
                        f.write(entry['formatted'] + '\n')
                    else:
                        f.write(json.dumps(entry) + '\n')
        elif self.output_format == 'json':
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(entries, f, indent=2)
        elif self.output_format == 'txt':
            with open(output_path, 'w', encoding='utf-8') as f:
                for entry in entries:
                    f.write(entry.get('formatted', '') + '\n\n')
        else:
            raise ValueError(f"Unsupported output format: {self.output_format}")
