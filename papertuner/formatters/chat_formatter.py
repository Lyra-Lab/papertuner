from typing import List, Dict, Any, Optional, Literal
import json
from .base import BaseFormatter


class ChatFormatter(BaseFormatter):
    """Formatter for various chat formats (OpenAI, Claude, LLama, etc.)."""
    
    FORMATS = Literal["openai", "claude", "llama", "mistral", "gemini"]
    
    def __init__(self, format_type: FORMATS = "openai", 
                 tasks: List[str] = ["summary"], 
                 system_template: Optional[str] = None,
                 include_metadata: bool = True):
        """
        Initialize the chat formatter.
        
        Args:
            format_type: Chat format to use (openai, claude, llama, mistral, gemini)
            tasks: List of tasks to create conversations for (summary, qa, etc.)
            system_template: Custom system message template (use {placeholders} for metadata)
            include_metadata: Whether to include metadata in the output
        """
        self.format_type = format_type
        self.tasks = tasks
        self.system_template = system_template
        self.include_metadata = include_metadata
        
        # Define default system templates
        self._default_templates = {
            "summary": "You are an AI assistant that has read the research paper titled '{title}' "
                      "by {authors}. The paper was published on {published_date} "
                      "in the following categories: {categories}.",
            "qa": "You are an AI assistant that helps answer questions about research papers. "
                 "You have access to the paper titled '{title}' by {authors}, "
                 "published on {published_date}.",
            "extraction": "You are an AI assistant that extracts structured information from research papers. "
                         "You're working with the paper '{title}' by {authors}."
        }
        
        # Define task prompts
        self._task_prompts = {
            "summary": "Can you summarize the key findings and contributions of this paper?",
            "qa": "What are the main methods used in this paper and what problem does it solve?",
            "extraction": "Extract the following information from the paper: research questions, methodology, main results, and limitations."
        }
    
    def _format_claude(self, paper_metadata: Dict[str, Any], text: str, task: str) -> Dict[str, Any]:
        """Format in Claude's conversation format."""
        system = self._get_system_message(paper_metadata, task)
        
        return {
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": self._task_prompts.get(task, "Summarize this paper.")},
                {"role": "assistant", "content": paper_metadata["summary"] if task == "summary" else ""}
            ],
            "metadata": paper_metadata if self.include_metadata else None
        }
    
    def _format_openai(self, paper_metadata: Dict[str, Any], text: str, task: str) -> Dict[str, Any]:
        """Format in OpenAI's chat format."""
        system = self._get_system_message(paper_metadata, task)
        
        return {
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": self._task_prompts.get(task, "Summarize this paper.")},
                {"role": "assistant", "content": paper_metadata["summary"] if task == "summary" else ""}
            ],
            "metadata": paper_metadata if self.include_metadata else None
        }
    
    def _format_llama(self, paper_metadata: Dict[str, Any], text: str, task: str) -> Dict[str, Any]:
        """Format in Llama's chat format."""
        system = self._get_system_message(paper_metadata, task)
        
        return {
            "conversations": [
                {"from": "system", "value": system},
                {"from": "human", "value": self._task_prompts.get(task, "Summarize this paper.")},
                {"from": "assistant", "value": paper_metadata["summary"] if task == "summary" else ""}
            ],
            "metadata": paper_metadata if self.include_metadata else None
        }
    
    def _format_mistral(self, paper_metadata: Dict[str, Any], text: str, task: str) -> Dict[str, Any]:
        """Format in Mistral's chat format."""
        system = self._get_system_message(paper_metadata, task)
        
        return {
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": self._task_prompts.get(task, "Summarize this paper.")},
                {"role": "assistant", "content": paper_metadata["summary"] if task == "summary" else ""}
            ],
            "metadata": paper_metadata if self.include_metadata else None
        }
    
    def _format_gemini(self, paper_metadata: Dict[str, Any], text: str, task: str) -> Dict[str, Any]:
        """Format in Gemini's content format."""
        system = self._get_system_message(paper_metadata, task)
        
        return {
            "contents": [
                {"role": "user", "parts": [{"text": f"{system}\n\n{self._task_prompts.get(task, 'Summarize this paper.')}"}]},
                {"role": "model", "parts": [{"text": paper_metadata["summary"] if task == "summary" else ""}]}
            ],
            "metadata": paper_metadata if self.include_metadata else None
        }
    
    def _get_system_message(self, paper_metadata: Dict[str, Any], task: str) -> str:
        """Get the system message for a paper."""
        if self.system_template:
            # Use custom template
            return self.system_template.format(
                title=paper_metadata["title"],
                authors=", ".join(paper_metadata["authors"]),
                published_date=paper_metadata["published"].isoformat(),
                categories=", ".join(paper_metadata["categories"])
            )
        else:
            # Use default template for the task
            template = self._default_templates.get(task, self._default_templates["summary"])
            return template.format(
                title=paper_metadata["title"],
                authors=", ".join(paper_metadata["authors"]),
                published_date=paper_metadata["published"].isoformat(),
                categories=", ".join(paper_metadata["categories"])
            )
    
    def format_entry(self, paper_metadata: Dict[str, Any], text: str) -> List[Dict[str, Any]]:
        """Format a single paper entry for the dataset in the specified chat format."""
        entries = []
        
        # Format based on the selected format type
        format_func = getattr(self, f"_format_{self.format_type}", self._format_openai)
        
        # Create an entry for each task
        for task in self.tasks:
            entry = format_func(paper_metadata, text, task)
            entry["task"] = task
            entries.append(entry)
        
        return entries
    
    def save(self, entries: List[Dict[str, Any]], output_path: str) -> None:
        """Save the formatted entries to a file."""
        # Flatten the list if entries contain lists
        flattened_entries = []
        for entry in entries:
            if isinstance(entry, list):
                flattened_entries.extend(entry)
            else:
                flattened_entries.append(entry)
        
        with open(output_path, "w", encoding="utf-8") as f:
            for entry in flattened_entries:
                f.write(json.dumps(entry) + "\n")