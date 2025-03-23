"""
PaperTuner: A tool for creating research assistant models.

This package contains functionality for:
1. Creating datasets from research papers
2. Fine-tuning language models for research assistance
"""

__version__ = "0.0.5"

# Import key classes so users can import directly from papertuner
from papertuner.dataset import ResearchPaperProcessor
from papertuner.train import ResearchAssistantTrainer

# Export key functions and classes
__all__ = [
    "ResearchPaperProcessor",
    "ResearchAssistantTrainer"
]
