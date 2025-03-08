# papertuner/__init__.py
"""
PaperTuner: A simple tool to generate datasets for fine-tuning LLMs.
"""

__version__ = "0.1.0"

from .config import load_config
from .data import HGDataset
from .ocr import OCRBase, GeminiOCR
from .sources import SourceClient, ArxivClient
