# papertuner/__init__.py
"""
PaperTuner: A simple tool to generate datasets for fine-tuning LLMs.
"""

import logging

__version__ = "0.1.0"

from .config import load_config
from .data import HGDataset
from .ocr import OCRBase, GeminiOCR, PyMuPDFOCR
from .sources import SourceClient, ArxivClient

def setup_logging(level=logging.INFO):
    """
    Set up basic logging configuration.

    Args:
        level (int): Logging level (e.g., logging.DEBUG, logging.INFO, logging.WARNING)
    """
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

# Set a default logger for the papertuner module to avoid no-op logging
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())
