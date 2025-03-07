from .base import BaseOCR
from .ollama_ocr import OllamaOCR

def create_ocr(ocr_type: str, **kwargs):
    """Create an OCR instance."""
    if ocr_type == "ollama":
        return OllamaOCR(**kwargs)
    else:
        raise ValueError(f"Unsupported OCR type: {ocr_type}")