from .base import BaseOCR
from .ollama_ocr import OllamaOCR
from .huggingface_ocr import HuggingFaceOCR

def create_ocr(ocr_type: str, **kwargs):
    """Create an OCR instance."""
    if ocr_type == "ollama":
        return OllamaOCR(**kwargs)
    elif ocr_type == "huggingface":
        return HuggingFaceOCR(**kwargs)
    else:
        raise ValueError(f"Unsupported OCR type: {ocr_type}")
