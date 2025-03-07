from typing import Dict, Any, Optional
from .base import BaseOCR
from .mistral import MistralOCR


def create_ocr(ocr_type: str, **kwargs) -> BaseOCR:
    """
    Factory function to create OCR instances.
    
    Args:
        ocr_type: Type of OCR to create ("mistral", etc.)
        **kwargs: Additional arguments to pass to the OCR constructor
        
    Returns:
        An instance of the requested OCR class
    """
    ocr_map = {
        "mistral": MistralOCR,
    }
    
    if ocr_type not in ocr_map:
        raise ValueError(f"Unknown OCR type: {ocr_type}. Available types: {list(ocr_map.keys())}")
    
    return ocr_map[ocr_type](**kwargs) 