"""Factory for creating OCR instances."""

from typing import Any
from .base import BaseOCR
from .mistral import MistralOCR

# Conditionally import other OCR implementations
try:
    from .gemini_ocr import GeminiOCR
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

try:
    from .ollama_ocr import OllamaOCR
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False

try:
    from .hf_transformers_ocr import HFTransformersOCR
    HF_TRANSFORMERS_AVAILABLE = True
except ImportError:
    HF_TRANSFORMERS_AVAILABLE = False

def create_ocr(ocr_type: str, **kwargs: Any) -> BaseOCR:
    """
    Create an OCR instance.

    Args:
        ocr_type: Type of OCR to create
        **kwargs: Additional arguments to pass to the OCR constructor

    Returns:
        OCR instance

    Raises:
        ValueError: If the OCR type is not supported
        ImportError: If the OCR implementation requires unavailable dependencies
    """
    # Build OCR map dynamically based on available implementations
    ocr_map = {}
    ocr_map["mistral"] = MistralOCR

    if GEMINI_AVAILABLE:
        ocr_map["gemini"] = GeminiOCR

    if OLLAMA_AVAILABLE:
        ocr_map["ollama"] = OllamaOCR

    if HF_TRANSFORMERS_AVAILABLE:
        ocr_map["transformers"] = HFTransformersOCR
        ocr_map["hf"] = HFTransformersOCR  # Alias for easier use

    if ocr_type not in ocr_map:
        available_types = list(ocr_map.keys())
        raise ValueError(f"Unsupported OCR type: {ocr_type}. Available types: {available_types}")

    try:
        # Check for required API keys based on OCR type
        if ocr_type == "mistral" and "api_key" not in kwargs and "MISTRAL_API_KEY" not in os.environ:
            raise ValueError("Mistral API key is required. Set it via the api_key parameter or MISTRAL_API_KEY environment variable.")

        if ocr_type == "gemini" and "api_key" not in kwargs and "GOOGLE_API_KEY" not in os.environ:
            raise ValueError("Google API key is required. Set it via the api_key parameter or GOOGLE_API_KEY environment variable.")

        # No API key checks for transformers or ollama OCR types

        return ocr_map[ocr_type](**kwargs)
    except ImportError as e:
        raise ImportError(f"Missing dependencies for OCR type '{ocr_type}': {e}")
    except Exception as e:
        raise RuntimeError(f"Error creating OCR of type '{ocr_type}': {e}")
