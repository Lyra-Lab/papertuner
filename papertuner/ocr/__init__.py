from .base import BaseOCR
from .arxiv_integrated import ArxivIntegratedOCR
from .mistral import MistralOCR

# Conditionally import other OCR implementations
try:
    from .gemini_ocr import GeminiOCR
except ImportError:
    pass

try:
    from .ollama_ocr import OllamaOCR
except ImportError:
    pass

try:
    from .hf_transformers_ocr import HFTransformersOCR
except ImportError:
    pass

from .factory import create_ocr
