"""Utility functions for PaperTuner."""

import os
import json
import time
import logging
import hashlib
import requests
import tempfile
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Callable
import concurrent.futures
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from collections import defaultdict

from papertuner.config import logger, DEFAULT_LLM_CONFIG

class LLMClient:
    """A unified client for interacting with different LLM providers."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize the LLM client with configuration."""
        self.config = config or DEFAULT_LLM_CONFIG
        self.provider = self.config.get("provider", "gemini")
        self.model = self.config.get("model", "gemini-2.0-flash")
        self.temperature = self.config.get("temperature", 0.3)
        self.max_tokens = self.config.get("max_tokens", 1500)
        self.timeout = self.config.get("timeout", 30)
        self.max_retries = self.config.get("max_retries", 3)
        
        # Setup clients based on available APIs
        self._setup_clients()
    
    def _setup_clients(self):
        """Setup API clients for available providers."""
        self.clients = {}
        
        # Setup Gemini client if API key is available
        if os.getenv("GEMINI_API_KEY"):
            try:
                from openai import OpenAI
                self.clients["gemini"] = OpenAI(
                    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
                    api_key=os.getenv("GEMINI_API_KEY")
                )
                logger.info("Gemini client initialized")
            except ImportError:
                logger.warning("OpenAI package not installed, Gemini API unavailable")
        
        # Setup OpenAI client if API key is available
        if os.getenv("OPENAI_API_KEY"):
            try:
                from openai import OpenAI
                self.clients["openai"] = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
                logger.info("OpenAI client initialized")
            except ImportError:
                logger.warning("OpenAI package not installed, OpenAI API unavailable")
        
        # Setup Anthropic client if API key is available
        if os.getenv("ANTHROPIC_API_KEY"):
            try:
                import anthropic
                self.clients["anthropic"] = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
                logger.info("Anthropic client initialized")
            except ImportError:
                logger.warning("Anthropic package not installed, Anthropic API unavailable")
        
        if not self.clients:
            logger.warning("No LLM clients available. Set API keys in environment variables.")
    
    @retry(
        stop=stop_after_attempt(3), 
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((requests.exceptions.RequestException, TimeoutError))
    )
    def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate text from a prompt using the configured LLM.
        
        Args:
            prompt: The text prompt to send
            **kwargs: Override configuration options
            
        Returns:
            str: The model's response text
        """
        # Override defaults with any provided kwargs
        provider = kwargs.get("provider", self.provider)
        model = kwargs.get("model", self.model)
        temperature = kwargs.get("temperature", self.temperature)
        max_tokens = kwargs.get("max_tokens", self.max_tokens)
        
        # Check if the requested provider is available
        if provider not in self.clients:
            fallback = self.config.get("fallback_provider")
            if fallback and fallback in self.clients:
                logger.warning(f"Provider {provider} unavailable, falling back to {fallback}")
                provider = fallback
                model = self.config.get("fallback_model", model)
            else:
                available = list(self.clients.keys())
                if available:
                    provider = available[0]
                    logger.warning(f"Using available provider: {provider}")
                else:
                    raise ValueError("No LLM providers available")
        
        client = self.clients[provider]
        
        try:
            if provider in ["gemini", "openai"]:
                # Use ChatCompletion API for OpenAI and Gemini
                response = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_tokens=max_tokens,
                    timeout=self.timeout
                )
                return response.choices[0].message.content.strip()
            
            elif provider == "anthropic":
                # Use Anthropic's API
                response = client.messages.create(
                    model=model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    messages=[{"role": "user", "content": prompt}]
                )
                return response.content[0].text
            
            else:
                raise ValueError(f"Unsupported provider: {provider}")
                
        except Exception as e:
            logger.error(f"API call to {provider} failed: {e}")
            raise

def load_json_file(file_path: Union[str, Path]) -> Optional[Dict[str, Any]]:
    """
    Load and parse a JSON file.
    
    Args:
        file_path: Path to the JSON file
        
    Returns:
        Optional[Dict[str, Any]]: Parsed JSON data or None if file cannot be loaded
    """
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        logger.warning(f"Failed to load {file_path}: {e}")
        return None

def save_json_file(data: Any, file_path: Union[str, Path], use_temp: bool = True) -> bool:
    """
    Save data to a JSON file.

    Args:
        data: The data to save
        file_path: Path to save to
        use_temp: Whether to use a temporary file first (safer)

    Returns:
        bool: Success status
    """
    file_path = Path(file_path)
    temp_file = file_path.parent / f".tmp_{file_path.name}" if use_temp else file_path

    try:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(temp_file, 'w') as f:
            json.dump(data, f, indent=2)

        if use_temp:
            temp_file.rename(file_path)

        return True
    except Exception as e:
        logger.error(f"Failed to save to {file_path}: {e}")
        if use_temp and temp_file.exists():
            try:
                os.remove(temp_file)
            except OSError:
                pass
        return False

def validate_qa_pair(qa_pair: Dict[str, Any]) -> bool:
    """
    Apply quality checks to ensure the QA pair focuses on problem-solving approaches.
    
    Args:
        qa_pair: A question-answer pair to validate
        
    Returns:
        bool: True if the QA pair meets quality criteria
    """
    if not qa_pair or not qa_pair.get("question") or not qa_pair.get("answer"):
        return False

    question = qa_pair["question"]
    answer = qa_pair["answer"]

    # Check minimum lengths
    if len(question) < 20 or len(answer) < 250:
        return False

    # Check for problem-solving focus in question
    question_lower = question.lower()
    problem_solving_keywords = ["how", "why", "approach", "solve", "address", "implement",
                               "architecture", "design", "technique", "method", "decision",
                               "strategy", "challenge", "framework", "structure", "mechanism"]

    if not any(keyword in question_lower for keyword in problem_solving_keywords):
        return False

    # Check for technical content in answer
    answer_lower = answer.lower()
    technical_keywords = ["model", "algorithm", "parameter", "layer", "network", "training",
                         "architecture", "implementation", "performance", "component",
                         "structure", "design", "feature", "optimization"]

    if not any(keyword in answer_lower for keyword in technical_keywords):
        return False

    # Check for comparative/reasoning language in answer
    reasoning_keywords = ["because", "therefore", "advantage", "benefit", "compared",
                         "better than", "instead of", "rather than", "alternative",
                         "trade-off", "superior", "effective", "efficient", "chosen"]

    if not any(keyword in answer_lower for keyword in reasoning_keywords):
        return False

    return True

def compute_hash(content: str) -> str:
    """
    Compute a unique hash for content to use as identifier.
    
    Args:
        content: String content to hash
        
    Returns:
        str: Hash of the content
    """
    return hashlib.sha256(content.encode('utf-8')).hexdigest()

def download_file(url: str, output_path: Optional[Path] = None, timeout: int = 30) -> Optional[Path]:
    """
    Download a file from a URL.
    
    Args:
        url: URL to download from
        output_path: Path to save file (if None, uses a temporary file)
        timeout: Request timeout in seconds
        
    Returns:
        Optional[Path]: Path to the downloaded file or None if download failed
    """
    try:
        if output_path is None:
            # Create a temporary file
            fd, temp_path = tempfile.mkstemp(suffix=".tmp")
            os.close(fd)
            output_path = Path(temp_path)
        
        response = requests.get(url, stream=True, timeout=timeout)
        response.raise_for_status()
        
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(8192):
                f.write(chunk)
        
        return output_path
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to download {url}: {e}")
        return None

def run_in_parallel(func: Callable, items: List[Any], max_workers: int = 5, **kwargs) -> List[Any]:
    """
    Run a function on multiple items in parallel.
    
    Args:
        func: Function to run on each item
        items: List of items to process
        max_workers: Maximum number of parallel workers
        **kwargs: Additional kwargs to pass to func
        
    Returns:
        List[Any]: Results for each item
    """
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Create a future for each item
        future_to_item = {executor.submit(func, item, **kwargs): item for item in items}
        
        # Process results as they complete
        for future in concurrent.futures.as_completed(future_to_item):
            item = future_to_item[future]
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                logger.error(f"Error processing item {item}: {e}")
                results.append(None)
    
    return results

# Legacy API call function maintained for backwards compatibility
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def api_call(client, prompt, max_tokens=1500):
    """
    Make a resilient API call to an LLM service.

    Args:
        client: The API client (OpenAI or compatible)
        prompt: The text prompt to send
        max_tokens: Maximum tokens in the response

    Returns:
        str: The model's response text
    """
    try:
        response = client.chat.completions.create(
            model="gemini-2.0-flash",  # Could be configurable
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=max_tokens
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"API call failed: {e}")
        raise  # Re-raise for retry to work
