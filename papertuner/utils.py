"""Utility functions for PaperTuner."""

import os
import json
import time
import logging
from pathlib import Path
from tenacity import retry, stop_after_attempt, wait_exponential
from collections import defaultdict

from papertuner.config import logger

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

def load_json_file(file_path):
    """Load and parse a JSON file."""
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        logger.warning(f"Failed to load {file_path}: {e}")
        return None

def save_json_file(data, file_path, use_temp=True):
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

def validate_qa_pair(qa_pair):
    """Apply quality checks to ensure the QA pair focuses on problem-solving approaches."""
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
