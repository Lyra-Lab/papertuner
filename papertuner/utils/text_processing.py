"""
Text processing utilities for PaperTuner.
"""
import re


def extract_xml_answer(text: str) -> str:
    """
    Extract answer content from various response formats.
    Handles both explicit <answer> tags and implicit answers (content after think/reasoning tags).
    
    Args:
        text: The text containing the answer in XML format
        
    Returns:
        The extracted answer text
    """
    # First try the original format with <answer> tags
    if "<answer>" in text and "</answer>" in text:
        answer = text.split("<answer>")[-1]
        answer = answer.split("</answer>")[0]
        return answer.strip()

    # Try the format with <think> tags
    elif "<think>" in text and "</think>" in text:
        # Extract everything after the </think> tag
        answer = text.split("</think>")[-1]
        return answer.strip()

    # Fallback: if no tags found, return the whole text
    return text.strip()


def extract_reasoning(text: str) -> str:
    """
    Extract reasoning content from various formats.
    Handles both <reasoning> and <think> tags.
    
    Args:
        text: The text containing the reasoning in XML format
        
    Returns:
        The extracted reasoning text
    """
    if "<reasoning>" in text and "</reasoning>" in text:
        reasoning = text.split("<reasoning>")[-1]
        reasoning = reasoning.split("</reasoning>")[0]
        return reasoning.strip()

    elif "<think>" in text and "</think>" in text:
        reasoning = text.split("<think>")[-1]
        reasoning = reasoning.split("</think>")[0]
        return reasoning.strip()

    return ""


def extract_sections(text: str) -> dict:
    """
    Extract key sections from ML research papers with flexible pattern matching.
    
    Args:
        text: The full text of the research paper
        
    Returns:
        Dictionary with extracted 'problem', 'methodology', and 'results' sections
    """
    try:
        # Problem/Introduction patterns
        problem_patterns = [
            r"(?:INTRODUCTION|BACKGROUND|PROBLEM STATEMENT|MOTIVATION|OVERVIEW).*?(?=\n\n[A-Z][A-Z\s]+\n)",
            r"(?:1[\.\s]+INTRODUCTION|1[\.\s]+BACKGROUND|I[\.\s]+INTRODUCTION).*?(?=\n\n[0-9]+[\.\s]+[A-Z]|\n\n[I|V|X]+[\.\s]+[A-Z])",
            r"(?:\n\nIntroduction\n|\n\nBackground\n|\n\nMotivation\n).*?(?=\n\n[A-Z][a-z])"
        ]

        # Methodology patterns
        method_patterns = [
            r"(?:METHODOLOGY|METHOD|APPROACH|EXPERIMENTAL DESIGN|PROPOSED METHOD|MODEL ARCHITECTURE|SYSTEM DESIGN|NETWORK ARCHITECTURE|IMPLEMENTATION|PROPOSED APPROACH).*?(?=\n\n[A-Z][A-Z\s]+\n)",
            r"(?:[2-4][\.\s]+(?:METHODOLOGY|METHOD|APPROACH|PROPOSED|MODEL|ARCHITECTURE)).*?(?=\n\n[0-9]+[\.\s]+[A-Z]|\n\n[I|V|X]+[\.\s]+[A-Z])",
            r"(?:\n\nMethodology\n|\n\nMethod\n|\n\nApproach\n|\n\nProposed method\n|\n\nArchitecture\n|\n\nModel\n|\n\nImplementation\n).*?(?=\n\n[A-Z][a-z])"
        ]

        # Results patterns
        result_patterns = [
            r"(?:RESULTS|EVALUATION|FINDINGS|EXPERIMENTS|EXPERIMENTAL RESULTS|PERFORMANCE|EVALUATION RESULTS).*?(?=\n\n[A-Z][A-Z\s]+\n)",
            r"(?:[3-6][\.\s]+(?:RESULTS|EVALUATION|EXPERIMENTS|PERFORMANCE)).*?(?=\n\n[0-9]+[\.\s]+[A-Z]|\n\n[I|V|X]+[\.\s]+[A-Z])",
            r"(?:\n\nResults\n|\n\nEvaluation\n|\n\nExperiments\n|\n\nPerformance\n|\n\nExperimental results\n).*?(?=\n\n[A-Z][a-z])"
        ]

        # Try all patterns for each section type
        problem_text = ""
        for pattern in problem_patterns:
            match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
            if match:
                problem_text = match.group(0)
                break

        method_text = ""
        for pattern in method_patterns:
            match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
            if match:
                method_text = match.group(0)
                break

        result_text = ""
        for pattern in result_patterns:
            match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
            if match:
                result_text = match.group(0)
                break

        # If we still don't have the methodology section, try a fallback approach
        if not method_text:
            # Look for sections that might contain methodology information
            method_related_keywords = [
                "architecture", "network", "model", "algorithm", "framework",
                "implementation", "system", "approach", "design", "experiment"
            ]

            # Search for paragraphs with methodology-related content
            paragraphs = re.split(r'\n\n+', text)
            method_paragraphs = []

            for paragraph in paragraphs:
                # Check if paragraph is likely about methodology
                if any(keyword in paragraph.lower() for keyword in method_related_keywords):
                    if len(paragraph) > 100:  # Only include substantial paragraphs
                        method_paragraphs.append(paragraph)

            if method_paragraphs:
                method_text = "\n\n".join(method_paragraphs[:3])  # Limit to first few relevant paragraphs

        # If we identified any sections, return them
        sections = {
            "problem": problem_text.strip(),
            "methodology": method_text.strip(),
            "results": result_text.strip()
        }

        return sections

    except Exception as e:
        print(f"Error extracting core sections: {e}")
        return {}


def validate_qa_pair(qa_pair: dict) -> bool:
    """
    Apply quality checks to ensure the QA pair focuses on problem-solving approaches.
    
    Args:
        qa_pair: Dictionary containing 'question' and 'answer' keys
        
    Returns:
        True if the QA pair is valid, False otherwise
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