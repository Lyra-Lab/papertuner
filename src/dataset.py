# experiment_script.py
import os
import json
import time
import datetime
import logging
from pathlib import Path
from collections import defaultdict
import requests
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from datasets import Dataset, concatenate_datasets
from huggingface_hub import create_repo, login
from tenacity import retry, stop_after_attempt, wait_exponential
import fitz  # PyMuPDF
import arxiv
from openai import OpenAI
import re

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Constants
RAW_DIR = Path("data/raw_dataset")
PROCESSED_DIR = Path("data/processed_dataset")
HF_TOKEN = os.getenv("HF_TOKEN")
HF_REPO_ID = os.getenv("HF_REPO_ID")

# API configuration constants
API_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
API_KEY = os.getenv("GEMINI_API_KEY") or "AIzaSyDP1tXAkNMw1caHcAIhOB4x9L0DyWMne58"

# Utility functions
def setup_dirs():
    try:
        RAW_DIR.mkdir(parents=True, exist_ok=True)
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        (PROCESSED_DIR / "papers").mkdir(parents=True, exist_ok=True)
        logging.info("Directories set up successfully.")
        return True
    except OSError as e:
        logging.error(f"Failed to setup directories: {e}")
        return False

def create_retry_strategy():
    return retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))

# OpenAI Client Setup
def api_call(client, prompt, max_tokens=1500):
    retry_strategy = create_retry_strategy()
    @retry_strategy
    def _api_call(client, prompt, max_tokens):
        try:
            response = client.chat.completions.create(
                model="gemini-2.0-flash",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=max_tokens
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logging.error(f"OpenAI API call failed: {e}")
            raise  # Re-raise for retry to work

    return _api_call(client, prompt, max_tokens)

# Download and Process
def has_been_processed(paper_id, processed_dir=PROCESSED_DIR):
    """
    Check if a paper has already been downloaded and processed.

    Args:
        paper_id (str): The ID of the paper to check.
        processed_dir (Path): Directory where processed files are stored.

    Returns:
        bool: True if the paper has been processed, False otherwise.
    """
    # Check if the processed file exists
    processed_file = processed_dir / "papers" / f"paper_{paper_id.split('/')[-1]}.json"

    if processed_file.exists():
        try:
            # Verify the file has valid content
            with open(processed_file, 'r') as f:
                data = json.load(f)
                # Check if the file has the expected structure
                if (data.get("metadata") and data.get("sections") and
                    data.get("qa") and data["qa"].get("question") and
                    data["qa"].get("answer")):
                    logging.info(f"Paper {paper_id} already processed. Skipping.")
                    return True
        except (json.JSONDecodeError, IOError) as e:
            logging.warning(f"Found existing but invalid processed file for {paper_id}: {e}")
            # We'll consider this as not processed and attempt to reprocess it
            return False

    return False

def extract_sections(text):
    """Extract key sections from ML research papers with more flexible pattern matching."""
    try:
        # More comprehensive patterns for ML papers
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

        # Log which sections were found
        found_sections = [k for k, v in sections.items() if v]
        if found_sections:
            logging.info(f"Extracted sections: {', '.join(found_sections)}")
        else:
            logging.warning("No sections extracted from paper")

        return sections

    except Exception as e:
        logging.error(f"Error extracting core sections: {e}")
        return {}

def generate_qa(client, paper_data, sections):
    """Generate a high-quality QA pair focused specifically on problem-solving approach selection."""
    # Prepare context with available content
    abstract = paper_data.get("abstract", "")
    problem = sections.get("problem", "")
    methodology = sections.get("methodology", "")
    results = sections.get("results", "")

    context = f"""
    Paper Title: {paper_data['title']}
    Abstract: {abstract}
    """

    if problem:
        context += f"\nProblem/Introduction: {problem[:500]}...\n"
    if methodology:
        context += f"\nMethodology/Approach: {methodology[:1000]}...\n"
    if results:
        context += f"\nResults: {results[:300]}...\n"

    prompt = f"""As an expert research methodology assistant, your task is to create a problem-solving focused question and answer pair based on this machine learning research paper.

{context}

IMPORTANT: The question MUST be about HOW to solve a specific technical challenge or WHY a particular architectural approach was chosen. Do NOT generate general questions about research purpose or findings.

Specifically, the question should ask one of the following:
1. "How did the researchers solve the problem of X...?" where X is a specific technical challenge mentioned in the paper
2. "Why did the researchers choose approach/architecture Y over alternatives...?" where Y is the specific methodology they implemented
3. "What architectural decisions were crucial for addressing the challenge of Z...?" where Z is a specific requirement or constraint

The answer should:
- Explain the specific approach or architecture that solved the problem
- Detail WHY this approach was superior to alternatives for this specific problem
- Include technical details about implementation decisions
- Mention constraints or requirements that influenced the architectural choices
- Highlight any novel components that made the solution effective

BAD EXAMPLE (too general):
Q: What is the purpose of the research?
A: The purpose of the research is to investigate the effectiveness of a new machine learning algorithm.

GOOD EXAMPLE (problem-solving focused):
Q: How did the researchers solve the challenge of training transformer models with limited computational resources?
A: The researchers addressed the computational limitation challenge through a novel parameter-efficient fine-tuning approach. Rather than training the entire transformer architecture (which would require significant GPU resources), they implemented adapter modules - small trainable components inserted between transformer layers. This approach reduced the trainable parameter count by 97% compared to full fine-tuning while maintaining 94% of the performance. They specifically chose adapter modules over alternatives like LoRA or prefix-tuning because adapters provided better stability during training and required less hyperparameter tuning. A critical implementation detail was their use of residual connections around each adapter, preventing gradient flow issues during backpropagation. This architectural decision was particularly important for maintaining model stability when working with limited training data.

Return only the question and answer in the format:
Q: [Problem-solving focused question]
A: [Detailed answer about the approach/architecture]
"""

    try:
        response = api_call(client, prompt, max_tokens=2000)

        # Parse the response
        q_start = response.find("Q:")
        a_start = response.find("A:")

        if q_start == -1 or a_start == -1:
            logging.warning(f"Could not parse Q&A format from response")
            return None

        question = response[q_start+2:a_start].strip()
        answer = response[a_start+2:].strip()

        # Additional validation for problem-solving focus
        question_lower = question.lower()
        solution_keywords = ["how", "why", "approach", "solve", "implement", "architecture",
                            "design", "technique", "method", "strategy", "decision"]

        if not any(keyword in question_lower for keyword in solution_keywords):
            logging.warning(f"Generated question lacks problem-solving focus: {question}")
            return None

        # Check answer quality
        if len(answer) < 250:  # Ensure detailed answers
            logging.warning(f"Generated answer too brief ({len(answer)} chars)")
            return None

        return {"question": question, "answer": answer}

    except Exception as e:
        logging.error(f"QA generation failed: {e}")
        return None

def validate_qa_pair(qa_pair):
    """Apply quality checks to ensure the QA pair focuses on problem-solving approaches."""
    if not qa_pair or not qa_pair.get("question") or not qa_pair.get("answer"):
        return False

    question = qa_pair["question"]
    answer = qa_pair["answer"]

    # Check minimum lengths
    if len(question) < 20 or len(answer) < 250:  # Increased minimum answer length
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

def download_pdf(session, url, paper_id):
    retry_strategy = create_retry_strategy()

    @retry_strategy
    def _download_pdf(session, url, paper_id):
        temp_path = RAW_DIR / f"temp_{paper_id.split('/')[-1]}.pdf"
        try:
            response = session.get(url, stream=True, timeout=10)
            response.raise_for_status()
            with open(temp_path, 'wb') as f:
                for chunk in response.iter_content(8192):
                    f.write(chunk)
            return temp_path
        except requests.exceptions.RequestException as e:
            logging.warning(f"Download failed for {url}: {str(e)}")
            return None

    return _download_pdf(session, url, paper_id)

def extract_text(pdf_path):
    if not os.path.exists(pdf_path):
        return ""
    def _extract():
        try:
            doc = fitz.open(pdf_path)
            text = " ".join([page.get_text() for page in doc])
            logging.info(f"Text extracted from {pdf_path}")
            return text
        except Exception as e:
            logging.error(f"Extraction failed for {pdf_path}: {str(e)}")
            return ""

    try:
        with ThreadPoolExecutor() as executor:
            future = executor.submit(_extract)
            return future.result(timeout=30)
    except TimeoutError:
        logging.warning(f"Timeout extracting text from {pdf_path}")
        return ""
    except Exception as e:
        logging.error(f"Unexpected error extracting text: {str(e)}")
        return ""

def save_paper(data, filename):
    (PROCESSED_DIR / "papers").mkdir(parents=True, exist_ok=True)
    temp_file = PROCESSED_DIR / "papers" / f".tmp_{filename}"
    target_file = PROCESSED_DIR / "papers" / filename

    try:
        with open(temp_file, 'w') as f:
            json.dump(data, f, indent=2)
        temp_file.rename(target_file)
        logging.info(f"Paper data saved to {target_file}")
        return True
    except Exception as e:
        logging.error(f"Failed to save {target_file}: {str(e)}")
        try:
            os.remove(temp_file)
        except OSError:
            pass
        return False

def process_paper(client, paper):
    # Check if paper has already been processed
    if has_been_processed(paper.entry_id):
        return None  # Skip this paper

    pdf_path = download_pdf(requests.Session(), paper.pdf_url, paper.entry_id)
    if not pdf_path:
        logging.warning(f"Skipping paper {paper.entry_id} due to download failure.")
        return None

    text = extract_text(pdf_path)
    if not text:
        logging.warning(f"Skipping paper {paper.entry_id} due to text extraction failure.")
        return None

    paper_data = {
        "id": paper.entry_id,
        "title": paper.title,
        "authors": [str(a) for a in paper.authors],
        "abstract": paper.summary,
        "categories": paper.categories,
        "pdf_url": paper.pdf_url
    }

    # Extract just the core sections we need
    sections = extract_sections(text)

    # Check if we have enough content to work with
    # Allow papers with just abstract and some content if we couldn't extract proper sections
    if not sections.get("methodology") and not sections.get("problem"):
        # Use the abstract and first 1000 characters as a fallback
        sections = {
            "problem": paper_data["abstract"],
            "methodology": text[:2000] if len(text) > 2000 else text,
            "results": text[-1000:] if len(text) > 3000 else ""
        }
        logging.info(f"Using abstract and text excerpts for paper {paper.entry_id} due to missing sections.")

    # Generate QA pair based on available content
    qa_pair = generate_qa(client, paper_data, sections)
    if not qa_pair or not validate_qa_pair(qa_pair):
        logging.warning(f"Skipping paper {paper.entry_id} due to low-quality QA pair.")
        return None

    # Return a simplified paper structure with just what we need
    return {
        "metadata": {
            "id": paper_data["id"],
            "title": paper_data["title"],
            "categories": paper_data["categories"]
        },
        "sections": sections,  # Keep the sections for reference
        "qa": qa_pair
    }

def load_processed_manifest():
    """
    Load the manifest of processed papers.

    Returns:
        list: List of paper IDs that have been processed.
    """
    manifest_path = PROCESSED_DIR / "manifest.json"
    if not manifest_path.exists():
        return []

    try:
        with open(manifest_path, 'r') as f:
            manifest = json.load(f)
            return [item.get("id") for item in manifest if item.get("id")]
    except (json.JSONDecodeError, IOError) as e:
        logging.warning(f"Failed to load manifest: {e}")
        return []

def save_to_manifest(new_items, manifest_path=PROCESSED_DIR / "manifest.json"):
    """
    Save new items to the manifest, appending to existing data.

    Args:
        new_items (list): New items to add to the manifest.
        manifest_path (Path): Path to the manifest file.
    """
    existing_items = []
    if manifest_path.exists():
        try:
            with open(manifest_path, 'r') as f:
                existing_items = json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            logging.warning(f"Failed to load existing manifest: {e}")
            # Start with an empty list if the file is corrupted

    # Combine existing and new items
    updated_manifest = existing_items + new_items

    # Write back to the manifest file
    with open(manifest_path, 'w') as f:
        json.dump(updated_manifest, f, indent=2)

    logging.info(f"Updated manifest with {len(new_items)} new items. Total: {len(updated_manifest)}")

def process_all_papers(max_papers=100):
    # Load already processed papers to avoid duplication
    processed_papers = load_processed_manifest()
    processed_ids = set(processed_papers)

    client = arxiv.Client()
    search = arxiv.Search(
        query=" OR ".join([
            "machine learning",
            "deep learning",
            "large language models",
            "LLM",
            "natural language processing",
            "NLP",
            "transformers",
            "neural networks",
            "computer vision",
            "reinforcement learning",
            "generative models",
            "transfer learning",
            "few-shot learning",
            "zero-shot learning",
            "meta-learning"
        ]),
        max_results=max_papers + len(processed_ids),  # Get more results to account for skipping
        sort_by=arxiv.SortCriterion.Relevance,
        sort_order=arxiv.SortOrder.Descending
    )

    # Use API_BASE_URL and API_KEY constants here
    openrouter_client = OpenAI(
        base_url=API_BASE_URL,
        api_key=API_KEY
    )

    new_manifest_items = []
    papers_processed = 0
    error_occurred = False

    for result in tqdm(client.results(search)):
        # Skip if we've already processed this paper
        if result.entry_id in processed_ids:
            logging.info(f"Skipping already processed paper: {result.entry_id}")
            continue

        # Limit to the requested number of new papers
        if papers_processed >= max_papers:
            logging.info(f"Reached maximum number of papers to process: {max_papers}")
            break

        try:
            # Apply rate limiting
            if papers_processed > 0 and papers_processed % 5 == 0:
                time.sleep(1 + 0.5 * (papers_processed % 3))

            paper = process_paper(openrouter_client, result)
            if not paper:
                logging.warning(f"Failed to process paper: {result.entry_id}. Skipping.")
                continue

            filename = f"paper_{result.entry_id.split('/')[-1]}.json"
            if not save_paper(paper, filename):
                logging.error(f"Failed to save paper: {result.entry_id}. Skipping manifest update.")
                error_occurred = True  # Indicate a save error
                continue

            new_manifest_item = {
                "id": result.entry_id,
                "filename": filename,
                "title": result.title,
                "processed_date": datetime.datetime.now().isoformat()
            }
            new_manifest_items.append(new_manifest_item)
            papers_processed += 1

            logging.info(f"Successfully processed paper {papers_processed}/{max_papers}: {result.entry_id}")

        except Exception as e:
            logging.error(f"Exception during paper processing for {result.entry_id}: {e}")
            error_occurred = True  # Indicate a processing error
        finally:
            pdf_path = RAW_DIR / f"temp_{result.entry_id.split('/')[-1]}.pdf"
            if pdf_path and os.path.exists(pdf_path):
                try:
                    os.remove(pdf_path)
                except OSError as e:
                    logging.error(f"Failed to remove temp PDF {pdf_path}: {e}")

    # Only update the manifest if we have new items
    if new_manifest_items:
        save_to_manifest(new_manifest_items)
        logging.info(f"Added {len(new_manifest_items)} papers to manifest")
    else:
        logging.info("No new papers were processed")

    if error_occurred:
        logging.error("process_all_papers encountered errors during processing.")
        return new_manifest_items or None  # Return None only if no papers were successfully processed

    return new_manifest_items

def validate_dataset(processed_dir=PROCESSED_DIR):
    processed_files = list((processed_dir / "papers").glob("paper_*.json"))
    if not processed_files:
        raise FileNotFoundError(f"No processed files found in {processed_dir / 'papers'}")
    valid_count = 0
    issues = []

    for file in processed_files:
        try:
            with open(file, "r") as f:
                data = json.load(f)

                if not data.get("qa"):
                    issues.append(f"Missing QA pair in {file.name}")
                    continue

                q = data["qa"].get("question", "").strip()
                a = data["qa"].get("answer", "").strip()

                if len(q) < 10 or len(a) < 50:
                    issues.append(f"Short QA pair in {file.name}")
                else:
                    valid_count += 1
        except json.JSONDecodeError as e:
            issues.append(f"Invalid JSON format in {file.name}: {e}")
        except Exception as e:
            issues.append(f"Error validating {file.name}: {e}")

    return {
        "total_files": len(processed_files),
        "valid_entries": valid_count,
        "validation_issues": issues
    }

def upload_to_hf(processed_dir=PROCESSED_DIR, split_ratios=(0.8, 0.1, 0.1)):
    if not HF_TOKEN:
        raise ValueError("HF_TOKEN not set in environment variables")

    if not HF_REPO_ID:
        raise ValueError("HF_REPO_ID not set in environment variables")

    processed_files = list((processed_dir / "papers").glob("paper_*.json"))
    qa_pairs = []
    metadata = defaultdict(list)

    try:
        for file in processed_files:
            with open(file, "r") as f:
                data = json.load(f)

                if "qa" in data and data["qa"].get("question") and data["qa"].get("answer"):
                    qa_pairs.append({
                        "question": data["qa"]["question"],
                        "answer": data["qa"]["answer"],
                        "paper_id": data["metadata"]["id"],
                        "categories": data["metadata"]["categories"]
                    })

                    # Aggregate metadata
                    metadata["titles"].append(data["metadata"]["title"])
                    metadata["authors"].extend(data["metadata"].get("authors", []))
    except json.JSONDecodeError as e:
        logging.error(f"JSON decode error while preparing dataset for HF upload: {e}")
        return False  # Indicate upload preparation failure
    except Exception as e:
        logging.error(f"Error preparing dataset for HF upload: {e}")
        return False  # Indicate upload preparation failure

    dataset = Dataset.from_list(qa_pairs)
    split = dataset.train_test_split(test_size=split_ratios[1] + split_ratios[2])
    test_val = split["test"].train_test_split(test_size=split_ratios[2]/sum(split_ratios[1:]))
    final_dataset = concatenate_datasets([
        split["train"],
        test_val["train"],
        test_val["test"]
    ])

    card_content = f"""\
# Research Assistant QA Dataset

## Overview
- Contains {len(dataset)} validated question-answer pairs
- Derived from {len(processed_files)} research papers
- Categories: {', '.join(set(sum(metadata['categories'], [])))}

## Fields
- `question`: Research methodology question
- `answer`: Detailed methodology answer
- `paper_id`: Source paper identifier
- `categories`: arXiv categories
"""

    try:
        login(token=HF_TOKEN)
        create_repo(repo_id=HF_REPO_ID, repo_type="dataset", exist_ok=True)

        final_dataset.push_to_hub(
            HF_REPO_ID,
            commit_message=f"Add dataset v1.0 with {len(dataset)} entries"
        )

        # Upload README separately since dataset_card is not a valid parameter for push_to_hub
        with open("README.md", "w") as f:
            f.write(card_content)

        api = HfApi(token=HF_TOKEN)
        api.upload_file(
            path_or_fileobj="README.md",
            path_in_repo="README.md",
            repo_id=HF_REPO_ID,
            repo_type="dataset"
        )

        print(f"Dataset uploaded to https://huggingface.co/datasets/{HF_REPO_ID}")  # Keep print for user feedback
        logging.info(f"Dataset uploaded to https://huggingface.co/datasets/{HF_REPO_ID}")
        return True  # Indicate upload success

    except Exception as e:
        logging.error(f"Failed to upload dataset to Hugging Face Hub: {e}")
        return False  # Indicate upload failure


def generate_statistics(processed_dir=PROCESSED_DIR):
    processed_files = list((processed_dir / "papers").glob("paper_*.json"))
    stats = {
        "total_papers": len(processed_files),
        "avg_answer_length": 0,
        "category_distribution": defaultdict(int),
        "domain_breakdown": defaultdict(int)
    }

    total_chars = 0
    try:
        for file in processed_files:
            with open(file, "r") as f:
                data = json.load(f)
                stats["category_distribution"][data["metadata"]["categories"][0]] += 1
                total_chars += len(data["qa"]["answer"])

                # Extract domain from categories
                domain = data["metadata"]["categories"][0].split(".")[0]
                stats["domain_breakdown"][domain] += 1
    except json.JSONDecodeError as e:
        logging.error(f"JSON decode error while generating statistics: {e}")
        return None  # Indicate statistics generation failure
    except Exception as e:
        logging.error(f"Error generating statistics: {e}")
        return None  # Indicate statistics generation failure

    stats["avg_answer_length"] = total_chars / len(processed_files) if processed_files else 0
    return stats

if __name__ == "__main__":
    if not setup_dirs():
        exit(1)  # Stop if directory setup fails

    # Process papers from arXiv, only adding new ones
    new_papers = process_all_papers(max_papers=100)
    if new_papers is None and not load_processed_manifest():  # Only exit if we have no papers at all
        logging.error("Initial paper processing failed and no existing papers. Script execution halted.")
        exit(1)

    # If we have some papers (either new or previously processed), continue with validation
    results = validate_dataset()
    print(f"Validation Results:\n- Total entries: {results['total_files']}"
          f"\n- Valid QA pairs: {results['valid_entries']}"
          f"\n- Issues found: {len(results['validation_issues'])})")
    if results['validation_issues']:
        logging.warning(f"Dataset validation issues found: {results['validation_issues']}")

    if HF_TOKEN and HF_REPO_ID:
        if not upload_to_hf():  # upload_to_hf returns False on failure
            logging.error("Dataset upload to Hugging Face Hub failed.")
    else:
        logging.warning("Skipping HF upload: HF_TOKEN or HF_REPO_ID not set.")

    stats = generate_statistics()
    if stats:  # generate_statistics returns None on failure
        print("Dataset Statistics:")
        print(f"- Total papers: {stats['total_papers']}")
        print(f"- Average answer length: {stats['avg_answer_length']:.1f} chars")
        print("- Category Distribution:")
        for cat, count in sorted(stats["category_distribution"].items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"  - {cat}: {count}")
        print("- Domain Breakdown:")
        for domain, count in sorted(stats["domain_breakdown"].items(), key=lambda x: x[1], reverse=True):
            print(f"  - {domain}: {count}")
    else:
        logging.error("Failed to generate dataset statistics.")

    logging.info("Script execution completed.")
    if results['validation_issues'] or stats is None:
        logging.error("Script finished with warnings. Please check the logs.")
        exit(1)  # Exit with error code if any part failed
