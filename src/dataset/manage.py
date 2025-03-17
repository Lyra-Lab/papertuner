# dataset/manage.py
import os
import json
import logging
from pathlib import Path
from collections import defaultdict
from datasets import Dataset, concatenate_datasets
from huggingface_hub import HfApi, create_repo, login

class DatasetManager:
    def __init__(self, processed_dir="processed_dataset"):
        self.processed_dir = Path(processed_dir)
        self.hf_token = os.getenv("HF_TOKEN")
        self.repo_id = os.getenv("HF_REPO_ID", "your-username/research-assistant-dataset")
        self.api = HfApi(token=self.hf_token) if self.hf_token else None

    def _load_processed_files(self):
        """Load all processed JSON files"""
        processed_files = list(self.processed_dir.glob("processed_*.json"))
        if not processed_files:
            raise FileNotFoundError(f"No processed files found in {self.processed_dir}")
        return processed_files

    def _build_dataset(self, files):
        """Create Hugging Face Dataset from processed files"""
        qa_pairs = []
        metadata = defaultdict(list)
        
        for file in files:
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
        
        return Dataset.from_list(qa_pairs), metadata

    def validate_dataset(self):
        """Perform quality checks on the dataset"""
        files = self._load_processed_files()
        valid_count = 0
        issues = []

        for file in files:
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

        return {
            "total_files": len(files),
            "valid_entries": valid_count,
            "validation_issues": issues
        }

    def upload_to_hf(self, split_ratios=(0.8, 0.1, 0.1)):
        """Upload dataset to Hugging Face Hub with train/val/test splits"""
        if not self.hf_token:
            raise ValueError("HF_TOKEN not set in environment variables")
        
        files = self._load_processed_files()
        dataset, metadata = self._build_dataset(files)
        
        # Create dataset splits
        split = dataset.train_test_split(test_size=split_ratios[1] + split_ratios[2])
        test_val = split["test"].train_test_split(test_size=split_ratios[2]/sum(split_ratios[1:]))
        
        final_dataset = concatenate_datasets({
            "train": split["train"],
            "validation": test_val["train"],
            "test": test_val["test"]
        })

        # Create dataset card
        card_content = f"""\
# Research Assistant QA Dataset

## Overview
- Contains {len(dataset)} validated question-answer pairs
- Derived from {len(files)} research papers
- Categories: {', '.join(set(sum(metadata['categories'], [])))}

## Fields
- `question`: Research methodology question
- `answer`: Detailed methodology answer
- `paper_id`: Source paper identifier
- `categories`: arXiv categories
"""

        # Upload to Hub
        login(token=self.hf_token)
        create_repo(self.repo_id, repo_type="dataset", exist_ok=True)
        
        final_dataset.push_to_hub(
            self.repo_id,
            commit_message=f"Add dataset v1.0 with {len(dataset)} entries",
            dataset_card=card_content
        )

        print(f"Dataset uploaded to https://huggingface.co/datasets/{self.repo_id}")

    def generate_statistics(self):
        """Create dataset analytics report"""
        files = self._load_processed_files()
        stats = {
            "total_papers": len(files),
            "avg_answer_length": 0,
            "category_distribution": defaultdict(int),
            "domain_breakdown": defaultdict(int)
        }

        total_chars = 0
        for file in files:
            with open(file, "r") as f:
                data = json.load(f)
                stats["category_distribution"][data["metadata"]["categories"][0]] += 1
                total_chars += len(data["qa"]["answer"])
                
                # Extract domain from categories
                domain = data["metadata"]["categories"][0].split(".")[0]
                stats["domain_breakdown"][domain] += 1

        stats["avg_answer_length"] = total_chars / len(files) if files else 0
        return stats

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Manage research dataset")
    parser.add_argument("--validate", action="store_true", help="Run validation checks")
    parser.add_argument("--upload", action="store_true", help="Upload to Hugging Face")
    parser.add_argument("--stats", action="store_true", help="Generate statistics")
    
    args = parser.parse_args()
    manager = DatasetManager()

    if args.validate:
        results = manager.validate_dataset()
        print(f"Validation Results:\n- Total entries: {results['total_files']}"
              f"\n- Valid QA pairs: {results['valid_entries']}"
              f"\n- Issues found: {len(results['validation_issues'])}")
        
    if args.upload:
        manager.upload_to_hf()
        
    if args.stats:
        stats = manager.generate_statistics()
        print("Dataset Statistics:")
        print(f"- Average answer length: {stats['avg_answer_length']:.1f} chars")
        print("- Category Distribution:")
        for cat, count in stats["category_distribution"].items():
            print(f"  - {cat}: {count}")
            
        print("- Domain Breakdown:")
        for domain, count in stats["domain_breakdown"].items():
            print(f"  - {domain}: {count}")
