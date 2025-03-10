# papertuner/data.py
"""
Dataset management for creating and uploading datasets.
"""

import os
import json
import logging
import shutil
from pathlib import Path
from datasets import Dataset, load_dataset
from huggingface_hub import HfApi, login
from tqdm import tqdm

logger = logging.getLogger(__name__)

class HGDataset:
    """Class for creating and managing datasets for HuggingFace."""

    def __init__(self, name, remote_username, hg_token=None):
        """
        Initialize HGDataset.

        Args:
            name (str): Name of the dataset
            remote_username (str): HuggingFace username
            hg_token (str, optional): HuggingFace token
        """
        self.name = name
        self.remote_username = remote_username
        self.hg_token = hg_token

        # Normalize the dataset name for use in repository
        self.repo_name = self._normalize_name(name)

        if hg_token:
            login(token=hg_token)

    def _normalize_name(self, name):
        """
        Normalize dataset name for HuggingFace repository.

        Args:
            name (str): Dataset name

        Returns:
            str: Normalized name
        """
        # Convert spaces to underscores and lowercase
        return name.replace(" ", "_").lower()

    def generate(self, client, ocr, upload=False, output_path="dataset", save_to_disk=True, min_text_length=500):
        """
        Generate a dataset from papers and optionally upload to HuggingFace.

        Args:
            client (SourceClient): Client for fetching papers
            ocr (OCRBase): OCR implementation for text extraction
            upload (bool): Whether to upload the dataset to HuggingFace
            output_path (str): Path to save the dataset locally
            save_to_disk (bool): Whether to save the dataset to disk
            min_text_length (int): Minimum text length to include in dataset
        """
        # Create output directory if it doesn't exist
        output_dir = Path(output_path)
        if save_to_disk and not output_dir.exists():
            output_dir.mkdir(parents=True)

        # Fetch papers from the source
        papers, temp_dir = client.fetch_papers()
        
        try:
            # Extract text from papers with progress bar
            dataset_entries = []
            for paper in tqdm(papers, desc="Processing papers"):
                try:
                    pdf_path = paper.get("pdf_path")
                    if pdf_path and os.path.exists(pdf_path):
                        # Extract text using OCR
                        text = ocr.extract_text(pdf_path)
                        
                        # Skip entries with insufficient text
                        if len(text) < min_text_length:
                            logger.warning(f"Paper {paper['id']} has insufficient text ({len(text)} chars). Skipping.")
                            continue

                        # Create dataset entry
                        entry = {
                            "id": paper["id"],
                            "title": paper["title"],
                            "authors": paper["authors"],
                            "abstract": paper["abstract"],
                            "text": text,
                            "categories": paper["categories"],
                            "published": paper["published"],
                            "updated": paper["updated"]
                        }

                        dataset_entries.append(entry)

                        # Save individual paper data if requested
                        if save_to_disk:
                            paper_file = output_dir / f"{paper['id'].split('/')[-1].replace('.', '_')}.json"
                            with open(paper_file, 'w', encoding='utf-8') as f:
                                json.dump(entry, f, ensure_ascii=False, indent=2)

                except Exception as e:
                    logger.error(f"Error processing paper {paper.get('id')}: {e}")

            if not dataset_entries:
                logger.warning("No papers were successfully processed. Dataset is empty.")
                # Create an empty dataset with the expected schema
                empty_schema = {
                    "id": "",
                    "title": "",
                    "authors": [],
                    "abstract": "",
                    "text": "",
                    "categories": [],
                    "published": "",
                    "updated": ""
                }
                dataset = Dataset.from_dict({k: [v] for k, v in empty_schema.items()})
                if save_to_disk:
                    with open(output_dir / "dataset.json", 'w', encoding='utf-8') as f:
                        json.dump({"data": []}, f, ensure_ascii=False, indent=2)
            else:
                # Create the dataset
                dataset = Dataset.from_list(dataset_entries)

                # Save the complete dataset if requested
                if save_to_disk:
                    dataset.save_to_json(str(output_dir / "dataset.json"))
                    
                    # Also save in Arrow format for easier loading
                    dataset.save_to_disk(output_dir / "arrow")

            # Upload to HuggingFace if requested
            if upload and dataset_entries:
                self._upload_to_huggingface(dataset, output_dir)

            self.dataset = dataset
        finally:
            # Clean up the temporary directory after processing
            if temp_dir and os.path.exists(temp_dir):
                logger.info(f"Cleaning up temporary directory: {temp_dir}")
                shutil.rmtree(temp_dir, ignore_errors=True)

    def _upload_to_huggingface(self, dataset, output_dir):
        """
        Upload dataset to HuggingFace.

        Args:
            dataset (Dataset): The dataset to upload
            output_dir (Path): Path where the dataset is saved locally
        """
        if not self.hg_token:
            raise ValueError("HuggingFace token is required for uploading")

        repo_id = f"{self.remote_username}/{self.repo_name}"

        try:
            # Create the repository if it doesn't exist
            api = HfApi(token=self.hg_token)

            # Check if the repo exists
            try:
                api.repo_info(repo_id=repo_id, repo_type="dataset")
                logger.info(f"Repository {repo_id} already exists")
            except Exception:
                # Create the repository
                api.create_repo(repo_id=repo_id, repo_type="dataset", private=False)
                logger.info(f"Created repository {repo_id}")

            # Push the dataset to the hub
            dataset.push_to_hub(
                repo_id=repo_id,
                token=self.hg_token
            )

            logger.info(f"Dataset uploaded to {repo_id}")

        except Exception as e:
            logger.error(f"Error uploading dataset to HuggingFace: {e}")
            raise

    def describe(self):
        num_files = len(self.dataset)
        num_entries = len(self.dataset[0])
        num_words = sum(len(entry["text"].split()) for entry in self.dataset)
        return f"Dataset: {self.name}\nFiles: {num_files}\nEntries: {num_entries}\nWords: {num_words}"
