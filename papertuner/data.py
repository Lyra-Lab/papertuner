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
        self.dataset = None

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
        
    def generate(self, client, ocr, upload=False, output_path="dataset", save_to_disk=True, min_text_length=500, resume=False):
        """
        Generate a dataset from papers and optionally upload to HuggingFace.

        Args:
            client (SourceClient): Client for fetching papers
            ocr (OCRBase): OCR implementation for text extraction
            upload (bool): Whether to upload the dataset to HuggingFace
            output_path (str): Path to save the dataset locally
            save_to_disk (bool): Whether to save the dataset to disk
            min_text_length (int): Minimum text length to include in dataset
            resume (bool): Whether to resume from existing dataset if available
        """
        # Create output directory if it doesn't exist
        output_dir = Path(output_path)
        if save_to_disk and not output_dir.exists():
            output_dir.mkdir(parents=True)
            
        existing_entries = []
        processed_ids = set()
        
        # Check for existing dataset if resume is enabled
        if resume and save_to_disk:
            json_path = output_dir / "dataset.json"
            arrow_dir = output_dir / "arrow"
            
            # Try to load existing dataset
            if json_path.exists():
                try:
                    with open(json_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        if 'data' in data and isinstance(data['data'], list):
                            existing_entries = data['data']
                            processed_ids = {entry['id'] for entry in existing_entries}
                            logger.info(f"Resuming from existing dataset with {len(existing_entries)} entries")
                except Exception as e:
                    logger.warning(f"Failed to load existing JSON dataset: {e}")
            
            # If JSON failed but arrow exists, try that instead
            elif arrow_dir.exists():
                try:
                    existing_dataset = Dataset.load_from_disk(str(arrow_dir))
                    existing_entries = existing_dataset.to_list()
                    processed_ids = {entry['id'] for entry in existing_entries}
                    logger.info(f"Resuming from existing Arrow dataset with {len(existing_entries)} entries")
                except Exception as e:
                    logger.warning(f"Failed to load existing Arrow dataset: {e}")

        # Fetch papers from the source
        papers, temp_dir = client.fetch_papers()
        
        try:
            # Extract text from papers with progress bar
            dataset_entries = list(existing_entries)  # Start with existing entries if resuming
            new_entries_count = 0
            
            for paper in tqdm(papers, desc="Processing papers"):
                # Skip already processed papers if resuming
                if paper['id'] in processed_ids:
                    logger.info(f"Skipping already processed paper: {paper['id']}")
                    continue
                    
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
                        processed_ids.add(paper['id'])
                        new_entries_count += 1

                        # Save individual paper data if requested
                        if save_to_disk:
                            paper_file = output_dir / f"{paper['id'].split('/')[-1].replace('.', '_')}.json"
                            with open(paper_file, 'w', encoding='utf-8') as f:
                                json.dump(entry, f, ensure_ascii=False, indent=2)
                            
                        # Periodically save the dataset to prevent complete data loss on failure
                        if save_to_disk and new_entries_count % 10 == 0:
                            self._save_intermediate_dataset(dataset_entries, output_dir)

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
                    # Save in JSON format
                    with open(output_dir / "dataset.json", 'w', encoding='utf-8') as f:
                        json.dump({"data": dataset_entries}, f, ensure_ascii=False, indent=2)
                    
                    # Also save in Arrow format for easier loading
                    dataset.save_to_disk(str(output_dir / "arrow"))
                    
                    logger.info(f"Dataset saved with {len(dataset_entries)} entries ({new_entries_count} new)")

            # Upload to HuggingFace if requested
            if upload and dataset_entries:
                self._upload_to_huggingface(dataset, output_dir)

            self.dataset = dataset
            return dataset
        finally:
            # Clean up the temporary directory after processing
            if temp_dir and os.path.exists(temp_dir):
                logger.info(f"Cleaning up temporary directory: {temp_dir}")
                shutil.rmtree(temp_dir, ignore_errors=True)
                
    def _save_intermediate_dataset(self, entries, output_dir):
        """
        Save dataset entries during processing to prevent data loss.
        
        Args:
            entries (list): List of dataset entries
            output_dir (Path): Output directory
        """
        try:
            # Save JSON format
            with open(output_dir / "dataset_intermediate.json", 'w', encoding='utf-8') as f:
                json.dump({"data": entries}, f, ensure_ascii=False, indent=2)
                
            # Rename to final file after successful write to prevent corruption
            os.replace(output_dir / "dataset_intermediate.json", output_dir / "dataset.json")
            
            # Also save in Arrow format
            temp_dataset = Dataset.from_list(entries)
            temp_dataset.save_to_disk(str(output_dir / "arrow_temp"))
            
            # Rename directory after successful save
            if os.path.exists(output_dir / "arrow"):
                shutil.rmtree(output_dir / "arrow")
            os.rename(output_dir / "arrow_temp", output_dir / "arrow")
            
            logger.info(f"Saved intermediate dataset with {len(entries)} entries")
        except Exception as e:
            logger.warning(f"Failed to save intermediate dataset: {e}")

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

    def load(self, path):
        """
        Load a dataset from disk.
        
        Args:
            path (str): Path to the dataset directory
        
        Returns:
            Dataset: Loaded dataset
        """
        path = Path(path)
        arrow_path = path / "arrow"
        
        if arrow_path.exists():
            try:
                self.dataset = Dataset.load_from_disk(str(arrow_path))
                logger.info(f"Loaded dataset from {arrow_path}")
                return self.dataset
            except Exception as e:
                logger.error(f"Failed to load dataset from Arrow format: {e}")
        
        # Fallback to JSON if Arrow fails
        json_path = path / "dataset.json"
        if json_path.exists():
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if 'data' in data and isinstance(data['data'], list):
                        self.dataset = Dataset.from_list(data['data'])
                        logger.info(f"Loaded dataset from {json_path}")
                        return self.dataset
            except Exception as e:
                logger.error(f"Failed to load dataset from JSON: {e}")
                
        raise FileNotFoundError(f"No dataset found at {path}")
        
    def describe(self):
        """
        Get a description of the dataset.
        
        Returns:
            str: Dataset description
        """
        if self.dataset is None:
            return "No dataset loaded"
            
        num_entries = len(self.dataset)
        if num_entries == 0:
            return f"Dataset: {self.name}\nEntries: 0"
            
        # Calculate total word count
        try:
            num_words = sum(len(entry["text"].split()) for entry in self.dataset)
        except:
            num_words = "unknown"
            
        return f"Dataset: {self.name}\nEntries: {num_entries}\nWords: {num_words}"
