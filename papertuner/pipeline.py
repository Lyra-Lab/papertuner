import os
import time
from typing import List, Dict, Any, Optional, Tuple
from tqdm import tqdm
import concurrent.futures
import logging

from .ocr.factory import create_ocr
from .sources.factory import create_source
from .formatters.factory import create_formatter
from .ocr.arxiv_integrated import ArxivIntegratedOCR

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatasetPipeline:
    """Main pipeline for generating datasets from research papers."""

    def __init__(
        self,
        ocr_type: str = "mistral",
        source_type: str = "arxiv",
        formatter_type: str = "jsonl",
        api_key: Optional[str] = None,
        ocr_kwargs: Optional[Dict[str, Any]] = None,
        source_kwargs: Optional[Dict[str, Any]] = None,
        formatter_kwargs: Optional[Dict[str, Any]] = None,
        parallel: bool = False,
        max_workers: int = 4,
        retry_failed: bool = True,
        max_retries: int = 2,
        **kwargs  # For backward compatibility
    ):
        # Initialize empty dicts if None
        ocr_kwargs = ocr_kwargs or {}
        source_kwargs = source_kwargs or {}
        formatter_kwargs = formatter_kwargs or {}

        # Only add API key to ocr_kwargs if provided AND using a service that needs it
        if api_key:
            if ocr_type == "mistral":
                ocr_kwargs["api_key"] = api_key
            elif ocr_type == "gemini":
                ocr_kwargs["api_key"] = api_key
            # Don't add API key for transformers OCR

        try:
            self.ocr = create_ocr(ocr_type, **ocr_kwargs)
            self.source = create_source(source_type, **source_kwargs)
            self.formatter = create_formatter(formatter_type, **formatter_kwargs)
            self.parallel = parallel
            self.max_workers = max_workers
        except Exception as e:
            logger.error(f"Error initializing pipeline: {e}")
            raise

    def process_paper(self, paper: Dict[str, Any], retry_count: int = 0) -> Optional[Dict[str, Any]]:
        """
        Process a single paper.

        Args:
            paper: Paper metadata dictionary
            retry_count: Current retry attempt

        Returns:
            Formatted entry or None if processing failed
        """
        paper_id = paper["id"]
        try:
            # Check if we can use direct ArXiv integration
            if isinstance(self.ocr, ArxivIntegratedOCR) and self.source.__class__.__name__ == "ArxivSource":
                # Use direct ArXiv integration
                logger.info(f"Processing paper {paper_id} using direct ArXiv integration")
                text = self.ocr.process_arxiv_id(paper_id, source=self.source)
            else:
                # Use standard URL-based processing
                pdf_url = paper["pdf_url"]
                logger.info(f"Processing paper {paper_id} from URL: {pdf_url}")
                text = self.ocr.process_url(pdf_url)

            if not text or len(text.strip()) < 100:
                logger.warning(f"Extracted text for paper {paper_id} is too short: {len(text) if text else 0} characters")
                if self.retry_failed and retry_count < self.max_retries:
                    logger.info(f"Retrying paper {paper_id} (attempt {retry_count + 1}/{self.max_retries})")
                    time.sleep(2)  # Add a small delay before retrying
                    return self.process_paper(paper, retry_count + 1)
                return None

            # Format the entry
            entry = self.formatter.format_entry(paper, text)
            return entry

        except Exception as e:
            logger.error(f"Error processing paper {paper_id}: {e}")

            # Retry if enabled and under max retries
            if self.retry_failed and retry_count < self.max_retries:
                logger.info(f"Retrying paper {paper_id} (attempt {retry_count + 1}/{self.max_retries})")
                time.sleep(2)  # Add a small delay before retrying
                return self.process_paper(paper, retry_count + 1)

            return None

    def generate(
        self,
        query: str,
        output_path: str,
        max_papers: int = 10,
        fail_on_empty: bool = False,
        save_intermediate: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Generate a dataset from research papers.

        Args:
            query: Search query for papers
            output_path: Path to save the dataset to
            max_papers: Maximum number of papers to process
            fail_on_empty: Whether to raise an exception if no papers were successfully processed
            save_intermediate: Whether to save intermediate results after each successful processing

        Returns:
            List of processed entries

        Raises:
            ValueError: If fail_on_empty is True and no papers were successfully processed
        """
        # Create output directory if it's a directory path
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Intermediate results path
        intermediate_path = f"{output_path}.intermediate.jsonl"

        # Search for papers
        logger.info(f"Searching for papers matching: {query}")
        papers = self.source.search(query, max_results=max_papers)
        logger.info(f"Found {len(papers)} papers")

        # Process each paper
        entries = []

        if self.parallel and len(papers) > 1:
            # Process papers in parallel
            logger.info(f"Processing papers in parallel with {self.max_workers} workers")
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # Submit all papers for processing
                future_to_paper = {executor.submit(self.process_paper, paper): paper for paper in papers}

                # Process results as they complete
                for future in tqdm(concurrent.futures.as_completed(future_to_paper),
                                  total=len(papers),
                                  desc="Processing papers"):
                    paper = future_to_paper[future]
                    try:
                        entry = future.result()
                        if entry:
                            entries.append(entry)
                            # Save intermediate results
                            if save_intermediate and entries:
                                try:
                                    self.formatter.save(entries, intermediate_path)
                                except Exception as e:
                                    logger.warning(f"Failed to save intermediate results: {e}")
                    except Exception as e:
                        logger.error(f"Error processing paper {paper['id']}: {e}")
        else:
            # Process papers sequentially
            for paper in tqdm(papers, desc="Processing papers"):
                entry = self.process_paper(paper)
                if entry:
                    entries.append(entry)
                    # Save intermediate results
                    if save_intermediate and entries:
                        try:
                            self.formatter.save(entries, intermediate_path)
                        except Exception as e:
                            logger.warning(f"Failed to save intermediate results: {e}")

        # Check if we have any successful entries
        if not entries:
            error_msg = "No papers were successfully processed"
            logger.error(error_msg)
            if fail_on_empty:
                raise ValueError(error_msg)

        # Save the final dataset
        logger.info(f"Saving dataset to {output_path}")
        if entries:
            self.formatter.save(entries, output_path)
            logger.info(f"Dataset saved with {len(entries)} entries")

            # Remove intermediate file if it exists
            if save_intermediate and os.path.exists(intermediate_path):
                try:
                    os.remove(intermediate_path)
                except Exception:
                    pass
        else:
            logger.warning("No entries to save")

        return entries
