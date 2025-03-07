import os
import logging
from typing import List, Dict, Any, Optional
from tqdm import tqdm

from .ocr import create_ocr
from .sources import create_source
from .formatters import create_formatter

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatasetPipeline:
    """Main pipeline for generating datasets from research papers."""

    def __init__(
        self,
        ocr_type: str = "ollama",
        source_type: str = "arxiv",
        formatter_type: str = "huggingface",
        ocr_kwargs: Optional[Dict[str, Any]] = None,
        source_kwargs: Optional[Dict[str, Any]] = None,
        formatter_kwargs: Optional[Dict[str, Any]] = None,
    ):
        # Initialize empty dicts if None
        ocr_kwargs = ocr_kwargs or {}
        source_kwargs = source_kwargs or {}
        formatter_kwargs = formatter_kwargs or {}

        try:
            self.ocr = create_ocr(ocr_type, **ocr_kwargs)
            self.source = create_source(source_type, **source_kwargs)
            self.formatter = create_formatter(formatter_type, **formatter_kwargs)
        except Exception as e:
            logger.error(f"Error initializing pipeline: {e}")
            raise

    def process_paper(self, paper: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Process a single paper.

        Args:
            paper: Paper metadata dictionary

        Returns:
            Formatted entry or None if processing failed
        """
        paper_id = paper["id"]
        try:
            # Use standard URL-based processing
            pdf_url = paper["pdf_url"]
            logger.info(f"Processing paper {paper_id} from URL: {pdf_url}")
            text = self.ocr.process_url(pdf_url)

            if not text or len(text.strip()) < 100:
                logger.warning(f"Extracted text for paper {paper_id} is too short: {len(text) if text else 0} characters")
                return None

            # Format the entry
            entry = self.formatter.format_entry(paper, text)
            return entry

        except Exception as e:
            logger.error(f"Error processing paper {paper_id}: {e}")
            return None

    def generate(
        self,
        query: str,
        output_path: str,
        max_papers: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Generate a dataset from research papers.

        Args:
            query: Search query for papers
            output_path: Path to save the dataset to
            max_papers: Maximum number of papers to process

        Returns:
            List of processed entries
        """
        # Create output directory if it's a directory path
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Search for papers
        logger.info(f"Searching for papers matching: {query}")
        papers = self.source.search(query, max_results=max_papers)
        logger.info(f"Found {len(papers)} papers")

        # Process each paper
        entries = []
        for paper in tqdm(papers, desc="Processing papers"):
            entry = self.process_paper(paper)
            if entry:
                entries.append(entry)

        # Save the final dataset
        logger.info(f"Saving dataset to {output_path}")
        if entries:
            self.formatter.save(entries, output_path)
            logger.info(f"Dataset saved with {len(entries)} entries")
        else:
            logger.warning("No entries to save")

        return entries
