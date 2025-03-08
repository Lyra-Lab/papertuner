import os
import logging
from typing import List, Dict, Any, Optional
from tqdm import tqdm

from .ocr import create_ocr
from .sources import create_source
from .formatters import create_formatter
from .logging_config import setup_logging

# Get logger
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
            # Log detailed configuration to file only
            logger.debug(f"Initializing pipeline: OCR={ocr_type}, Source={source_type}, Formatter={formatter_type}")
            logger.debug(f"OCR kwargs: {ocr_kwargs}")
            logger.debug(f"Source kwargs: {source_kwargs}")
            logger.debug(f"Formatter kwargs: {formatter_kwargs}")

            # Initialize components
            self.source = create_source(source_type, **source_kwargs)
            self.ocr = create_ocr(ocr_type, **ocr_kwargs)
            self.formatter = create_formatter(formatter_type, **formatter_kwargs)

            # Only log success message to console
            logger.info("Pipeline initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing pipeline: {e}")
            raise

    def process_paper(self, paper: Dict[str, Any], position: int, total: int) -> Optional[Dict[str, Any]]:
        """
        Process a single paper.

        Args:
            paper: Paper metadata dictionary
            position: Current paper position for progress display
            total: Total number of papers

        Returns:
            Formatted entry or None if processing failed
        """
        paper_id = paper["id"]
        title = paper.get("title", "").strip()
        # Truncate title if too long
        short_title = (title[:40] + "...") if len(title) > 40 else title

        try:
            # Log processing start
            logger.debug(f"Processing paper {paper_id}: {title}")

            # Set up a progress description
            progress_desc = f"Paper {position}/{total}: {short_title}"

            # Use standard URL-based processing
            pdf_url = paper["pdf_url"]

            # Process the paper - progress bars are handled inside the OCR module
            with tqdm(total=1, desc=progress_desc, leave=True) as paper_pbar:
                text = self.ocr.process_url(pdf_url)

                if not text or len(text.strip()) < 100:
                    logger.warning(f"Paper {paper_id}: Insufficient text extracted")
                    paper_pbar.set_description(f"❌ {progress_desc} (failed)")
                    return None

                # Format the entry
                entry = self.formatter.format_entry(paper, text)
                paper_pbar.update(1)
                paper_pbar.set_description(f"✓ {progress_desc}")

            return entry

        except Exception as e:
            logger.warning(f"Paper {paper_id}: Failed to process")
            logger.debug(f"Error details: {str(e)}")
            return None

    def generate(
        self,
        query: str,
        output_path: str,
        max_papers: int = 10,
        verbose: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Generate a dataset from research papers.

        Args:
            query: Search query for papers
            output_path: Path to save the dataset to
            max_papers: Maximum number of papers to process
            verbose: Whether to show detailed progress

        Returns:
            List of processed entries
        """
        # Set up logging based on verbosity
        log_file = setup_logging(verbose)

        # Create output directory if it's a directory path
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Search for papers with progress indication
        logger.info(f"🔍 Searching for papers: \"{query}\"")

        # Use progress spinner for search
        with tqdm(total=None, desc="Searching...", leave=True) as search_pbar:
            papers = self.source.search(query, max_results=max_papers)
            search_pbar.set_description(f"Found {len(papers)} papers matching \"{query}\"")

        # Process each paper
        entries = []

        # Process papers - we don't need a separate tqdm here since we show
        # progress inside the process_paper method
        for i, paper in enumerate(papers):
            entry = self.process_paper(paper, i+1, len(papers))
            if entry:
                entries.append(entry)

        # Show summary
        success_count = len(entries)
        if success_count > 0:
            logger.info(f"📊 Successfully processed {success_count} of {len(papers)} papers ({success_count/len(papers):.0%})")
        else:
            logger.warning("⚠️ Failed to process any papers")

        # Save the final dataset with progress indication
        if entries:
            logger.info(f"💾 Saving dataset to {output_path}")

            # Use progress spinner for saving
            with tqdm(total=None, desc="Saving dataset...", leave=True) as save_pbar:
                self.formatter.save(entries, output_path)
                save_pbar.set_description(f"✓ Dataset saved with {len(entries)} entries")
        else:
            logger.warning("⚠️ No entries to save")

        logger.info(f"📝 Detailed log available at: {log_file}")

        return entries
