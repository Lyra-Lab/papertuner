import os
from typing import List, Dict, Any, Optional
from tqdm import tqdm

from .ocr.factory import create_ocr
from .sources.factory import create_source
from .formatters.factory import create_formatter


class DatasetPipeline:
    """Main pipeline for generating datasets from research papers."""
    
    def __init__(
        self,
        ocr_type: str = "mistral",
        source_type: str = "arxiv",
        formatter_type: str = "jsonl",
        **kwargs
    ):
        """
        Initialize the dataset pipeline.
        
        Args:
            ocr_type: Type of OCR to use
            source_type: Type of source to use
            formatter_type: Type of formatter to use
            **kwargs: Additional arguments to pass to the components
        """
        self.ocr = create_ocr(ocr_type, **kwargs.get("ocr_kwargs", {}))
        self.source = create_source(source_type, **kwargs.get("source_kwargs", {}))
        self.formatter = create_formatter(formatter_type, **kwargs.get("formatter_kwargs", {}))
    
    def generate(
        self,
        query: str,
        output_path: str,
        max_papers: int = 10,
    ) -> None:
        """
        Generate a dataset from research papers.
        
        Args:
            query: Search query for papers
            output_path: Path to save the dataset to
            max_papers: Maximum number of papers to process
        """
        # Search for papers
        print(f"Searching for papers matching: {query}")
        papers = self.source.search(query, max_results=max_papers)
        print(f"Found {len(papers)} papers")
        
        # Process each paper
        entries = []
        for paper in tqdm(papers, desc="Processing papers"):
            try:
                # Get the PDF URL
                pdf_url = paper["pdf_url"]
                
                # Extract text using OCR
                print(f"Extracting text from: {paper['title']}")
                text = self.ocr.process_url(pdf_url)
                
                # Format the entry
                entry = self.formatter.format_entry(paper, text)
                entries.append(entry)
                
            except Exception as e:
                print(f"Error processing paper {paper['id']}: {e}")
        
        # Save the dataset
        print(f"Saving dataset to {output_path}")
        self.formatter.save(entries, output_path)
        print(f"Dataset saved with {len(entries)} entries") 