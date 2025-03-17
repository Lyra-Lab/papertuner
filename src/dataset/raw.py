import os
import json
import time
import fitz  # PyMuPDF
import arxiv
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from tqdm import tqdm  # Fixed typo
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, TimeoutError

class ArxivRawDataset:
    def __init__(self, output_dir="raw_dataset", max_papers=100):
        self.output_dir = Path(output_dir)
        self.max_papers = max_papers
        self.queries = [
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
        ]
        self.categories = [
            "cs.LG",  # Machine Learning
            "cs.AI",  # Artificial Intelligence
            "cs.CL",  # Computation and Language (NLP)
            "cs.CV",  # Computer Vision and Pattern Recognition
            "cs.NE",  # Neural and Evolutionary Computing
            "stat.ML",  # Machine Learning (Statistics)
            "q-bio.QM"  # Quantitative Biology: Methods (for bioinformatics and computational biology)
        ]
        (self.output_dir / "papers").mkdir(parents=True, exist_ok=True)
        
        # Configure retry strategy
        self.session = requests.Session()
        retries = Retry(total=5, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
        self.session.mount('https://', HTTPAdapter(max_retries=retries))
        self.session.mount('http://', HTTPAdapter(max_retries=retries))

    def _download_pdf(self, url, paper_id):
        """Download PDF with retries and timeout"""
        temp_path = self.output_dir / f"temp_{paper_id.split('/')[-1]}.pdf"
        
        try:
            response = self.session.get(url, stream=True, timeout=10)
            response.raise_for_status()
            
            with open(temp_path, 'wb') as f:
                for chunk in response.iter_content(8192):
                    f.write(chunk)
            return temp_path
                    
        except requests.exceptions.RequestException as e:
            print(f"Download failed for {url}: {str(e)}")
            return None

    def _extract_text(self, pdf_path):
        """Extract text with timeout and error handling"""
        if not os.path.exists(pdf_path):
            return ""
            
        def _extract():
            try:
                doc = fitz.open(pdf_path)
                return " ".join([page.get_text() for page in doc])
            except Exception as e:
                print(f"Extraction failed for {pdf_path}: {str(e)}")
                return ""
        
        # Run extraction in thread with timeout
        try:
            with ThreadPoolExecutor() as executor:
                future = executor.submit(_extract)
                return future.result(timeout=30)
        except TimeoutError:
            print(f"Timeout extracting text from {pdf_path}")
            return ""
        except Exception as e:
            print(f"Unexpected error extracting text: {str(e)}")
            return ""

    def _save_paper(self, data, filename):
        """Atomic file write with error handling"""
        temp_file = self.output_dir / "papers" / f".tmp_{filename}"
        target_file = self.output_dir / "papers" / filename
        
        try:
            with open(temp_file, 'w') as f:
                json.dump(data, f, indent=2)
            temp_file.rename(target_file)
        except Exception as e:
            print(f"Failed to save {target_file}: {str(e)}")
            try:
                os.remove(temp_file)
            except OSError:
                pass

    def run(self):
        """Main pipeline with error handling"""
        search = arxiv.Search(
            query=" OR ".join(self.queries),
            max_results=self.max_papers,
            sort_by=arxiv.SortCriterion.Relevance
        )

        manifest = []
        
        try:
            for i, result in enumerate(tqdm(search.results()), 1):
                # Rate limiting with jitter
                if i % 5 == 0:
                    time.sleep(1 + 0.5 * (i % 3))  # Add some randomness
                
                try:
                    # Download PDF
                    pdf_path = self._download_pdf(result.pdf_url, result.entry_id)
                    if not pdf_path:
                        continue

                    # Extract text
                    full_text = self._extract_text(pdf_path)
                    
                    # Build paper record
                    paper_data = {
                        "id": result.entry_id,
                        "title": result.title,
                        "authors": [str(a) for a in result.authors],
                        "abstract": result.summary,
                        "full_text": full_text,
                        "published": result.published.isoformat(),
                        "categories": result.categories,
                        "pdf_url": result.pdf_url
                    }

                    # Save paper
                    filename = f"paper_{result.entry_id.split('/')[-1]}.json"
                    self._save_paper(paper_data, filename)
                    manifest.append({
                        "id": result.entry_id,
                        "filename": filename,
                        "title": result.title
                    })

                finally:
                    # Cleanup temp PDF
                    try:
                        if pdf_path and os.path.exists(pdf_path):
                            os.remove(pdf_path)
                    except Exception as e:
                        print(f"Failed to delete {pdf_path}: {str(e)}")

        except (arxiv.HTTPError, requests.RequestException) as e:
            print(f"Search failed: {str(e)}")
        except Exception as e:
            print(f"Unexpected error: {str(e)}")
        finally:
            # Save manifest even if interrupted
            try:
                with open(self.output_dir / "manifest.json", 'w') as f:
                    json.dump(manifest, f, indent=2)
            except Exception as e:
                print(f"Failed to save manifest: {str(e)}")

if __name__ == "__main__":
    dataset = ArxivRawDataset(
        output_dir="../../data/raw_dataset",
        max_papers=10000  # Start small for testing
    )
    dataset.run()
    print(f"Dataset created at {dataset.output_dir}")
