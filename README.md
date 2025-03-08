# PaperTuner

A streamlined tool for creating fine-tuning datasets from scientific papers, focusing on extracting text with Ollama and organizing data in HuggingFace format.

## Overview

PaperTuner helps researchers and ML engineers create high-quality training datasets from scientific literature. It:

1. Searches for relevant papers on ArXiv using customizable queries
2. Extracts full text from PDFs using Ollama's vision-capable models
3. Formats the data as HuggingFace datasets for direct use in fine-tuning pipelines

Perfect for creating domain-specific datasets to fine-tune LLMs on scientific literature, especially for specialized fields where quality data is scarce.

## Features

- **ArXiv Integration**: Search papers by keyword, category, date range, and more
- **Ollama OCR**: Extract text from PDFs using locally-running vision models
- **HuggingFace Format**: Output datasets ready for use with the HuggingFace ecosystem
- **Simple Pipeline**: All steps integrated in one cohesive workflow

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/papertuner.git
cd papertuner

# Install the package using uv
uv pip install -e .
```

## Requirements

- Python 3.8+
- [Ollama](https://ollama.ai/) running locally or on a remote server
- A vision-capable model in Ollama (e.g., llava:latest)
- pdf2image dependencies (poppler)
  - Ubuntu/Debian: `sudo apt-get install poppler-utils`
  - macOS: `brew install poppler`
  - Windows: [poppler for Windows](https://github.com/oschwartz10612/poppler-windows/releases/) - Ensure `poppler-utils/bin` is in your PATH

## Getting Started

This section guides you to quickly start using PaperTuner and run the example script to generate your first dataset of machine learning papers.

### Prerequisites

Before you begin, make sure you have these set up:

*   **Python:**  Python 3.8 or later.
*   **Ollama:**
    *   **Install Ollama:** Follow the [Ollama installation guide](https://ollama.com/download) to install and run Ollama.
    *   **Run Ollama Server:** Ensure Ollama is running in the background. The example script assumes it's accessible at `http://localhost:11434`. If your Ollama server is running elsewhere, you'll need to adjust `ollama_host` in the `example.py` script.
    *   **Download Ollama Vision Model:** The example uses `granite3.2-vision:latest`. Download it by running:
        ```bash
        ollama pull granite3.2-vision:latest
        ```
        *You can also use other vision models like `llava:latest`.*
*   **Hugging Face Account & Token (Optional):** If you plan to push your dataset to the Hugging Face Hub, you'll need an account and an API token with write access. Get your token from [Hugging Face Settings](https://huggingface.co/settings/tokens).

### Installation Steps

1.  **Clone PaperTuner:**
    ```bash
    git clone <YOUR_REPOSITORY_URL>  # Replace with your repository URL
    cd papertuner
    ```

2.  **Install PaperTuner Package:**
    ```bash
    uv pip install -e .  # Recommended: fast installation using uv
    # or, alternatively use pip:
    # pip install -e .
    ```

### Running the Example Script

1.  **Navigate to `papertuner` directory:** Open your terminal in the cloned `papertuner` directory.

2.  **Set Hugging Face Token (Optional):** If pushing to the Hub, set your Hugging Face API token as an environment variable:
    ```bash
    export HF_TOKEN=<YOUR_HUGGING_FACE_API_TOKEN>  # Replace with your actual token
    ```
    *Note: To skip pushing to the Hub for this example, either don't set `HF_TOKEN` or set `push_to_hub=False` in `example.py`.*

3.  **Execute the Script:**
    ```bash
    python papertuner/example.py
    ```

### Understanding the Example Script

The `example.py` script performs these actions:

*   **ArXiv Paper Search:**  It searches ArXiv for machine learning papers published in 2023 using a predefined query.
*   **Text Extraction with Ollama:** It downloads PDFs of the found papers and uses the `granite3.2-vision:latest` Ollama model to extract text.
*   **Hugging Face Dataset Creation:**  It structures the extracted text and paper metadata into a Hugging Face Dataset format.
*   **Dataset Saving and Hub Upload (Optional):** It saves the dataset locally in the `machine_learning_dataset` folder and optionally pushes it to the Hugging Face Hub under `densud2/machine-learning-papers` (you can customize this in `example.py`).

### What to Expect After Running

After running `example.py`, you should observe:

*   **Terminal Logs:**  Detailed output showing the script's progress, including the number of papers processed and any messages.
*   **Local Dataset Folder:** A folder named `machine_learning_dataset` created in your current directory, containing the Hugging Face Dataset files.
*   **(Optional) Hugging Face Hub Dataset:** If you set `HF_TOKEN` and `push_to_hub=True`, a new dataset named `machine-learning-papers` will be available on your Hugging Face account (or under `densud2` if you use the default settings).

### Next Steps After Running the Example

*   **Explore `example.py` Code:** Open `papertuner/example.py` to understand the `DatasetPipeline` configuration and customization options.
*   **Experiment with Parameters:** Modify `example.py` to test different configurations:
    *   **ArXiv Query Modification:** Change the ArXiv query to search for papers in different research domains.
    *   **Adjust `max_papers`:**  Process more or fewer papers.
    *   **Ollama Model Selection:** Try different Ollama vision models or OCR parameters.
    *   **Customize Output:** Change the local dataset save path and the Hugging Face Hub dataset name/location.

This "Getting Started" guide is designed to help you quickly run the example script and grasp the fundamental workflow of PaperTuner. Adapt and explore further to tailor PaperTuner to your specific dataset needs!

### Using the Generated Dataset

Once you have generated a dataset, you can easily load and use it with the HuggingFace `datasets` library:

```python
from datasets import load_from_disk

# Load your locally saved dataset
dataset = load_from_disk("machine_learning_dataset")

# Explore the dataset structure and content
print(dataset)
print(dataset.column_names)
print(dataset[0]["title"])
print(dataset[0]["summary"])
```

### Installing Dependencies with uv

If you prefer using [uv](https://github.com/astral-sh/uv) for managing Python dependencies:

```bash
# Install core dependencies
uv pip install arxiv datasets ollama pdf2image pillow requests tqdm

# Install development dependencies
uv pip install --dev pytest black isort mypy
```

## Configuration Options

*Below are example configurations. Refer to the `DatasetPipeline` documentation for a comprehensive list of available options.*

### OCR Configuration (`ocr_kwargs`)

```python
ocr_kwargs={
    "model_name": "llava:latest",    # Ollama vision model to use (e.g., granite3.2-vision:latest)
    "ollama_host": "http://localhost:11434", # Address of your Ollama server if not local
    "temperature": 0.1,             # Lower values for more deterministic OCR results
    "max_tokens": 8192,              # Maximum tokens for OCR output per paper (adjust based on paper length)
    "max_retries": 3,              # Number of retry attempts for failed OCR extractions
}
```

### Source Configuration (`source_kwargs`)

```python
source_kwargs={
    "max_results": 100,          # Maximum number of papers to retrieve from ArXiv
    "sort_by": "relevance",      # Sort order for ArXiv search results (e.g., relevance, date)
    # ... other ArXiv API parameters can be configured here
}
```

### Formatter Configuration (`formatter_kwargs`)

```python
formatter_kwargs={
    "save_locally": True,
    "push_to_hub": False,        # Set to True to push the dataset to Hugging Face Hub
    "hub_dataset_name": "username/dataset-name", #  Hugging Face dataset repository name (required if `push_to_hub=True`)
    "hub_token": "your_token",   # Hugging Face API token (or set HF_TOKEN environment variable)
}
```

## ArXiv Search Query Tips

- **Categories**:  `cat:cs.AI` for AI papers, `cat:physics.comp-ph` for computational physics, etc.
- **Title/Abstract**: `ti:keyword` to search in titles, `abs:keyword` to search in abstracts.
- **Date Range**: `submittedDate:[YYYYMMDD TO YYYYMMDD]` for papers within a specific date range (e.g., `submittedDate:[20230101 TO 20231231]`).
- **Authors**: `au:lastname` to filter by author's last name.
- **Combining Queries**: Use `AND`, `OR`, and `NOT` for complex search criteria. Refer to the [ArXiv API documentation](https://arxiv.org/help/api/query_syntax) for detailed syntax.

## Limitations

- The quality of PDF text extraction is dependent on the capabilities of the chosen vision model and the complexity of the PDF document.
- Papers with intricate formatting, equations, or low visual quality might not be extracted perfectly.
- The overall processing time is influenced by your local hardware specifications and the complexity of the papers being processed.

## Contributing

Contributions are welcome! Please submit Pull Requests to help improve PaperTuner.

## License

MIT License

## Acknowledgments

- ArXiv API:  For accessing scientific papers and metadata.
- Ollama: For providing local LLM inference and OCR capabilities.
- Hugging Face `datasets`: For dataset formatting and seamless ecosystem integration.

---

*Disclaimer: PaperTuner is designed for research and educational purposes. Please ensure you comply with ArXiv's terms of service when using this tool, and always cite papers appropriately in your research and publications.*
