# PaperTuner: PhD-Level Research Assistant

## Overview

PaperTuner is a project aimed at developing a PhD-level research assistant that predicts the optimal approach in scientific research (e.g., methodology design) based on prior stages and the current state of the research project (problem, literature, hypothesis). The system is trained on sequences of historical papers to learn human researchers' decision-making logic.

## Project Structure

```
.
├── data
│   ├── processed_dataset
│   └── raw_dataset
├── data_reqs.txt
├── main.py        # Data extraction script
├── pyproject.toml
├── README.md      # This file
├── src
│   ├── dataset.py # Dataset-related functions (deprecated/legacy)
│   └── train.py   # Training script
└── uv.lock
```

## Requirements

1.  **Python Environment:** Ensure you have Python 3.8+ installed.
2.  **Dependencies:** Install the required Python packages.
    ```bash
    pip install -r requirements.txt
    ```

    or with `uv` if you prefer a faster resolution and install
     ```bash
     uv pip install -r requirements.txt
     ```

3.  **API Keys:** Set the following environment variables:

    *   `HF_TOKEN`: Hugging Face API token for uploading the dataset.
    *   `HF_REPO_ID`: Hugging Face repository ID to upload the dataset to.
    *   `OPENROUTER_API_KEY` or `GEMINI_API_KEY`: API key for the Gemini model.

    Create a `.env` file in the root directory with the required environment variables:

    ```
    HF_TOKEN=your_huggingface_token
    HF_REPO_ID=your_huggingface_repo_id
    GEMINI_API_KEY=your_gemini_api_key
    ```

## Usage

### 1. Data Extraction

Run the data extraction script (`main.py`) to download and process research papers from arXiv:

```bash
python main.py
```

This script will:

*   Set up necessary directories.
*   Download research papers related to machine learning.
*   Extract relevant sections (problem, methodology, results) from the papers.
*   Generate question-answer pairs based on the extracted sections.
*   Save the processed data to the `data/processed_dataset` directory.
*   Upload the dataset to the Hugging Face Hub.

### 2. Model Training

After extracting and processing the data, you can train the research assistant model using the `train.py` script:

```bash
python src/train.py
```

This script will:

*   Load the processed dataset from the `data/processed_dataset` directory.
*   Fine-tune the DeepSeek-R1-1.5B model using the GRPO (Generative Reinforcement from Preference Optimization) approach.
*   Save the trained model (LoRA weights and GGUF format) to the `data/trained_model` directory.

### 3. Customization

*   `main.py`: You can customize the search queries, number of papers to process, and API key settings in this script.
*   `src/train.py`: You can adjust the training hyperparameters, model settings, and reward functions in this script.

## Notes

*   The data extraction script uses the Gemini API to generate question-answer pairs. Ensure you have a valid API key and the necessary permissions.
*   The training script fine-tunes the DeepSeek-R1-1.5B model using GRPO. Make sure you have the required hardware (GPU) and dependencies installed.
*   This project is inspired by the DeepScalar project and leverages pre-trained and potentially pre-distilled Large Language Models (LLMs).
