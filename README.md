# Research Assistant AI

An AI-powered PhD-level research assistant that predicts the optimal "next step" in scientific research based on prior stages (problem, literature, hypothesis).

## Project Overview

This system aims to replicate human researchers' decision-making logic by training on sequences of historical papers. A critic model evaluates outputs against real-world methodologies, creating an RL reward signal grounded in actual scientific practice.

## Setup

### Requirements

- Python 3.10+
- CUDA-compatible GPU (recommended)

### Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd research_assistant
   ```

2. Create a virtual environment and install dependencies:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. Set up Weights & Biases:
   ```bash
   wandb login
   ```

## Usage

### Training

To train the model:
```bash
python main.py --model meta-llama/Llama-2-7b-hf --dataset <path-to-dataset> --run_name my_experiment
```

### Evaluation

To evaluate the model:
```bash
python -m evaluation.evaluator --model_path models/final_model --dataset <path-to-test-dataset> --output_file evaluation/results.json
```

### Demo

To launch the interactive demo:
```bash
python -m app.demo
```

## Project Structure

research_assistant/
├── data/ # Raw and processed datasets
├── models/ # Pretrained models, fine-tuned checkpoints
├── training/ # Training scripts (SFT, RL)
├── evaluation/ # Metrics, expert review results
├── app/ # Demo UI (Gradio/Streamlit)
└── configs/ # Hyperparameters, paths, model settings
