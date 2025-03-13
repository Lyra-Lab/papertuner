#!/bin/bash
# Create a virtual environment using uv
uv venv

# Activate the virtual environment
source .venv/bin/activate  # On Windows use: .venv\Scripts\activate

# Install dependencies using uv
uv pip install torch transformers datasets accelerate wandb unsloth vllm triton pynvml
cd research_assistant

chmod +x run_experiment.sh

echo "Quickstart setup complete! To run an experiment:"
echo "1. Configure your Weights & Biases account"
echo "2. Execute: ./run_experiment.sh"