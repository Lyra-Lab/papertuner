#!/bin/bash

# Create a new directory for the project
mkdir research_assistant_project
cd research_assistant_project

# Create a virtual environment using uv
uv venv

# Activate the virtual environment
source .venv/bin/activate  # On Windows use: .venv\Scripts\activate

# Install dependencies using uv
uv pip install torch transformers datasets accelerate wandb unsloth vllm triton pynvml

# Clone the repository
git clone https://github.com/yourusername/research_assistant.git
cd research_assistant

# Create a sample experiment script
cat > run_experiment.sh << 'EOF'
#!/bin/bash

# Login to Weights & Biases
wandb login

# Run training with default parameters
python main.py \
    --model "unsloth/Phi-4" \
    --dataset "gsm8k" \
    --run_name "quickstart_experiment" \
    --num_epochs 1 \
    --batch_size 1 \
    --grad_accum 4 \
    --learning_rate 5e-6 \
    --max_seq_length 512 \
    --lora_rank 16 \
    --save_method "lora"
EOF

# Make the experiment script executable
chmod +x run_experiment.sh

echo "Quickstart setup complete! To run an experiment:"
echo "1. Configure your Weights & Biases account"
echo "2. Execute: ./run_experiment.sh"