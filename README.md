# Phi-4 GRPO Training

This repository contains scripts for training the Phi-4 model using GRPO (Generative Reward-Prompted Optimization) for mathematical reasoning tasks.

## Installation

1. Clone this repository:

```bash
git clone https://github.com/Lyra-Lab/papertuner.git
cd papertuner
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. (Optional) Log in to Weights & Biases:

```bash
wandb login
```

## Usage

### Training

To train the model with default parameters:

```bash
python train.py
```

With custom parameters:

```bash
python train.py \
--model_name "unsloth/Phi-4" \
--max_seq_length 512 \
--lora_rank 16 \
--learning_rate 5e-6 \
--max_steps 500 \
--batch_size 1 \
--gradient_accumulation_steps 4 \
--num_generations 6 \
--output_dir "outputs" \
--wandb_project "phi4-grpo" \
--wandb_name "my-experiment"
```

To disable Weights & Biases logging:

```bash
python train.py --no_wandb
```

### Inference

To run inference with a trained model:

```bash
python inference.py \
--model_name "unsloth/Phi-4" \
--lora_path "outputs/grpo_saved_lora" \
```

