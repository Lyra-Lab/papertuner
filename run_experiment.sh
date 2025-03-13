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