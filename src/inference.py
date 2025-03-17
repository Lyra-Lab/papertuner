#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Inference script for Phi-4 GRPO model
"""

import argparse
import logging
from typing import List, Dict

from unsloth import FastLanguageModel, is_bfloat16_supported
from vllm import SamplingParams

from config import Config, default_config

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run inference with Phi-4 GRPO model")
    
    parser.add_argument("--model_name", type=str, default=default_config.model.model_name,
                        help="Model name or path")
    parser.add_argument("--lora_path", type=str, required=True,
                        help="Path to LoRA weights")
    parser.add_argument("--max_seq_length", type=int, default=default_config.model.max_seq_length,
                        help="Maximum sequence length")
    parser.add_argument("--temperature", type=float, default=default_config.inference.temperature,
                        help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=default_config.inference.top_p,
                        help="Top-p sampling parameter")
    parser.add_argument("--max_tokens", type=int, default=default_config.inference.max_tokens,
                        help="Maximum number of tokens to generate")
    parser.add_argument("--question", type=str, required=True,
                        help="Question to answer")
    parser.add_argument("--output_file", type=str, default=None,
                        help="Output file to save the response")
    
    return parser.parse_args()

def load_model(model_name: str, max_seq_length: int):
    """Load the model."""
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        load_in_4bit=True,
        fast_inference=True,
    )
    
    return model, tokenizer

def run_inference(model, tokenizer, lora_path: str, question: str, 
                  temperature: float, top_p: float, max_tokens: int, system_prompt: str):
    """Run inference with the model."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]
    
    text = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )
    
    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )
    
    lora_request = model.load_lora(lora_path)
    
    output = model.fast_generate(
        [text],
        sampling_params=sampling_params,
        lora_request=lora_request,
    )[0].outputs[0].text
    
    return output

def main():
    """Main function."""
    args = parse_args()
    
    # Load model and tokenizer
    logger.info(f"Loading model {args.model_name}")
    model, tokenizer = load_model(args.model_name, args.max_seq_length)
    
    # Run inference
    logger.info(f"Running inference with question: {args.question}")
    output = run_inference(
        model=model,
        tokenizer=tokenizer,
        lora_path=args.lora_path,
        question=args.question,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        system_prompt=default_config.system_prompt,
    )
    
    # Print output
    print("\n" + "="*50)
    print("QUESTION:")
    print(args.question)
    print("\nRESPONSE:")
    print(output)
    print("="*50 + "\n")
    
    # Save output to file if specified
    if args.output_file:
        with open(args.output_file, "w", encoding="utf-8") as f:
            f.write(f"Question: {args.question}\n\nResponse: {output}")
        logger.info(f"Saved output to {args.output_file}")

if __name__ == "__main__":
    main() 