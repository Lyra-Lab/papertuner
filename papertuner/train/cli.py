"""
Command-line interface for model training and inference.
"""
import os
import argparse
import logging
from papertuner.train.trainer import MLAssistantTrainer
from papertuner.utils.constants import DEFAULT_MODEL_NAME, DEFAULT_MAX_SEQ_LENGTH, \
    DEFAULT_LORA_RANK, DEFAULT_MAX_STEPS, DEFAULT_SAVE_STEPS, DEFAULT_GPU_MEMORY_UTILIZATION


def main():
    """Main entry point for the training CLI."""
    parser = argparse.ArgumentParser(description="PaperTuner Model Training Tool")
    
    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Train command
    train_parser = subparsers.add_parser("train", help="Train a model using GRPO")
    train_parser.add_argument("--model-name", type=str, default=DEFAULT_MODEL_NAME,
                            help=f"Base model name to use for training (default: {DEFAULT_MODEL_NAME})")
    train_parser.add_argument("--output-dir", type=str, default="outputs",
                            help="Directory to save model outputs (default: outputs)")
    train_parser.add_argument("--max-seq-length", type=int, default=DEFAULT_MAX_SEQ_LENGTH,
                            help=f"Maximum sequence length for model (default: {DEFAULT_MAX_SEQ_LENGTH})")
    train_parser.add_argument("--lora-rank", type=int, default=DEFAULT_LORA_RANK,
                            help=f"LoRA rank for fine-tuning (default: {DEFAULT_LORA_RANK})")
    train_parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS,
                            help=f"Maximum training steps (default: {DEFAULT_MAX_STEPS})")
    train_parser.add_argument("--save-steps", type=int, default=DEFAULT_SAVE_STEPS,
                            help=f"Save checkpoint every X steps (default: {DEFAULT_SAVE_STEPS})")
    train_parser.add_argument("--gpu-memory-utilization", type=float, default=DEFAULT_GPU_MEMORY_UTILIZATION,
                            help=f"GPU memory utilization (default: {DEFAULT_GPU_MEMORY_UTILIZATION})")
    train_parser.add_argument("--dataset-path", type=str, default=None,
                            help="Path to custom dataset (default: use built-in ML QA dataset)")
    train_parser.add_argument("--report-to", type=str, default="none",
                            help="Platform to report results to (e.g., wandb, tensorboard, none)")
    
    # Inference command
    infer_parser = subparsers.add_parser("inference", help="Run inference with a trained model")
    infer_parser.add_argument("--model-name", type=str, default=DEFAULT_MODEL_NAME,
                             help=f"Base model name (default: {DEFAULT_MODEL_NAME})")
    infer_parser.add_argument("--lora-path", type=str, required=True,
                             help="Path to trained LoRA weights")
    infer_parser.add_argument("--max-seq-length", type=int, default=DEFAULT_MAX_SEQ_LENGTH,
                             help=f"Maximum sequence length for model (default: {DEFAULT_MAX_SEQ_LENGTH})")
    infer_parser.add_argument("--query", type=str, required=True,
                             help="Query for inference")
    infer_parser.add_argument("--temperature", type=float, default=0.8,
                             help="Sampling temperature (default: 0.8)")
    infer_parser.add_argument("--top-p", type=float, default=0.95,
                             help="Top-p sampling parameter (default: 0.95)")
    infer_parser.add_argument("--max-tokens", type=int, default=1024,
                             help="Maximum tokens to generate (default: 1024)")
    
    # Export command
    export_parser = subparsers.add_parser("export", help="Export a trained model")
    export_parser.add_argument("--model-name", type=str, default=DEFAULT_MODEL_NAME,
                              help=f"Base model name (default: {DEFAULT_MODEL_NAME})")
    export_parser.add_argument("--lora-path", type=str, required=True,
                              help="Path to trained LoRA weights")
    export_parser.add_argument("--output-dir", type=str, default="exported_model",
                              help="Directory to save exported model (default: exported_model)")
    export_parser.add_argument("--format", type=str, choices=["merged", "gguf"], default="merged",
                              help="Export format (default: merged)")
    export_parser.add_argument("--quantization", type=str, nargs="+", default=["q4_k_m"],
                              help="Quantization methods for GGUF (default: q4_k_m)")
    export_parser.add_argument("--push-to-hub", action="store_true",
                              help="Push model to HuggingFace Hub")
    export_parser.add_argument("--hub-model-id", type=str, default=None,
                              help="Model ID for HuggingFace Hub")
    export_parser.add_argument("--hf-token", type=str, default=None,
                              help="HuggingFace token")
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Execute the appropriate command
    if args.command == "train":
        logging.info("Starting training...")
        trainer = MLAssistantTrainer(
            model_name=args.model_name,
            output_dir=args.output_dir,
            max_seq_length=args.max_seq_length,
            lora_rank=args.lora_rank,
            max_steps=args.max_steps,
            save_steps=args.save_steps,
            gpu_memory_utilization=args.gpu_memory_utilization
        )
        
        trainer.load_model()
        trainer.load_dataset(dataset_path=args.dataset_path)
        trainer.setup_reward_functions()
        trainer.train(report_to=args.report_to)
        
        logging.info(f"Training completed. Model saved to {args.output_dir}")
        
    elif args.command == "inference":
        logging.info("Running inference...")
        trainer = MLAssistantTrainer(
            model_name=args.model_name,
            max_seq_length=args.max_seq_length
        )
        
        trainer.load_model()
        output = trainer.run_inference(
            query=args.query,
            lora_path=args.lora_path,
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens
        )
        
        print("\n" + "="*40 + " OUTPUT " + "="*40 + "\n")
        print(output)
        print("\n" + "="*88 + "\n")
        
    elif args.command == "export":
        logging.info(f"Exporting model to {args.format} format...")
        trainer = MLAssistantTrainer(
            model_name=args.model_name
        )
        
        trainer.load_model()
        
        # Load the trained LoRA weights
        if os.path.exists(args.lora_path):
            logging.info(f"Loading LoRA weights from {args.lora_path}")
            trainer.model.load_lora(args.lora_path)
        else:
            logging.error(f"LoRA weights not found at {args.lora_path}")
            return
        
        # Export the model
        if args.format == "merged":
            trainer.save_model(
                model_path=args.output_dir,
                save_method="merged_16bit",
                push_to_hub=args.push_to_hub,
                hub_model_id=args.hub_model_id,
                token=args.hf_token
            )
        elif args.format == "gguf":
            trainer.save_model_gguf(
                model_path=args.output_dir,
                quantization_methods=args.quantization,
                push_to_hub=args.push_to_hub,
                hub_model_id=args.hub_model_id,
                token=args.hf_token
            )
        
        logging.info(f"Model exported successfully to {args.output_dir}")
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main() 