"""Training module for PaperTuner research assistant models."""

import os
import argparse
import unsloth
import torch
from pathlib import Path
import datasets
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import FastLanguageModel, is_bfloat16_supported

from papertuner.config import (
    logger, DEFAULT_MODEL_NAME, DEFAULT_MAX_SEQ_LENGTH,
    DEFAULT_LORA_RANK, DEFAULT_SYSTEM_PROMPT, DEFAULT_TARGET_MODULES,
    DEFAULT_TRAINING_ARGS
)

class ResearchAssistantTrainer:
    """Handles training of research assistant models using parameter-efficient fine-tuning."""

    def __init__(
        self,
        model_name=DEFAULT_MODEL_NAME,
        max_seq_length=DEFAULT_MAX_SEQ_LENGTH,
        lora_rank=DEFAULT_LORA_RANK,
        target_modules=DEFAULT_TARGET_MODULES,
        system_prompt=DEFAULT_SYSTEM_PROMPT,
        output_dir=DEFAULT_TRAINING_ARGS["output_dir"],
        batch_size=DEFAULT_TRAINING_ARGS["per_device_train_batch_size"],
        gradient_accumulation_steps=DEFAULT_TRAINING_ARGS["gradient_accumulation_steps"],
        learning_rate=DEFAULT_TRAINING_ARGS["learning_rate"],
        max_steps=DEFAULT_TRAINING_ARGS["max_steps"],
        save_steps=DEFAULT_TRAINING_ARGS["save_steps"],
        warmup_ratio=DEFAULT_TRAINING_ARGS["warmup_ratio"],
        load_in_4bit=True
    ):
        """
        Initialize the trainer with configuration.

        Args:
            model_name: Base model to fine-tune
            max_seq_length: Maximum sequence length for the model
            lora_rank: Rank for LoRA adaptation
            target_modules: Modules to apply LoRA to
            system_prompt: System prompt for the model
            output_dir: Directory to save model checkpoints
            batch_size: Batch size per device
            gradient_accumulation_steps: Steps to accumulate gradients
            learning_rate: Learning rate for training
            max_steps: Maximum training steps
            save_steps: Steps between saving checkpoints
            warmup_ratio: Portion of training for LR warmup
            load_in_4bit: Whether to load model in 4-bit quantization
        """
        self.model_name = model_name
        self.max_seq_length = max_seq_length
        self.lora_rank = lora_rank
        self.target_modules = target_modules
        self.system_prompt = system_prompt
        self.output_dir = output_dir
        self.batch_size = batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.learning_rate = learning_rate
        self.max_steps = max_steps
        self.save_steps = save_steps
        self.warmup_ratio = warmup_ratio
        self.load_in_4bit = load_in_4bit

        logger.info(f"Trainer initialized with model: {model_name}")

    def load_model(self):
        """Load and prepare the model with LoRA adapters using optimized settings."""
        # Auto-detect the best dtype for the GPU
        dtype = None  # None for auto detection

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.model_name,
            max_seq_length=self.max_seq_length,
            dtype=dtype,
            load_in_4bit=self.load_in_4bit,
        )

        model = FastLanguageModel.get_peft_model(
            model,
            r=self.lora_rank,
            target_modules=self.target_modules,
            lora_alpha=self.lora_rank,
            lora_dropout=0,  # Optimized setting
            bias="none",     # Optimized setting
            use_gradient_checkpointing="unsloth",  # Enable long context finetuning
            random_state=7,  # Using the recommended random seed
            use_rslora=False,
            loftq_config=None
        )

        logger.info(f"Model loaded: {self.model_name}")
        logger.info(f"LoRA rank: {self.lora_rank}")
        logger.info(f"Max sequence length: {self.max_seq_length}")

        return model, tokenizer

    def load_dataset(self, dataset_name):
        """Load and format the training dataset."""
        try:
            dataset = datasets.load_dataset(dataset_name, split="train")
            logger.info(f"Loaded dataset: {dataset_name} with {len(dataset)} examples")
            return dataset
        except Exception as e:
            logger.error(f"Failed to load dataset {dataset_name}: {e}")
            raise

    def format_prompt(self, instruction, input_text="", output_text=""):
        """Format prompts using alpaca-style template with system prompt."""
        # Construct a prompt template similar to alpaca format but with our system prompt
        prompt_template = f"""<s>[INST] {self.system_prompt}

### Instruction:
{instruction}

### Input:
{input_text} [/INST]

{output_text}</s>"""
        return prompt_template

    def formatting_prompts_func(self, examples):
        """Format the dataset examples into model-compatible prompts."""
        questions = examples["question"]
        answers = examples["answer"]
        texts = []
        
        for question, answer in zip(questions, answers):
            # Format using our template and add EOS token to prevent infinite generation
            text = self.format_prompt(
                instruction=question,
                input_text="",
                output_text=answer
            )
            texts.append(text)
            
        return {"text": texts}

    def train(self, dataset_name):
        """Train the model using SFTTrainer for parameter-efficient fine-tuning."""
        try:
            # 1. Load model and tokenizer
            model, tokenizer = self.load_model()
            
            # 2. Load dataset
            dataset = self.load_dataset(dataset_name)
            
            # 3. Format the dataset
            formatted_dataset = dataset.map(
                self.formatting_prompts_func,
                batched=True,
            )
            
            # 4. Configure training arguments
            training_args = TrainingArguments(
                per_device_train_batch_size=self.batch_size,
                gradient_accumulation_steps=self.gradient_accumulation_steps,
                warmup_ratio=self.warmup_ratio,
                max_steps=self.max_steps,
                learning_rate=self.learning_rate,
                fp16=not is_bfloat16_supported(),
                bf16=is_bfloat16_supported(),
                logging_steps=1,
                optim="adamw_8bit",
                weight_decay=0.01,
                lr_scheduler_type="linear",
                save_steps=self.save_steps,
                output_dir=self.output_dir,
                report_to="none",  # Use "wandb" for WandB
                seed=3407,
            )
            
            # 5. Create SFT Trainer
            trainer = SFTTrainer(
                model=model,
                tokenizer=tokenizer,
                train_dataset=formatted_dataset,
                dataset_text_field="text",
                max_seq_length=self.max_seq_length,
                dataset_num_proc=2,
                packing=False,  # Can make training faster for short sequences
                args=training_args,
            )
            
            # 6. Train the model
            logger.info("Starting training...")
            start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
            trainer_stats = trainer.train()
            
            # 7. Log statistics
            used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
            used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
            
            logger.info(f"Training completed in {trainer_stats.metrics['train_runtime']} seconds")
            logger.info(f"Peak reserved memory: {used_memory} GB")
            logger.info(f"Peak reserved memory for training: {used_memory_for_lora} GB")
            
            # 8. Save the model
            output_path = Path(self.output_dir) / "final_model"
            model.save_pretrained(output_path)
            tokenizer.save_pretrained(output_path)
            logger.info(f"Model saved to {output_path}")
            
            return model, tokenizer, output_path
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise

    def run_inference(self, model, tokenizer, question, lora_path=None):
        """Run inference with the trained model."""
        try:
            # Load the trained model if path is provided
            if lora_path:
                model, tokenizer = self.load_model()
                # Use PeftModel to load the saved LoRA adapter weights
                from peft import PeftModel
                model = PeftModel.from_pretrained(model, lora_path)
            
            # Set model to inference mode
            FastLanguageModel.for_inference(model)
            
            # Format the prompt
            prompt = self.format_prompt(instruction=question, input_text="", output_text="")
            inputs = tokenizer([prompt], return_tensors="pt").to("cuda")
            
            # Generate the response
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                use_cache=True
            )
            
            # Decode and return the response
            result = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
            return result
            
        except Exception as e:
            logger.error(f"Inference failed: {e}")
            raise

    def demo_comparison(self, model, tokenizer, lora_path, dataset_name):
        """Compare base and fine-tuned model responses on sample questions."""
        try:
            # Load dataset for examples
            dataset = self.load_dataset(dataset_name)
            
            # Get random sample
            if len(dataset) > 0:
                sample_indices = torch.randint(0, len(dataset), (3,)).tolist()
                samples = [dataset[i] for i in sample_indices]
                
                for i, sample in enumerate(samples):
                    question = sample["question"]
                    reference = sample["answer"]
                    
                    logger.info(f"\n===== Example {i+1} =====")
                    logger.info(f"Question: {question}")
                    
                    # Get fine-tuned model response
                    ft_response = self.run_inference(model, tokenizer, question, lora_path)
                    
                    logger.info(f"\nReference Answer: {reference[:300]}...")
                    logger.info(f"\nFine-tuned Model Response: {ft_response[:300]}...")
                    logger.info("\n" + "="*30)
                    
        except Exception as e:
            logger.error(f"Demo comparison failed: {e}")
            raise

    def push_to_hf(self, model, tokenizer, repo_id, token=None):
        """Push the trained model to Hugging Face Hub."""
        try:
            from huggingface_hub import HfApi
            
            output_path = Path(self.output_dir) / "final_model"
            
            # Ensure model is saved
            if not output_path.exists():
                model.save_pretrained(output_path)
                tokenizer.save_pretrained(output_path)
                
            # Push to hub
            api = HfApi(token=token)
            logger.info(f"Pushing model to {repo_id}...")
            api.create_repo(repo_id=repo_id, exist_ok=True, private=True)
            api.upload_folder(
                folder_path=str(output_path),
                repo_id=repo_id,
                commit_message="Upload model with unsloth fine-tuning"
            )
            logger.info(f"Model pushed to {repo_id}")
            
        except Exception as e:
            logger.error(f"Failed to push to Hugging Face: {e}")
            raise

    def save_merged_model(self, model, tokenizer, output_path, save_method="merged_16bit"):
        """Save the fine-tuned model in a merged format for deployment.
        
        Args:
            model: The trained model
            tokenizer: The model tokenizer
            output_path: Path to save the merged model
            save_method: One of "merged_16bit", "merged_4bit", or "lora"
        """
        try:
            logger.info(f"Saving merged model to {output_path} using {save_method} format...")
            
            # Make sure model is in the right mode
            FastLanguageModel.for_inference(model)
            
            # Save the model in the specified format
            model.save_pretrained_merged(
                output_path,
                tokenizer,
                save_method=save_method
            )
            
            logger.info(f"Model successfully saved to {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Failed to save merged model: {e}")
            raise
            
    def save_gguf(self, model, tokenizer, output_path, quantization_method="q8_0"):
        """Save the fine-tuned model in GGUF format for llama.cpp deployment.
        
        Args:
            model: The trained model
            tokenizer: The model tokenizer
            output_path: Path to save the GGUF model
            quantization_method: Quantization method ("q8_0", "q4_k_m", "q5_k_m", etc.)
        """
        try:
            logger.info(f"Saving model in GGUF format to {output_path} using {quantization_method}...")
            
            # Make sure model is in the right mode
            FastLanguageModel.for_inference(model)
            
            # Save the model in GGUF format
            model.save_pretrained_gguf(
                output_path,
                tokenizer,
                quantization_method=quantization_method
            )
            
            logger.info(f"GGUF model successfully saved to {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Failed to save GGUF model: {e}")
            raise


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train a research assistant model")
    
    parser.add_argument(
        "--model_name", 
        type=str, 
        default=DEFAULT_MODEL_NAME,
        help="Base model to fine-tune"
    )
    parser.add_argument(
        "--dataset_name", 
        type=str, 
        required=True,
        help="Dataset name (local path or HF Hub ID)"
    )
    parser.add_argument(
        "--max_seq_length", 
        type=int, 
        default=DEFAULT_MAX_SEQ_LENGTH,
        help="Maximum sequence length"
    )
    parser.add_argument(
        "--lora_rank", 
        type=int, 
        default=DEFAULT_LORA_RANK,
        help="Rank for LoRA adaptation"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default=DEFAULT_TRAINING_ARGS["output_dir"],
        help="Directory to save model"
    )
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=DEFAULT_TRAINING_ARGS["per_device_train_batch_size"],
        help="Batch size per device"
    )
    parser.add_argument(
        "--gradient_accumulation_steps", 
        type=int, 
        default=DEFAULT_TRAINING_ARGS["gradient_accumulation_steps"],
        help="Gradient accumulation steps"
    )
    parser.add_argument(
        "--max_steps", 
        type=int, 
        default=DEFAULT_TRAINING_ARGS["max_steps"],
        help="Maximum training steps"
    )
    parser.add_argument(
        "--learning_rate", 
        type=float, 
        default=DEFAULT_TRAINING_ARGS["learning_rate"],
        help="Learning rate"
    )
    parser.add_argument(
        "--push_to_hub", 
        action="store_true", 
        help="Push model to HF Hub"
    )
    parser.add_argument(
        "--repo_id", 
        type=str, 
        default=None,
        help="HF Hub repository ID"
    )
    parser.add_argument(
        "--demo", 
        action="store_true", 
        help="Run demo after training"
    )
    parser.add_argument(
        "--export_format", 
        type=str, 
        choices=["lora", "merged_16bit", "merged_4bit", "gguf"],
        default="lora",
        help="Format to export the model (lora, merged_16bit, merged_4bit, gguf)"
    )
    parser.add_argument(
        "--gguf_quantization", 
        type=str, 
        default="q8_0",
        help="GGUF quantization method when export_format is gguf"
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    # Initialize trainer
    trainer = ResearchAssistantTrainer(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        lora_rank=args.lora_rank,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate
    )
    
    # Train the model
    model, tokenizer, output_path = trainer.train(args.dataset_name)
    
    # Save in the requested format
    if args.export_format != "lora":
        export_path = Path(args.output_dir) / f"model_{args.export_format}"
        if args.export_format == "gguf":
            export_path = trainer.save_gguf(
                model, 
                tokenizer, 
                export_path, 
                quantization_method=args.gguf_quantization
            )
        else:
            export_path = trainer.save_merged_model(
                model, 
                tokenizer, 
                export_path, 
                save_method=args.export_format
            )
        logger.info(f"Model exported to {export_path} in {args.export_format} format")
    
    # Run demo if requested
    if args.demo:
        trainer.demo_comparison(model, tokenizer, output_path, args.dataset_name)
    
    # Push to Hugging Face Hub if requested
    if args.push_to_hub and args.repo_id:
        token = os.getenv("HF_TOKEN")
        trainer.push_to_hf(model, tokenizer, args.repo_id, token)
    
    logger.info("Process completed successfully.")


if __name__ == "__main__":
    main()
