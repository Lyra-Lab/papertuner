"""Training module for PaperTuner research assistant models."""

import os
import sys
import json
import argparse
import time
from typing import Dict, Any, List, Optional, Tuple, Union
import random
import unsloth
import torch
import numpy as np
from pathlib import Path
import datasets
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util
from trl import GRPOConfig, GRPOTrainer
from huggingface_hub import HfApi, create_repo, login
from unsloth import FastLanguageModel, is_bfloat16_supported
from vllm import SamplingParams, LLM
import evaluate

from papertuner.config import (
    logger, DEFAULT_MODEL_NAME, DEFAULT_MAX_SEQ_LENGTH,
    DEFAULT_LORA_RANK, DEFAULT_SYSTEM_PROMPT, DEFAULT_TARGET_MODULES,
    DEFAULT_TRAINING_ARGS, MODEL_DIR, Config
)
from papertuner.utils import load_json_file, save_json_file, compute_hash

class ResearchAssistantTrainer:
    """Handles training of research assistant models using GRPO."""

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
        num_generations=DEFAULT_TRAINING_ARGS["num_generations"],
        use_vllm=DEFAULT_TRAINING_ARGS["use_vllm"],
        mixed_precision=DEFAULT_TRAINING_ARGS["mixed_precision"],
        config_file=None
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
            num_generations: Number of generations per prompt for GRPO
            use_vllm: Whether to use vLLM for inference
            mixed_precision: Precision for training (bf16, fp16, none)
            config_file: Path to a configuration file
        """
        # Load configuration if provided
        if config_file:
            self.config = Config(config_file)
            # Override with config file values if they exist
            model_name = self.config.get("model_name", model_name)
            max_seq_length = self.config.get("max_seq_length", max_seq_length)
            lora_rank = self.config.get("lora_rank", lora_rank)
            target_modules = self.config.get("target_modules", target_modules)
            system_prompt = self.config.get("system_prompt", system_prompt)
            
            training_args = self.config.get("training_args", {})
            output_dir = training_args.get("output_dir", output_dir)
            batch_size = training_args.get("per_device_train_batch_size", batch_size)
            gradient_accumulation_steps = training_args.get("gradient_accumulation_steps", gradient_accumulation_steps)
            learning_rate = training_args.get("learning_rate", learning_rate)
            max_steps = training_args.get("max_steps", max_steps)
            save_steps = training_args.get("save_steps", save_steps)
            warmup_ratio = training_args.get("warmup_ratio", warmup_ratio)
            num_generations = training_args.get("num_generations", num_generations)
            use_vllm = training_args.get("use_vllm", use_vllm)
            mixed_precision = training_args.get("mixed_precision", mixed_precision)
        else:
            self.config = None

        self.model_name = model_name
        self.max_seq_length = max_seq_length
        self.lora_rank = lora_rank
        self.target_modules = target_modules
        self.system_prompt = system_prompt
        self.output_dir = Path(output_dir)
        self.batch_size = batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.learning_rate = learning_rate
        self.max_steps = max_steps
        self.save_steps = save_steps
        self.warmup_ratio = warmup_ratio
        self.num_generations = num_generations
        self.use_vllm = use_vllm
        self.mixed_precision = mixed_precision
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup metrics
        self.metrics = {
            "rouge": evaluate.load("rouge"),
            "bertscore": evaluate.load("bertscore"),
            "bleurt": evaluate.load("bleurt", config_name="bleurt-tiny-512")
        }

        # Initialize embedding model for reward function
        self.embedding_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

        logger.info(f"Trainer initialized with model: {model_name}")
        
        # Save configuration
        self.save_configuration()
    
    def save_configuration(self):
        """Save the training configuration to the output directory."""
        config = {
            "model_name": self.model_name,
            "max_seq_length": self.max_seq_length,
            "lora_rank": self.lora_rank,
            "target_modules": self.target_modules,
            "system_prompt": self.system_prompt,
            "training_args": {
                "output_dir": str(self.output_dir),
                "per_device_train_batch_size": self.batch_size,
                "gradient_accumulation_steps": self.gradient_accumulation_steps,
                "learning_rate": self.learning_rate,
                "max_steps": self.max_steps,
                "save_steps": self.save_steps,
                "warmup_ratio": self.warmup_ratio,
                "num_generations": self.num_generations,
                "use_vllm": self.use_vllm,
                "mixed_precision": self.mixed_precision,
            },
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        save_json_file(config, self.output_dir / "training_config.json")

    def load_model(self, checkpoint_path=None):
        """
        Load and prepare the model with LoRA adapters using optimized settings.
        
        Args:
            checkpoint_path: Optional path to a checkpoint to load
            
        Returns:
            tuple: (model, tokenizer, peft_model)
        """
        # Determine precision settings
        use_bf16 = self.mixed_precision == "bf16" and is_bfloat16_supported()
        use_fp16 = self.mixed_precision == "fp16" or (self.mixed_precision == "bf16" and not use_bf16)
        
        # Load the base model
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.model_name,
            max_seq_length=self.max_seq_length,
            load_in_4bit=True,
            fast_inference=self.use_vllm,
            max_lora_rank=self.lora_rank,
            gpu_memory_utilization=0.7,
            bf16=use_bf16,
            fp16=use_fp16,
        )

        # Setup LoRA adapters
        peft_model = FastLanguageModel.get_peft_model(
            model,
            r=self.lora_rank,
            target_modules=self.target_modules,
            lora_alpha=self.lora_rank,
            use_gradient_checkpointing="unsloth",  # Enable long context finetuning
            random_state=7,  # Using the recommended random seed
        )
        
        # Load from checkpoint if provided
        if checkpoint_path and os.path.exists(checkpoint_path):
            try:
                logger.info(f"Loading adapter from checkpoint: {checkpoint_path}")
                peft_model.load_adapter(checkpoint_path, "default")
                logger.info("Adapter loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load checkpoint: {e}")

        logger.info(f"Model loaded: {self.model_name}")
        logger.info(f"LoRA rank: {self.lora_rank}")
        logger.info(f"Max sequence length: {self.max_seq_length}")
        logger.info(f"Precision: {self.mixed_precision}")

        return model, tokenizer, peft_model

    def load_dataset(self, dataset_name, split="train", cache_dir=None, filter_by_category=None):
        """
        Load and format the training dataset.
        
        Args:
            dataset_name: Name of the dataset to load
            split: Dataset split to load (train, validation, test)
            cache_dir: Directory to cache the dataset
            filter_by_category: Optional category to filter questions by
            
        Returns:
            datasets.Dataset: Formatted dataset
        """
        try:
            dataset = datasets.load_dataset(dataset_name, split=split, cache_dir=cache_dir)
            logger.info(f"Loaded dataset: {dataset_name} ({split}) with {len(dataset)} examples")
            
            # Filter by category if specified
            if filter_by_category:
                filtered_dataset = dataset.filter(lambda x: x.get("category") == filter_by_category)
                logger.info(f"Filtered to {len(filtered_dataset)} examples with category '{filter_by_category}'")
                dataset = filtered_dataset

            # Format the dataset for training
            def format_example(x):
                return {
                    'prompt': [
                        {'role': 'system', 'content': self.system_prompt},
                        {'role': 'user', 'content': x['question']}
                    ],
                    'answer': x['answer']
                }

            formatted_dataset = dataset.map(format_example)
            return formatted_dataset

        except Exception as e:
            logger.error(f"Failed to load dataset {dataset_name}: {e}")
            raise

    def extract_answer(self, text):
        """
        Extract the answer part from a response with <think> tags.
        
        Args:
            text: Response text potentially containing <think> tags
            
        Returns:
            str: The extracted answer portion
        """
        # Check if the response follows the expected format
        if "</think>" in text:
            answer = text.split("</think>")[-1].strip()
            return answer
        return text

    def correctness_reward_func(self, prompts, completions, answer, **kwargs):
        """
        Reward function based on semantic similarity to reference answer.

        Args:
            prompts: Input prompts
            completions: Model completions
            answer: Reference answers

        Returns:
            list: Reward scores for each completion
        """
        responses = [completion[0]['content'] for completion in completions]
        q = prompts[0][-1]['content']

        # Extract answers from responses and ground truth
        extracted_responses = [self.extract_answer(r) for r in responses]
        extracted_answer = self.extract_answer(answer[0])

        # Compute embeddings
        response_embeddings = self.embedding_model.encode(extracted_responses, convert_to_tensor=True)
        answer_embedding = self.embedding_model.encode([extracted_answer], convert_to_tensor=True)

        # Calculate cosine similarity
        cos_similarities = util.pytorch_cos_sim(response_embeddings, answer_embedding)
        
        # Normalize the similarities to [0, 1] range and shape for return
        rewards = (cos_similarities + 1) / 2  # Convert from [-1, 1] to [0, 1]
        return rewards.squeeze().tolist()

    def get_trainer(self, model, tokenizer, dataset):
        """
        Create a GRPO trainer instance.
        
        Args:
            model: Model to train
            tokenizer: Tokenizer for the model
            dataset: Dataset to train on
            
        Returns:
            GRPOTrainer: Configured trainer
        """
        # Configure the training environment
        grpo_config = GRPOConfig(
            num_generations=self.num_generations,
            peft_config=None,  # We've already set up the PEFT model
            max_prompt_length=DEFAULT_TRAINING_ARGS["max_prompt_length"],
            reward_model=self.correctness_reward_func,
            use_vllm=self.use_vllm,
        )

        # Set up the trainer
        trainer = GRPOTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=dataset,
            grpo_config=grpo_config,
            args={
                "output_dir": str(self.output_dir),
                "num_train_epochs": 1,
                "learning_rate": self.learning_rate,
                "per_device_train_batch_size": self.batch_size,
                "gradient_accumulation_steps": self.gradient_accumulation_steps,
                "optim": DEFAULT_TRAINING_ARGS["optim"],
                "save_steps": self.save_steps,
                "logging_steps": DEFAULT_TRAINING_ARGS["logging_steps"],
                "max_steps": self.max_steps,
                "remove_unused_columns": False,
                "report_to": DEFAULT_TRAINING_ARGS["report_to"],
                "lr_scheduler_type": DEFAULT_TRAINING_ARGS["lr_scheduler_type"],
                "warmup_ratio": self.warmup_ratio,
            },
        )

        return trainer

    def train(self, dataset_name, checkpoint_path=None, cache_dir=None, filter_by_category=None):
        """
        Train a model using the specified dataset.
        
        Args:
            dataset_name: Name of the dataset to train on
            checkpoint_path: Optional path to a checkpoint to continue training from
            cache_dir: Directory to cache the dataset
            filter_by_category: Optional category to filter questions by
            
        Returns:
            dict: Training results with model and tokenizer
        """
        try:
            # Load model and dataset
            model, tokenizer, peft_model = self.load_model(checkpoint_path)
            dataset = self.load_dataset(dataset_name, cache_dir=cache_dir, filter_by_category=filter_by_category)
            
            # Create trainer and train the model
            trainer = self.get_trainer(peft_model, tokenizer, dataset)
            
            logger.info("Starting training")
            trainer.train()
            
            # Save the final model
            final_output_dir = self.output_dir / "final"
            self.save_model(peft_model, tokenizer, final_output_dir)
            
            # Evaluate the model
            if len(dataset) > 0:
                eval_dataset = dataset
                if len(dataset) > 50:  # Use a subset for quick evaluation
                    eval_dataset = dataset.select(range(min(50, len(dataset))))
                
                evaluation_results = self.evaluate_model(peft_model, tokenizer, eval_dataset)
                eval_path = self.output_dir / "evaluation_results.json"
                save_json_file(evaluation_results, eval_path)
                logger.info(f"Evaluation results saved to {eval_path}")
            
            return {
                "model": peft_model,
                "tokenizer": tokenizer,
                "lora_path": str(final_output_dir),
                "evaluation": evaluation_results if 'evaluation_results' in locals() else None
            }
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise

    def save_model(self, model, tokenizer, output_path):
        """
        Save the trained model and tokenizer.
        
        Args:
            model: The trained model
            tokenizer: The model tokenizer
            output_path: Path to save the model
            
        Returns:
            str: Path to the saved model
        """
        try:
            # Create output directory
            output_path = Path(output_path)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Save the adapter weights
            model.save_pretrained(output_path)
            
            # Save the tokenizer
            tokenizer.save_pretrained(output_path)
            
            # Save model card with information
            model_card = {
                "base_model": self.model_name,
                "adapter_type": "LoRA",
                "rank": self.lora_rank,
                "max_sequence_length": self.max_seq_length,
                "target_modules": self.target_modules,
                "system_prompt": self.system_prompt,
                "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "papertuner_version": "0.0.8"  # Should be dynamically retrieved
            }
            
            save_json_file(model_card, output_path / "model_card.json")
            logger.info(f"Model saved to {output_path}")
            
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            raise

    def run_inference(self, model, tokenizer, question, lora_path=None):
        """
        Run inference with the fine-tuned model.
        
        Args:
            model: The model to use
            tokenizer: Tokenizer for the model
            question: Question to ask the model
            lora_path: Optional path to LoRA adapter to load
            
        Returns:
            str: The model's response
        """
        try:
            # If we have a path to LoRA weights, load them
            if lora_path and not hasattr(model, "_has_loaded_adapter"):
                model.load_adapter(lora_path, "default")
                model._has_loaded_adapter = True  # Mark as loaded to avoid reloading
            
            # Format the input with system prompt
            formatted_prompt = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": question}
            ]
            
            # Convert to model input format
            prompt = tokenizer.apply_chat_template(
                formatted_prompt,
                tokenize=False,
                add_generation_prompt=True
            )
            
            # Generate response
            outputs = model.generate(
                input_ids=tokenizer.encode(prompt, return_tensors="pt").to(model.device),
                max_new_tokens=512,
                temperature=0.3,
                top_p=0.9,
                do_sample=True
            )
            
            # Decode and return
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Extract just the assistant's response
            response = response.split("Assistant: ")[-1]
            
            return response
            
        except Exception as e:
            logger.error(f"Inference failed: {e}")
            return f"Error during inference: {str(e)}"

    def evaluate_model(self, model, tokenizer, dataset, lora_path=None, num_examples=None):
        """
        Evaluate a model on a dataset using multiple metrics.
        
        Args:
            model: Model to evaluate
            tokenizer: Tokenizer for the model
            dataset: Dataset to evaluate on
            lora_path: Optional path to adapter weights
            num_examples: Number of examples to evaluate (None for all)
            
        Returns:
            dict: Evaluation results
        """
        if num_examples is not None:
            dataset = dataset.select(range(min(num_examples, len(dataset))))
        
        logger.info(f"Evaluating model on {len(dataset)} examples")
        
        predictions = []
        references = []
        scores = []
        
        # If we have a path to LoRA weights, load them
        if lora_path and not hasattr(model, "_has_loaded_adapter"):
            model.load_adapter(lora_path, "default")
            model._has_loaded_adapter = True
        
        # Run inference on each example
        for example in tqdm(dataset, desc="Evaluating"):
            question = example["prompt"][-1]["content"]
            reference = self.extract_answer(example["answer"])
            
            # Get model prediction
            prediction = self.run_inference(model, tokenizer, question, lora_path)
            prediction = self.extract_answer(prediction)
            
            predictions.append(prediction)
            references.append(reference)
            
            # Calculate semantic similarity score
            pred_embedding = self.embedding_model.encode([prediction], convert_to_tensor=True)
            ref_embedding = self.embedding_model.encode([reference], convert_to_tensor=True)
            similarity = util.pytorch_cos_sim(pred_embedding, ref_embedding).item()
            scores.append(similarity)
        
        # Calculate metrics
        results = {
            "semantic_similarity": {
                "mean": float(np.mean(scores)),
                "median": float(np.median(scores)),
                "std": float(np.std(scores)),
                "min": float(np.min(scores)),
                "max": float(np.max(scores)),
            }
        }
        
        # Add Rouge metrics
        rouge_results = self.metrics["rouge"].compute(
            predictions=predictions, 
            references=references,
            use_aggregator=True
        )
        results["rouge"] = {k: float(v) for k, v in rouge_results.items()}
        
        # Add BERTScore (only for a subset to save computation time)
        if len(predictions) <= 20:
            bertscore_results = self.metrics["bertscore"].compute(
                predictions=predictions, 
                references=references, 
                lang="en"
            )
            results["bertscore"] = {
                "precision": float(np.mean(bertscore_results["precision"])),
                "recall": float(np.mean(bertscore_results["recall"])),
                "f1": float(np.mean(bertscore_results["f1"])),
            }
        
        # Add example predictions
        if len(predictions) <= 10:
            results["examples"] = [
                {"question": dataset[i]["prompt"][-1]["content"],
                 "reference": references[i],
                 "prediction": predictions[i],
                 "similarity": float(scores[i])}
                for i in range(len(predictions))
            ]
        else:
            # Add a sample of predictions
            sample_indices = random.sample(range(len(predictions)), 10)
            results["examples"] = [
                {"question": dataset[i]["prompt"][-1]["content"],
                 "reference": references[i],
                 "prediction": predictions[i],
                 "similarity": float(scores[i])}
                for i in sample_indices
            ]
        
        return results

    def demo_comparison(self, model, tokenizer, dataset_name, split="test", lora_path=None, num_examples=5):
        """
        Run a demonstration comparing base model to fine-tuned model.
        
        Args:
            model: Fine-tuned model
            tokenizer: Tokenizer for models
            dataset_name: Dataset to sample questions from
            split: Dataset split to use
            lora_path: Path to LoRA adapter for fine-tuned model
            num_examples: Number of examples to show
        
        Returns:
            dict: Comparison results
        """
        # Load dataset
        dataset = self.load_dataset(dataset_name, split=split)
        if len(dataset) == 0:
            logger.warning(f"Empty dataset: {dataset_name} ({split})")
            return {"error": "Empty dataset"}
        
        # Load base model for comparison
        base_model, base_tokenizer, _ = self.load_model()  # No checkpoint path = base model
        
        # Select random examples
        if len(dataset) <= num_examples:
            examples = dataset
        else:
            indices = random.sample(range(len(dataset)), num_examples)
            examples = dataset.select(indices)
        
        comparisons = []
        
        for example in examples:
            question = example["prompt"][-1]["content"]
            reference = self.extract_answer(example["answer"])
            
            # Generate response with base model
            base_response = self.run_inference(base_model, base_tokenizer, question)
            base_response = self.extract_answer(base_response)
            
            # Generate response with fine-tuned model
            ft_response = self.run_inference(model, tokenizer, question, lora_path)
            ft_response = self.extract_answer(ft_response)
            
            comparisons.append({
                "question": question,
                "reference": reference,
                "base_model_response": base_response,
                "fine_tuned_response": ft_response
            })
        
        return {"examples": comparisons}

    def push_to_hf(self, model, tokenizer, repo_id, token=None, lora_path=None, commit_message=None):
        """
        Push the model to Hugging Face Hub.
        
        Args:
            model: Model to push
            tokenizer: Tokenizer for the model
            repo_id: Hugging Face repository ID
            token: HF API token (defaults to env var)
            lora_path: Path to LoRA adapter
            commit_message: Custom commit message
            
        Returns:
            str: URL to the model on the hub
        """
        token = token or os.environ.get("HF_TOKEN")
        if not token:
            raise ValueError("No Hugging Face token provided. Set the HF_TOKEN environment variable or pass token")
        
        # Log in to Hugging Face
        login(token=token, add_to_git_credential=False)
        
        # Create repo if it doesn't exist
        try:
            api = HfApi(token=token)
            create_repo(repo_id=repo_id, token=token, private=False, exist_ok=True)
            logger.info(f"Repository {repo_id} ready for upload")
        except Exception as e:
            logger.error(f"Failed to create repository: {e}")
            raise
        
        # If we have a path to LoRA weights, ensure they're loaded
        if lora_path and not hasattr(model, "_has_loaded_adapter"):
            model.load_adapter(lora_path, "default")
            model._has_loaded_adapter = True
        
        try:
            # Prepare a temporary directory for the complete model
            tmp_dir = Path(MODEL_DIR) / f"tmp_upload_{int(time.time())}"
            tmp_dir.mkdir(parents=True, exist_ok=True)
            
            # Save model and tokenizer to this directory
            self.save_model(model, tokenizer, tmp_dir)
            
            # Create model card
            readme_content = f"""# {repo_id.split('/')[-1]}

## Model Description
This is a research assistant model fine-tuned with [PaperTuner](https://github.com/yourusername/papertuner).

- **Base Model:** {self.model_name}
- **Adapter Type:** LoRA (rank {self.lora_rank})
- **Max Sequence Length:** {self.max_seq_length}
- **Created:** {time.strftime("%Y-%m-%d")}

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "{repo_id}"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

def ask(question):
    prompt = f"<|system|>\\nYou are a helpful research assistant.\\n<|user|>\\n{question}\\n<|assistant|>\\n"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=512)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Example usage
response = ask("How would you design a transformer model for time series forecasting?")
print(response)
```

## Training Details
This model was fine-tuned using GRPO (Growing Rank Pruned Optimization) with a reward function based on semantic similarity.
"""
            
            with open(tmp_dir / "README.md", "w") as f:
                f.write(readme_content)
            
            # Push to Hub
            commit_message = commit_message or f"Upload model fine-tuned with PaperTuner"
            url = api.upload_folder(
                folder_path=str(tmp_dir),
                repo_id=repo_id,
                commit_message=commit_message
            )
            
            logger.info(f"Model pushed to Hugging Face Hub: {url}")
            return url
            
        except Exception as e:
            logger.error(f"Failed to push model to Hugging Face: {e}")
            raise
        finally:
            # Cleanup temporary directory
            import shutil
            if 'tmp_dir' in locals() and tmp_dir.exists():
                shutil.rmtree(tmp_dir)
    
    def create_gradio_demo(self, model, tokenizer, lora_path=None, output_dir=None, share=False):
        """Create a Gradio demo for the model."""
        try:
            import gradio as gr
        except ImportError:
            logger.error("Gradio not installed. Install with 'pip install gradio'")
            return None
        
        # Load the adapter if path is provided
        if lora_path and not hasattr(model, "_has_loaded_adapter"):
            model.load_adapter(lora_path, "default")
            model._has_loaded_adapter = True
        
        def predict(question):
            response = self.run_inference(model, tokenizer, question, lora_path)
            return response
        
        demo = gr.Interface(
            fn=predict,
            inputs=gr.Textbox(lines=3, placeholder="Ask a research question..."),
            outputs=gr.Textbox(lines=10),
            title="Research Assistant Model",
            description=f"This model was fine-tuned from {self.model_name} using PaperTuner.",
            examples=[
                "How would you design a transformer model for time series forecasting?",
                "What are the best approaches for handling class imbalance in image segmentation?",
                "Explain how contrastive learning can be applied to multimodal data."
            ]
        )
        
        if output_dir:
            demo.save_to_disk(output_dir)
            logger.info(f"Demo saved to {output_dir}")
            
        return demo.launch(share=share)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train a research assistant model")
    
    parser.add_argument(
        "--model", 
        type=str, 
        default=DEFAULT_MODEL_NAME,
        help="Base model to fine-tune"
    )
    
    parser.add_argument(
        "--dataset", 
        type=str, 
        required=True,
        help="Dataset to train on (Hugging Face dataset ID)"
    )
    
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default="./outputs",
        help="Directory to save model outputs"
    )
    
    parser.add_argument(
        "--lora-rank", 
        type=int, 
        default=DEFAULT_LORA_RANK,
        help="Rank for LoRA adapters"
    )
    
    parser.add_argument(
        "--max-steps", 
        type=int, 
        default=DEFAULT_TRAINING_ARGS["max_steps"],
        help="Maximum training steps"
    )
    
    parser.add_argument(
        "--config", 
        type=str, 
        default=None,
        help="Path to a configuration file"
    )
    
    parser.add_argument(
        "--checkpoint", 
        type=str, 
        default=None,
        help="Path to a checkpoint to resume training from"
    )
    
    parser.add_argument(
        "--push-to-hub", 
        action="store_true",
        help="Push model to Hugging Face Hub after training"
    )
    
    parser.add_argument(
        "--hub-model-id", 
        type=str, 
        default=None,
        help="Model ID for Hugging Face Hub (required if --push-to-hub is set)"
    )
    
    parser.add_argument(
        "--filter-category", 
        type=str, 
        default=None,
        help="Filter dataset by question category"
    )
    
    parser.add_argument(
        "--create-demo", 
        action="store_true",
        help="Create a Gradio demo after training"
    )
    
    parser.add_argument(
        "--share-demo", 
        action="store_true",
        help="Share the Gradio demo publicly (if --create-demo)"
    )
    
    return parser.parse_args()

def main():
    """Main function for CLI usage."""
    args = parse_args()
    
    # Create trainer
    trainer = ResearchAssistantTrainer(
        model_name=args.model,
        lora_rank=args.lora_rank,
        output_dir=args.output_dir,
        max_steps=args.max_steps,
        config_file=args.config
    )
    
    # Train the model
    results = trainer.train(
        dataset_name=args.dataset,
        checkpoint_path=args.checkpoint,
        filter_by_category=args.filter_category
    )
    
    # Push to Hub if requested
    if args.push_to_hub:
        if not args.hub_model_id:
            logger.error("--hub-model-id is required when --push-to-hub is set")
            sys.exit(1)
        
        trainer.push_to_hf(
            model=results["model"],
            tokenizer=results["tokenizer"],
            repo_id=args.hub_model_id,
            lora_path=results["lora_path"]
        )
    
    # Create demo if requested
    if args.create_demo:
        trainer.create_gradio_demo(
            model=results["model"],
            tokenizer=results["tokenizer"],
            lora_path=results["lora_path"],
            output_dir=f"{args.output_dir}/demo",
            share=args.share_demo
        )
    
    logger.info(f"Training completed. Model saved to {results['lora_path']}")
    
if __name__ == "__main__":
    main()
