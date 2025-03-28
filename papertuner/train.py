"""Training module for PaperTuner research assistant models."""

import os
import argparse
import unsloth
import torch
from pathlib import Path
import datasets
from sentence_transformers import SentenceTransformer, util
from trl import GRPOConfig, GRPOTrainer
from unsloth import FastLanguageModel, is_bfloat16_supported
from vllm import SamplingParams

from papertuner.config import (
    logger, DEFAULT_MODEL_NAME, DEFAULT_MAX_SEQ_LENGTH,
    DEFAULT_LORA_RANK, DEFAULT_SYSTEM_PROMPT, DEFAULT_TARGET_MODULES,
    DEFAULT_TRAINING_ARGS
)

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
        use_vllm=DEFAULT_TRAINING_ARGS["use_vllm"]
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
        self.num_generations = num_generations
        self.use_vllm = use_vllm

        # Initialize embedding model for reward function
        self.embedding_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

        logger.info(f"Trainer initialized with model: {model_name}")

    def load_model(self):
        """Load and prepare the model with LoRA adapters using optimized settings."""
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.model_name,
            max_seq_length=self.max_seq_length,
            load_in_4bit=True,
            fast_inference=self.use_vllm,
            max_lora_rank=self.lora_rank,
            gpu_memory_utilization=0.7,
        )

        peft_model = FastLanguageModel.get_peft_model(
            model,
            r=self.lora_rank,
            target_modules=self.target_modules,
            lora_alpha=self.lora_rank,
            use_gradient_checkpointing="unsloth",  # Enable long context finetuning
            random_state=7,  # Using the recommended random seed
        )

        logger.info(f"Model loaded: {self.model_name}")
        logger.info(f"LoRA rank: {self.lora_rank}")
        logger.info(f"Max sequence length: {self.max_seq_length}")

        return model, tokenizer, peft_model

    def load_dataset(self, dataset_name):
        """Load and format the training dataset."""
        try:
            dataset = datasets.load_dataset(dataset_name, split="train")
            logger.info(f"Loaded dataset: {dataset_name} with {len(dataset)} examples")

            def format_example(x):
                return {
                    'prompt': [
                        {'role': 'system', 'content': self.system_prompt},
                        {'role': 'user', 'content': x['question']}
                    ],
                    'answer': x['answer']
                }

            return dataset.map(format_example)

        except Exception as e:
            logger.error(f"Failed to load dataset {dataset_name}: {e}")
            raise

    def extract_answer(self, text):
        """Extract the answer part from a response with <think> tags."""
        return text.split("</think>")[-1].strip() if "</think>" in text else text

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

        # Compute similarities and rewards
        rewards = []
        for resp_emb in response_embeddings:
            sim = util.pytorch_cos_sim(resp_emb.unsqueeze(0), answer_embedding).item()
            rewards.append(sim * 2)  # Scale similarity to range [0, 2]

        # Log an example for monitoring
        if len(rewards) > 0:
            logger.info('-'*20)
            logger.info(f"Question:\n{q}")
            logger.info(f"\nReference Answer (excerpt):\n{extracted_answer[:300]}...")
            logger.info(f"\nModel Response (excerpt):\n{extracted_responses[0][:300]}...")
            logger.info(f"\nSimilarity Score: {rewards[0]/2:.4f}")
            logger.info(f"Reward: {rewards[0]:.4f}")

        return rewards

    def get_trainer(self, model, tokenizer, dataset):
        """Configure and create the GRPO trainer with optimized settings."""
        return GRPOTrainer(
            model=model,
            processing_class=tokenizer,
            reward_funcs=[self.correctness_reward_func],
            args=GRPOConfig(
                use_vllm=self.use_vllm,
                learning_rate=self.learning_rate,
                adam_beta1=DEFAULT_TRAINING_ARGS["adam_beta1"],
                adam_beta2=DEFAULT_TRAINING_ARGS["adam_beta2"],
                weight_decay=DEFAULT_TRAINING_ARGS["weight_decay"],
                warmup_ratio=self.warmup_ratio,
                lr_scheduler_type=DEFAULT_TRAINING_ARGS["lr_scheduler_type"],
                optim=DEFAULT_TRAINING_ARGS["optim"],
                logging_steps=DEFAULT_TRAINING_ARGS["logging_steps"],
                bf16=is_bfloat16_supported(),
                fp16=not is_bfloat16_supported(),
                per_device_train_batch_size=self.batch_size,
                gradient_accumulation_steps=self.gradient_accumulation_steps,
                num_generations=self.num_generations,
                max_prompt_length=DEFAULT_TRAINING_ARGS["max_prompt_length"],
                max_completion_length=self.max_seq_length - DEFAULT_TRAINING_ARGS["max_prompt_length"],
                max_steps=self.max_steps,
                save_steps=self.save_steps,
                max_grad_norm=DEFAULT_TRAINING_ARGS["max_grad_norm"],
                report_to=DEFAULT_TRAINING_ARGS["report_to"],
                output_dir=self.output_dir,
            ),
            train_dataset=dataset,
        )

    def train(self, dataset_name):
        """Run the training process end-to-end."""
        # Ensure output directory exists
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

        # Load dataset
        logger.info(f"Loading dataset: {dataset_name}")
        dataset = self.load_dataset(dataset_name)

        # Load model
        logger.info(f"Loading model: {self.model_name}")
        model, tokenizer, peft_model = self.load_model()

        # Initialize trainer
        logger.info("Setting up trainer")
        trainer = self.get_trainer(peft_model, tokenizer, dataset)

        # Run training
        logger.info("Starting GRPO training")
        trainer.train()

        # Save trained LoRA weights
        lora_path = Path(self.output_dir) / "final_lora"
        logger.info(f"Saving LoRA adapter to {lora_path}")
        peft_model.save_lora(str(lora_path))

        return {
            "model": model,
            "tokenizer": tokenizer,
            "peft_model": peft_model,
            "lora_path": lora_path
        }

    def run_inference(self, model, tokenizer, question, lora_path=None):
        """
        Run inference using the trained model.

        Args:
            model: The model to use
            tokenizer: The tokenizer
            question: Question to answer
            lora_path: Optional path to LoRA weights

        Returns:
            str: The model's response
        """
        prompt = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": question}
            ],
            tokenize=False,
            add_generation_prompt=True
        )

        lora_request = None
        if lora_path:
            lora_request = model.load_lora(lora_path)

        response = model.fast_generate(
            [prompt],
            sampling_params=SamplingParams(
                temperature=0.8,
                top_p=0.95,
                max_tokens=1024,
            ),
            lora_request=lora_request,
        )[0].outputs[0].text

        return response

    def demo_comparison(self, model, tokenizer, lora_path, dataset_name):
        """
        Run a demo comparison of model outputs before and after training.

        Args:
            model: The model
            tokenizer: The tokenizer
            lora_path: Path to saved LoRA weights
            dataset_name: The name of the dataset to load questions from
        """
        # Load the dataset to extract test questions
        dataset = datasets.load_dataset(dataset_name, split="train")
        test_questions = [example['question'] for example in dataset.select(random.sample(range(len(dataset)), 3))]

        for i, question in enumerate(test_questions):
            logger.info(f"\n=== Test Question {i+1} ===")
            logger.info(question)

            logger.info("\n=== Pre-training Response ===")
            pre_response = self.run_inference(model, tokenizer, question)
            logger.info(pre_response)

            logger.info("\n=== Post-training Response ===")
            post_response = self.run_inference(model, tokenizer, question, lora_path)
            logger.info(post_response)

    def push_to_hf(self, model, tokenizer, repo_id, token=None):
        """
        Upload the model to Hugging Face Hub.

        Args:
            model: The model to upload
            tokenizer: The tokenizer
            repo_id: Hugging Face repo ID
            token: HF API token
        """
        if token is None:
            token = os.environ.get("HF_TOKEN")

        if not token:
            logger.warning("No HF_TOKEN found. Skipping upload to Hugging Face Hub.")
            return False

        try:
            logger.info(f"Pushing model to HuggingFace Hub: {repo_id}")
            model.push_to_hub_gguf(
                repo_id,
                tokenizer,
                quantization_method=["q4_k_m", "q8_0", "q5_k_m"],
                token=token
            )
            logger.info(f"Model successfully uploaded to {repo_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to upload model to HuggingFace Hub: {e}")
            return False


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train a research assistant model")

    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_NAME,
                        help=f"Base model to fine-tune (default: {DEFAULT_MODEL_NAME})")
    parser.add_argument("--dataset", type=str, default="densud2/ml_qa_dataset",
                        help="Dataset to use for training")
    parser.add_argument("--output-dir", type=str, default=DEFAULT_TRAINING_ARGS["output_dir"],
                        help="Directory to save model outputs")
    parser.add_argument("--lora-rank", type=int, default=DEFAULT_LORA_RANK,
                        help=f"Rank for LoRA fine-tuning (default: {DEFAULT_LORA_RANK})")
    parser.add_argument("--max-seq-length", type=int, default=DEFAULT_MAX_SEQ_LENGTH,
                        help=f"Maximum sequence length (default: {DEFAULT_MAX_SEQ_LENGTH})")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_TRAINING_ARGS["per_device_train_batch_size"],
                        help="Batch size per device")
    parser.add_argument("--grad-accum", type=int, default=DEFAULT_TRAINING_ARGS["gradient_accumulation_steps"],
                        help="Gradient accumulation steps")
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_TRAINING_ARGS["learning_rate"],
                        help=f"Learning rate (default: {DEFAULT_TRAINING_ARGS['learning_rate']})")
    parser.add_argument("--max-steps", type=int, default=DEFAULT_TRAINING_ARGS["max_steps"],
                        help=f"Maximum training steps (default: {DEFAULT_TRAINING_ARGS['max_steps']})")
    parser.add_argument("--push-to-hub", action="store_true",
                        help="Push model to HuggingFace Hub")
    parser.add_argument("--hub-repo-id", type=str, default=None,
                        help="HuggingFace Hub repository ID")
    parser.add_argument("--skip-demo", action="store_true",
                        help="Skip the demo comparison after training")

    return parser.parse_args()


def main():
    """Main entry point for the training script."""
    args = parse_args()

    print("=" * 50)
    print(f"PaperTuner: Research Assistant Model Trainer")
    print("=" * 50)

    trainer = ResearchAssistantTrainer(
        model_name=args.model,
        max_seq_length=args.max_seq_length,
        lora_rank=args.lora_rank,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        max_steps=args.max_steps
    )

    # Run training
    training_results = trainer.train(args.dataset)

    # Run demo comparison if not skipped
    if not args.skip_demo:
        trainer.demo_comparison(
            training_results["model"],
            training_results["tokenizer"],
            str(training_results["lora_path"])
        )

    # Push to Hub if requested
    if args.push_to_hf:
        repo_id = args.hub_repo_id or f"{os.getenv('HF_USERNAME', 'user')}/ml-researcher"
        trainer.push_to_hf(
            training_results["model"],
            training_results["tokenizer"],
            repo_id
        )

    print("\n" + "=" * 50)
    print(f"Training completed! Model saved at: {training_results['lora_path']}")
    print("=" * 50)


if __name__ == "__main__":
    main()
