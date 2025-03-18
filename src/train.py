# src/train.py
import os
import logging
import torch
from pathlib import Path
from datasets import load_dataset
from unsloth import FastLanguageModel
from trl import GRPOConfig, GRPOTrainer
import re

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Constants
MODEL_OUTPUT_DIR = Path("data/trained_model")
LORA_OUTPUT_DIR = Path("data/lora_weights")
DATASET_ID = "densud2/ml_qa_dataset"  # Your published dataset

def setup_dirs():
    """Create necessary directories for model training."""
    try:
        MODEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        LORA_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        logging.info("Training directories set up successfully.")
        return True
    except OSError as e:
        logging.error(f"Failed to setup training directories: {e}")
        return False

def load_dataset_from_huggingface():
    """Load the research methodology dataset from Hugging Face."""
    try:
        # Load the dataset directly from Hugging Face
        dataset = load_dataset(DATASET_ID)
        logging.info(f"Successfully loaded dataset from {DATASET_ID}")

        # If the dataset has splits, get the training split
        if isinstance(dataset, dict) and "train" in dataset:
            dataset = dataset["train"]
            logging.info(f"Using train split with {len(dataset)} examples")

        # Define system prompt for the research assistant
        SYSTEM_PROMPT = """
        You are a PhD-level research assistant specialized in helping researchers design optimal methodologies.
        Given a research problem, literature review, and hypothesis, predict the most appropriate methodology.

        Respond in the following format:
        <reasoning>
        Step through your thought process, considering multiple methodological approaches, their advantages,
        disadvantages, and appropriateness for the specific research context. Analyze potential confounding
        variables and how different methods might address them.
        </reasoning>
        <approach>
        Clearly state the recommended methodology with justification.
        </approach>
        """

        # Transform the dataset into the format required for GRPO training
        formatted_dataset = []

        for item in dataset:
            # Format user content from question and any additional context
            question = item["question"]

            # Create a structured problem statement from the question
            user_content = f"Problem: {question}\n\n"

            # Add any additional context if available in the dataset
            if "paper_title" in item:
                user_content += f"Research Context: {item['paper_title']}\n\n"

            if "categories" in item:
                categories = item["categories"]
                if isinstance(categories, list) and categories:
                    user_content += f"Domain: {', '.join(categories)}\n\n"

            # Add a simple hypothesis section if not already in the question
            if "hypothesis" not in question.lower():
                user_content += "Hypothesis: Based on the research context, we aim to determine the most effective approach.\n\n"

            # Use the answer as the correct approach
            correct_approach = item["answer"]

            formatted_dataset.append({
                "prompt": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_content}
                ],
                "correct_approach": correct_approach
            })

        logging.info(f"Created formatted dataset with {len(formatted_dataset)} training examples")
        return formatted_dataset

    except Exception as e:
        logging.error(f"Error loading dataset from Hugging Face: {e}")
        raise

def extract_approach(text: str) -> str:
    """Extract approach from formatted model response."""
    if "<approach>" not in text or "</approach>" not in text:
        return ""
    approach = text.split("<approach>")[-1]
    approach = approach.split("</approach>")[0]
    return approach.strip()

def approach_similarity_reward(completions, correct_approach, **kwargs) -> list[float]:
    """Reward function to evaluate approach similarity."""
    responses = [completion[0]['content'] for completion in completions]
    extracted_approaches = [extract_approach(r) for r in responses]

    rewards = []
    for approach in extracted_approaches:
        if len(approach) > 50:  # Basic length check
            # Calculate similarity using word overlap
            overlap = len(set(approach.lower().split()) & set(correct_approach.lower().split()))
            total = len(set(approach.lower().split()) | set(correct_approach.lower().split()))
            similarity = overlap / total if total > 0 else 0
            rewards.append(similarity * 2.0)  # Scale to 0-2 range
        else:
            rewards.append(0.0)

    # Print example for debugging (only occasionally to avoid log spam)
    if rewards and kwargs.get("step", 0) % 20 == 0:
        logging.info(f"Sample reward: {rewards[0]:.4f}")
        logging.info(f"Sample approach (extract): {extracted_approaches[0][:100]}...")

    return rewards

def format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function to check if response follows required XML format."""
    pattern = r"<reasoning>.*?</reasoning>\s*<approach>.*?</approach>"
    responses = [completion[0]["content"] for completion in completions]
    matches = [bool(re.search(pattern, r, re.DOTALL)) for r in responses]
    return [0.5 if match else 0.0 for match in matches]

def reasoning_depth_reward(completions, **kwargs) -> list[float]:
    """Reward function to evaluate thoroughness of reasoning."""
    responses = [completion[0]["content"] for completion in completions]
    rewards = []

    for response in responses:
        # Extract reasoning section
        if "<reasoning>" not in response or "</reasoning>" not in response:
            rewards.append(0.0)
            continue

        reasoning = response.split("<reasoning>")[-1].split("</reasoning>")[0].strip()

        # Calculate reward based on reasoning depth indicators
        depth_score = 0.0

        # Check for multiple approaches consideration
        if "advantage" in reasoning.lower() and "disadvantage" in reasoning.lower():
            depth_score += 0.2

        # Check for methodological comparisons
        if any(m in reasoning.lower() for m in ["compared to", "versus", "alternative", "instead of"]):
            depth_score += 0.1

        # Check for consideration of confounding variables
        if any(m in reasoning.lower() for m in ["confound", "control for", "account for", "limitation"]):
            depth_score += 0.2

        # Reward for longer reasoning (with a reasonable cap)
        word_count = len(reasoning.split())
        if word_count > 100:
            depth_score += min((word_count - 100) / 400, 0.5)  # Max 0.5 for length

        rewards.append(depth_score)

    return rewards

def train_model(dataset_items):
    """Train the research assistant model using GRPO."""
    logging.info("Initializing model training")

    # Convert list to Dataset object
    from datasets import Dataset
    dataset = Dataset.from_list(dataset_items)

    # Model configuration
    max_seq_length = 2048  # Longer for scientific reasoning traces
    lora_rank = 32  # Higher rank for more complex reasoning

    # Load the base model - DeepSeek-R1 (already has reasoning capabilities)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "deepseek-ai/deepseek-r1-1.5b",  # Using DeepSeek-R1
        max_seq_length = max_seq_length,
        load_in_4bit = True,  # Use 4bit quantization to save memory
        fast_inference = True,  # Enable vLLM fast inference
        max_lora_rank = lora_rank,
        gpu_memory_utilization = 0.6,  # Adjust based on your GPU
    )

    # Set up LoRA for efficient fine-tuning
    model = FastLanguageModel.get_peft_model(
        model,
        r = lora_rank,
        target_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha = lora_rank,
        use_gradient_checkpointing = "unsloth",  # Enable long context fine-tuning
        random_state = 3407,
    )

    # Set up training arguments
    max_prompt_length = 512  # Longer prompt length for scientific context

    # Define the number of training steps based on dataset size
    num_epochs = 3
    total_steps = len(dataset) * num_epochs
    max_steps = min(total_steps, 1000)  # Cap at 1000 steps

    training_args = GRPOConfig(
        learning_rate = 3e-6,  # Slightly lower for more stable training
        adam_beta1 = 0.9,
        adam_beta2 = 0.99,
        weight_decay = 0.05,
        warmup_ratio = 0.1,
        lr_scheduler_type = "cosine",
        optim = "paged_adamw_8bit",
        logging_steps = 10,
        per_device_train_batch_size = 1,
        gradient_accumulation_steps = 2,  # Increase for stability
        num_generations = 4,  # Balance between diversity and memory
        max_prompt_length = max_prompt_length,
        max_completion_length = max_seq_length - max_prompt_length,
        # num_train_epochs = num_epochs,
        max_steps = max_steps,
        save_steps = 100,
        max_grad_norm = 0.3,  # More aggressive gradient clipping
        report_to = "none",  # Can use "wandb" if you want to track with Weights & Biases
        output_dir = str(MODEL_OUTPUT_DIR),
    )

    # Initialize and run GRPO trainer
    trainer = GRPOTrainer(
        model = model,
        processing_class = tokenizer,
        reward_funcs = [
            format_reward_func,         # 0.5 for correct format
            reasoning_depth_reward,     # Up to 1.0 for thorough reasoning
            approach_similarity_reward, # Up to 2.0 for similarity to correct approach
        ],
        args = training_args,
        train_dataset = dataset,
    )

    logging.info(f"Starting training for {max_steps} steps")
    trainer.train()

    # Save the trained model
    lora_path = str(LORA_OUTPUT_DIR / "research_assistant_grpo_lora")
    model.save_lora(lora_path)
    logging.info(f"LoRA weights saved to {lora_path}")

    # Save to GGUF format for deployment
    gguf_path = str(MODEL_OUTPUT_DIR / "research_assistant_gguf")
    model.save_pretrained_gguf(
        gguf_path,
        tokenizer,
        quantization_method = "q5_k_m"  # Good balance of quality and size
    )
    logging.info(f"GGUF model saved to {gguf_path}")

    return {
        "model": model,
        "tokenizer": tokenizer,
        "lora_path": lora_path,
        "gguf_path": gguf_path
    }

def test_model(model_info, test_query):
    """Test the trained model with a sample query."""
    model = model_info["model"]
    tokenizer = model_info["tokenizer"]
    lora_path = model_info["lora_path"]

    # System prompt for consistent formatting
    SYSTEM_PROMPT = """
    You are a PhD-level research assistant specialized in helping researchers design optimal methodologies.
    Given a research problem, literature review, and hypothesis, predict the most appropriate methodology.

    Respond in the following format:
    <reasoning>
    Step through your thought process, considering multiple methodological approaches, their advantages,
    disadvantages, and appropriateness for the specific research context. Analyze potential confounding
    variables and how different methods might address them.
    </reasoning>
    <approach>
    Clearly state the recommended methodology with justification.
    </approach>
    """

    text = tokenizer.apply_chat_template([
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": test_query},
    ], tokenize=False, add_generation_prompt=True)

    try:
        from vllm import SamplingParams
        sampling_params = SamplingParams(
            temperature=0.7,
            top_p=0.9,
            max_tokens=1536,
        )
        output = model.fast_generate(
            text,
            sampling_params=sampling_params,
            lora_request=model.load_lora(lora_path),
        )[0].outputs[0].text

        return output
    except Exception as e:
        logging.error(f"Error during test inference: {e}")
        return "Error generating response"

def main():
    """Main function to train the research assistant model."""
    if not setup_dirs():
        logging.error("Failed to set up directories. Exiting.")
        return

    try:
        # Load dataset directly from Hugging Face
        dataset_items = load_dataset_from_huggingface()

        # Train the model
        model_info = train_model(dataset_items)

        # Test the model with a sample query
        test_query = """
        Problem: Examining the effectiveness of different feedback mechanisms on student learning in online programming courses.

        Literature Review: Studies indicate feedback timing, specificity, and delivery method impact learning outcomes. Automated feedback systems have shown promise, but questions remain about their effectiveness compared to human feedback for complex programming concepts.

        Hypothesis: Students receiving a combination of immediate automated feedback and delayed human expert feedback will demonstrate greater improvement in code quality and conceptual understanding compared to students receiving only one type of feedback.
        """

        result = test_model(model_info, test_query)
        logging.info(f"Test result:\n{result}")

        logging.info("Training and testing complete!")

    except Exception as e:
        logging.error(f"Training process failed: {e}")

if __name__ == "__main__":
    main()
