from datasets import load_dataset
from unsloth import FastLanguageModel, is_bfloat16_supported
from trl import GRPOConfig, GRPOTrainer
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO)

# Constants
MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
DATASET_REPO = "densud2/ml_qa_dataset"
OUTPUT_DIR = "output/rl_finetuned"
HF_MODEL_NAME = "username/grpo_finetuned_model"  # Replace 'username' with your Hugging Face username
HF_TOKEN = os.environ.get("HF_TOKEN")  # Your Hugging Face token, get it from https://huggingface.co/settings/tokens

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Define system prompt
SYSTEM_PROMPT = """You are an expert PhD-level research assistant that helps researchers design optimal methodologies.
Provide detailed, technical advice that explains both how to implement a methodology and why it's appropriate.
Use <think></think> tags to show your reasoning process, and then provide a clear conclusion."""

def extract_thinking(text):
    if "<think>" not in text or "</think>" not in text:
        return ""
    thinking = text.split("<think>")[-1].split("</think>")[0].strip()
    return thinking

# Initialize the SentenceTransformer model for semantic similarity
st_model = SentenceTransformer('all-mpnet-base-v2')
def semantic_similarity(generated, reference):
    embeddings = st_model.encode([generated, reference])
    similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
    return (similarity + 1) / 2

def verify_output_format(generated):
    """Verify if the generated text contains the required thinking tags."""
    has_thinking = "<think>" in generated and "</think>" in generated
    return has_thinking

# Reward function that counts the presence of think tags
def count_think_tags(text) -> float:
    count = 0.0
    if text.count("<think>") == 1:
        count += 0.25
    if text.count("</think>") == 1:
        count += 0.25
    return count

# Think tags format verification
def think_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion has think tags."""
    pattern = r"<think>.*?</think>"
    import re
    responses = [completion[0]["content"] for completion in completions]
    matches = [bool(re.search(pattern, r, re.DOTALL)) for r in responses]
    return [0.5 if match else 0.0 for match in matches]

# Updated composite reward function that matches GRPOTrainer's expected signature
def composite_reward(prompts=None, completions=None, **kwargs):
    """Reward function that conforms to GRPO's expected interface.

    Args:
        prompts: List of prompts
        completions: List of completions
        **kwargs: Additional keyword arguments

    Returns:
        List of reward scores
    """
    # Get references from the dataset
    train_dataset = kwargs.get('train_dataset', None)
    batch_indices = kwargs.get('batch_indices', None)

    if train_dataset is None or batch_indices is None:
        # Fallback for testing/debugging
        return [0.0] * len(completions)

    # Extract references from the dataset based on batch indices
    references = [train_dataset[idx]['reference'] for idx in batch_indices]

    rewards = []
    for completion, reference in zip(completions, references):
        if not completion or not reference:
            rewards.append(0.0)
            continue

        response = completion[0]["content"] if isinstance(completion[0], dict) else completion[0]

        if not verify_output_format(response):
            rewards.append(0.0)  # Penalize if the output format is incorrect
            continue

        reward = semantic_similarity(response, reference)
        rewards.append(reward)

    return rewards

def load_and_format_dataset():
    # Load dataset from Hugging Face
    try:
        dataset = load_dataset(DATASET_REPO)["train"]
        logging.info(f"Loaded {len(dataset)} examples from HF dataset")
    except Exception as e:
        logging.error(f"Failed to load dataset: {e}")
        raise

    # Format for instruction tuning
    def format_example(example):
        return {
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": example["question"]}
            ],
            "reference": example["answer"]  # Used by the reward function
        }

    return dataset.map(format_example)

# Main training function
def main():
    # Load dataset
    dataset = load_and_format_dataset()

    # Initialize model with LoRA
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=1024,
        load_in_4bit=True,
        fast_inference=True,
        max_lora_rank=32,
        gpu_memory_utilization=0.6,  # Reduce if OOM
    )

    # Configure LoRA
    model = FastLanguageModel.get_peft_model(
        model,
        r=32,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=32,
        use_gradient_checkpointing="unsloth",  # Enable long context finetuning
        random_state=3407,
    )

    # Training arguments using GRPOConfig directly
    max_prompt_length = 256
    training_args = GRPOConfig(
        use_vllm=True,  # This is the key fix for Qwen models!
        learning_rate=5e-6,
        adam_beta1=0.9,
        adam_beta2=0.99,
        weight_decay=0.1,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        optim="paged_adamw_8bit",
        logging_steps=10,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        num_generations=4,  # Decrease if out of memory
        max_prompt_length=max_prompt_length,
        max_completion_length=1024 - max_prompt_length,
        max_steps=250,
        save_steps=100,
        max_grad_norm=0.1,
        report_to="none",
        output_dir=OUTPUT_DIR,
        bf16=is_bfloat16_supported(),
        fp16=not is_bfloat16_supported(),
    )

    # Initialize GRPO trainer directly
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[
            composite_reward,
            think_format_reward_func,  # Added to explicitly reward <think> tags
        ],
        args=training_args,
        train_dataset=dataset,
    )

    # Train
    trainer.train()

    # Save the LoRA first
    model.save_lora(f"{OUTPUT_DIR}/grpo_saved_lora")

    # Save model with different quantization methods
    model.save_pretrained_merged(f"{OUTPUT_DIR}/final_model", tokenizer, save_method="lora")

    # Save model with different quantization methods
    quantization_methods = ["f16", "q4_k_m", "q5_k_m", "q8_0"]

    for quant_method in quantization_methods:
        model.save_pretrained_gguf(f"{OUTPUT_DIR}/model_{quant_method}", tokenizer, quantization_method=quant_method)

    # Upload the model to Hugging Face with the best quantization method (e.g., q4_k_m)
    if HF_TOKEN:
        model.push_to_hub_gguf(
            HF_MODEL_NAME,
            tokenizer,
            quantization_method="q4_k_m",
            token=HF_TOKEN
        )

if __name__ == "__main__":
    main()
