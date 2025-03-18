# Fix for the TypeError with reference_column
from datasets import load_dataset
from unsloth import FastLanguageModel
from transformers import TrainingArguments
from trl import GRPOTrainer
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

def composite_reward(completions, references, **kwargs):
    """Composite reward for a list of completions and their corresponding references."""
    rewards = []
    for completion, reference in zip(completions, references):
        response = completion[0]["content"]
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
            "reference": example["answer"]  # Renamed to match what we'll use in the trainer
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
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=32,  # LoRA rank
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=32,
        use_gradient_checkpointing="unsloth",
    )

    # Training arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=3,
        per_device_train_batch_size=8,
        learning_rate=5e-5,
        logging_steps=10,
        save_steps=100,
        remove_unused_columns=False,
    )

    # Initialize GRPO trainer
    # Remove the reference_column parameter and ensure dataset has "reference" column
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[
            composite_reward,
        ],
        args=training_args,
        train_dataset=dataset,
        eval_dataset=dataset,  # Ensure you have an eval dataset if needed
    )

    # Train
    trainer.train()

    # Save model
    model.save_pretrained_merged(f"{OUTPUT_DIR}/final_model", tokenizer, save_method="lora")
    model.save_lora(f"{OUTPUT_DIR}/grpo_saved_lora")

    # Save model with different quantization methods
    quantization_methods = ["f16", "q4_k_m", "q5_k_m", "q8_0"]

    for quant_method in quantization_methods:
        model.save_pretrained_gguf(f"{OUTPUT_DIR}/model_{quant_method}", tokenizer, quantization_method=quant_method)

    # Upload the model to Hugging Face with the best quantization method (e.g., q4_k_m)
    model.push_to_hub_gguf(
        HF_MODEL_NAME,
        tokenizer,
        quantization_method="q4_k_m",
        token=HF_TOKEN
    )

if __name__ == "__main__":
    main()
