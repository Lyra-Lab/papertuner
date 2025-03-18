from datasets import load_dataset
from unsloth import FastLanguageModel
from transformers import TrainingArguments
from trl import GRPOTrainer
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO)

# Constants
MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
DATASET_REPO = "densud2/ml_qa_dataset"
OUTPUT_DIR = "output/rl_finetuned"
HF_MODEL_NAME = "username/grpo_finetuned_model"  # Replace 'username' with your Hugging Face username
HF_TOKEN = ""  # Your Hugging Face token, get it from https://huggingface.co/settings/tokens

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Define system prompt
SYSTEM_PROMPT = """You are an expert PhD-level research assistant that helps researchers design optimal methodologies.
Provide detailed, technical advice that explains both how to implement a methodology and why it's appropriate.
Structure your response with clear reasoning and a specific approach."""

def extract_xml_approach(text):
    if "<approach>" not in text or "</approach>" not in text:
        return ""
    approach = text.split("<approach>")[-1].split("</approach>")[0].strip()
    return approach

def extract_xml_reasoning(text):
    if "<reasoning>" not in text or "</reasoning>" not in text:
        return ""
    reasoning = text.split("<reasoning>")[-1].split("</reasoning>")[0].strip()
    return reasoning

# Initialize models (do this once)
st_model = SentenceTransformer('all-mpnet-base-v2')
nli_model = AutoModelForSequenceClassification.from_pretrained('roberta-large-mnli')
nli_tokenizer = AutoTokenizer.from_pretrained('roberta-large-mnli')

def semantic_similarity(generated, reference):
    embeddings = st_model.encode([generated, reference])
    similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
    return (similarity + 1) / 2

def factual_verification(reference, generated):
    inputs = nli_tokenizer(reference, generated, return_tensors='pt', truncation=True, max_length=512)
    with torch.no_grad():
        outputs = nli_model(**inputs)
    probs = torch.softmax(outputs.logits, dim=1)
    return probs[:, 2].item()

def calculate_reward(generated, reference):
    sem_sim = semantic_similarity(generated, reference)
    fact_ver = factual_verification(reference, generated)
    return 0.6 * sem_sim + 0.4 * fact_ver

def composite_reward(completions, references, **kwargs):
    """Composite reward for a list of completions and their corresponding references."""
    rewards = []
    for completion, reference in zip(completions, references):
        response = completion[0]["content"]
        reward = calculate_reward(response, reference)
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
            "original_answer": example["answer"]
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
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[
            composite_reward,
        ],
        args=training_args,
        train_dataset=dataset,
        eval_dataset=dataset,  # Ensure you have an eval dataset if needed
        reference_column="original_answer",  # Add reference column for reward calculation
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
