from datasets import load_dataset
from unsloth import FastLanguageModel
from transformers import TrainingArguments
from trl import GRPOTrainer
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO)

# Constants
MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
DATASET_REPO = "densud2/ml_qa_dataset"
OUTPUT_DIR = "output/rl_finetuned"
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

def format_adherence_reward(completions, **kwargs):
    """Reward for following the XML format"""
    responses = [completion[0]["content"] for completion in completions]
    rewards = []
    for response in responses:
        score = 0.0
        if "<reasoning>" in response and "</reasoning>" in response:
            score += 0.5
        if "<approach>" in response and "</approach>" in response:
            score += 0.5
        rewards.append(score)
    return rewards

def technical_reward(completions, **kwargs):
    """Reward for technical content"""
    responses = [completion[0]["content"] for completion in completions]
    technical_terms = ["method", "architecture", "implement", "model", "technique", "process"]
    rewards = []
    for response in responses:
        approach = extract_xml_approach(response)
        count = sum(1 for term in technical_terms if term in approach.lower())
        rewards.append(min(1.0, count / 5))
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
            format_adherence_reward,
            technical_reward,
        ],
        args=training_args,
        train_dataset=dataset,
    )

    # Train
    trainer.train()

    # Save model
    model.save_pretrained_merged(f"{OUTPUT_DIR}/final_model", tokenizer, save_method="lora")

    # Test the model
    sample_question = "How would you design a deep learning model for time series forecasting?"

    text = tokenizer.apply_chat_template([
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": sample_question},
    ], tokenize=False, add_generation_prompt=True)

    from vllm import SamplingParams
    sampling_params = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=512)

    output = model.fast_generate(
        [text],
        sampling_params=sampling_params,
    )[0].outputs[0].text

    print(f"Sample output:\n{output}")

if __name__ == "__main__":
    main()
