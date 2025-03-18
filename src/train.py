import json
import logging
from pathlib import Path
from tqdm import tqdm
from typing import List
from datasets import load_dataset, Dataset
from unsloth import FastLanguageModel
from transformers import TrainingArguments
from trl import GRPOTrainer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Constants and configuration
MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
DATASET_REPO = "your-username/research-methodology-qa"  # Replace with your actual dataset repo
OUTPUT_DIR = Path("output/rl_finetuned")
MAX_SEQ_LENGTH = 1024
LORA_RANK = 32
BATCH_SIZE = 8
LEARNING_RATE = 5e-5
NUM_TRAIN_EPOCHS = 3
SAVE_STEPS = 100
LOGGING_STEPS = 10

# Create output directory
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Define the system prompt
SYSTEM_PROMPT = """You are an expert PhD-level research assistant that helps researchers design optimal methodologies for their research problems.
Provide detailed, technical and practical advice that explains both how to implement a methodology and why it's appropriate.
Your response should be thorough and scientifically sound."""

# XML-style format for structured responses
XML_FORMAT = """\
<reasoning>
{reasoning}
</reasoning>
<approach>
{approach}
</approach>
"""

def extract_approach(text: str) -> str:
    """Extract the approach section from the XML-formatted response."""
    if "<approach>" not in text or "</approach>" not in text:
        return ""

    approach = text.split("<approach>")[-1]
    approach = approach.split("</approach>")[0]
    return approach.strip()

def extract_reasoning(text: str) -> str:
    """Extract the reasoning section from the XML-formatted response."""
    if "<reasoning>" not in text or "</reasoning>" not in text:
        return ""

    reasoning = text.split("<reasoning>")[-1]
    reasoning = reasoning.split("</reasoning>")[0]
    return reasoning.strip()

def load_research_qa_dataset() -> Dataset:
    """Load and prepare the research methodology QA dataset."""
    try:
        # Try loading from Hugging Face Hub
        data = load_dataset(DATASET_REPO)
        logging.info(f"Loaded dataset from {DATASET_REPO}")
    except Exception as e:
        logging.warning(f"Failed to load dataset from HF Hub: {e}")
        # Fallback to local dataset
        logging.info("Attempting to load dataset from local files...")
        data_files = list(Path("data/processed_dataset/papers").glob("paper_*.json"))

        if not data_files:
            raise ValueError("No local dataset files found. Please run dataset.py first.")

        examples = []
        for file in tqdm(data_files, desc="Loading local files"):
            try:
                with open(file, 'r') as f:
                    paper_data = json.load(f)

                    # Handle multiple QA pairs in newer format
                    if "qa_pairs" in paper_data and isinstance(paper_data["qa_pairs"], list):
                        for qa in paper_data["qa_pairs"]:
                            if qa.get("question") and qa.get("answer"):
                                examples.append({
                                    "question": qa["question"],
                                    "answer": qa["answer"],
                                    "category": qa.get("category", "General")
                                })
                    # Handle legacy single QA format
                    elif "qa" in paper_data and paper_data["qa"].get("question") and paper_data["qa"].get("answer"):
                        examples.append({
                            "question": paper_data["qa"]["question"],
                            "answer": paper_data["qa"]["answer"],
                            "category": "General"
                        })
            except json.JSONDecodeError:
                logging.warning(f"Skipping invalid JSON file: {file}")

        data = Dataset.from_list(examples)
        logging.info(f"Loaded {len(examples)} examples from local files")

    # Format the dataset for instruction tuning
    def format_for_chat(example):
        return {
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": example["question"]}
            ],
            "original_answer": example["answer"]
        }

    formatted_data = data.map(format_for_chat)
    logging.info(f"Dataset prepared with {len(formatted_data)} examples")

    return formatted_data

# Reward functions
def format_adherence_reward(completions, **kwargs) -> List[float]:
    """Reward function that checks if the completion follows the expected format."""
    responses = [completion[0]["content"] for completion in completions]

    rewards = []
    for response in responses:
        score = 0.0
        # Check if format has reasoning and approach sections
        if "<reasoning>" in response and "</reasoning>" in response:
            score += 0.5
        if "<approach>" in response and "</approach>" in response:
            score += 0.5
        rewards.append(score)

    return rewards

def technical_content_reward(completions, **kwargs) -> List[float]:
    """Reward function that checks for technical content and specific implementation details."""
    responses = [completion[0]["content"] for completion in completions]

    technical_keywords = [
        "algorithm", "method", "architecture", "implement", "parameter",
        "technique", "framework", "model", "structure", "design",
        "approach", "procedure", "system", "process", "analysis",
        "data", "evaluation", "metric", "step", "component"
    ]

    rewards = []
    for response in responses:
        approach = extract_approach(response)
        # Count the number of technical keywords in the approach
        keyword_count = sum(1 for keyword in technical_keywords if keyword in approach.lower())
        # Normalize the reward based on keyword count
        score = min(1.0, keyword_count / 10)
        rewards.append(score)

    return rewards

def reasoning_quality_reward(completions, **kwargs) -> List[float]:
    """Reward function that evaluates the quality of reasoning."""
    responses = [completion[0]["content"] for completion in completions]

    reasoning_markers = [
        "because", "therefore", "thus", "consequently",
        "as a result", "due to", "since", "leads to",
        "enables", "advantage", "benefit", "limitation",
        "trade-off", "comparison", "alternatively"
    ]

    rewards = []
    for response in responses:
        reasoning = extract_reasoning(response)
        marker_count = sum(1 for marker in reasoning_markers if marker in reasoning.lower())
        # Normalize the reward based on marker count
        score = min(1.0, marker_count / 5)
        rewards.append(score)

    return rewards

def length_and_detail_reward(completions, **kwargs) -> List[float]:
    """Reward function that considers the length and detail of the response."""
    responses = [completion[0]["content"] for completion in completions]

    rewards = []
    for response in responses:
        approach = extract_approach(response)
        reasoning = extract_reasoning(response)

        # A good response should have both sections with reasonable length
        approach_length = len(approach.split())
        reasoning_length = len(reasoning.split())

        # Calculate score based on length (diminishing returns beyond certain length)
        approach_score = min(0.5, approach_length / 200)
        reasoning_score = min(0.5, reasoning_length / 200)

        rewards.append(approach_score + reasoning_score)

    return rewards

def combined_reward(completions, prompts=None, **kwargs) -> List[float]:
    """Combined reward function that aggregates all reward components."""
    format_scores = format_adherence_reward(completions)
    technical_scores = technical_content_reward(completions)
    reasoning_scores = reasoning_quality_reward(completions)
    length_scores = length_and_detail_reward(completions)

    # Combine all rewards with weights
    combined_scores = [
        0.3 * format_score +
        0.3 * technical_score +
        0.2 * reasoning_score +
        0.2 * length_score
        for format_score, technical_score, reasoning_score, length_score
        in zip(format_scores, technical_scores, reasoning_scores, length_scores)
    ]

    # Log some example rewards for debugging
    if prompts is not None and len(prompts) > 0:
        sample_prompt = prompts[0][-1]['content']
        sample_response = completions[0][0]['content']
        sample_scores = {
            'format': format_scores[0],
            'technical': technical_scores[0],
            'reasoning': reasoning_scores[0],
            'length': length_scores[0],
            'combined': combined_scores[0]
        }
        logging.info(f"Sample prompt: {sample_prompt[:100]}...")
        logging.info(f"Sample response: {sample_response[:100]}...")
        logging.info(f"Sample scores: {sample_scores}")

    return combined_scores

def main():
    logging.info(f"Starting RL fine-tuning with model: {MODEL_NAME}")

    # Load dataset
    dataset = load_research_qa_dataset()

    # Initialize the model with LoRA
    logging.info("Initializing model with LoRA...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=True,
        fast_inference=True,
        max_lora_rank=LORA_RANK,
        gpu_memory_utilization=0.8,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_RANK,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=LORA_RANK,
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )

    # Set up training arguments
    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        num_train_epochs=NUM_TRAIN_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=4,
        learning_rate=LEARNING_RATE,
        weight_decay=0.01,
        logging_steps=LOGGING_STEPS,
        save_steps=SAVE_STEPS,
        save_total_limit=3,
        log_level="info",
        remove_unused_columns=False,
        push_to_hub=False,
    )

    # Initialize GRPO trainer
    logging.info("Initializing GRPO trainer...")
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[
            format_adherence_reward,
            technical_content_reward,
            reasoning_quality_reward,
            length_and_detail_reward,
            combined_reward,
        ],
        args=training_args,
        train_dataset=dataset,
    )

    # Start training
    logging.info("Starting training...")
    trainer.train()

    # Save the model
    logging.info("Saving fine-tuned model...")
    model.save_pretrained_merged(str(OUTPUT_DIR / "final_model"), tokenizer, save_method="lora")

    # Generate a sample inference
    sample_question = "How would you design an effective attention mechanism for a transformer model that needs to handle very long sequences?"

    text = tokenizer.apply_chat_template([
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": sample_question},
    ], tokenize=False, add_generation_prompt=True)

    from vllm import SamplingParams
    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.9,
        max_tokens=512,
    )

    logging.info("Generating sample output...")
    output = model.fast_generate(
        [text],
        sampling_params=sampling_params,
        lora_request=None,
    )[0].outputs[0].text

    logging.info(f"Sample output:\n{output}")

    logging.info("Fine-tuning complete!")

if __name__ == "__main__":
    main()
