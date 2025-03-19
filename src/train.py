from unsloth import FastLanguageModel, is_bfloat16_supported
import torch
import os
import re
from datasets import load_dataset, Dataset
from trl import GRPOConfig, GRPOTrainer
from vllm import SamplingParams
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

max_seq_length = 1024  # Can increase for longer reasoning traces
lora_rank = 64  # Larger rank = smarter, but slower

MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
DATASET_REPO = "densud2/ml_qa_dataset"
OUTPUT_DIR = "output/rl_finetuned"
HF_MODEL_NAME = "username/grpo_finetuned_model"  # Replace 'username' with your Hugging Face username
HF_TOKEN = os.environ.get("HF_TOKEN")  # Your Hugging Face token, get it from https://huggingface.co/settings/tokens

# Load the sentence transformer model for embeddings and similarity calculation
model_embedding = SentenceTransformer('all-mpnet-base-v2')

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=max_seq_length,
    load_in_4bit=True,  # False for LoRA 16bit
    fast_inference=True,  # Enable vLLM fast inference
    max_lora_rank=lora_rank,
    gpu_memory_utilization=0.5,  # Reduce if out of memory
)

model = FastLanguageModel.get_peft_model(
    model,
    r=lora_rank,  # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],  # Remove QKVO if out of memory
    lora_alpha=lora_rank,
    use_gradient_checkpointing="unsloth",  # Enable long context finetuning
    random_state=3407,
)

# System prompt and format definitions
SYSTEM_PROMPT = """
Respond in the following format:
<think>
Write your reasoning step-by-step here. This section should contain your thought process.
</think>
<answer>
Write your final answer here.
</answer>
"""

XML_COT_FORMAT = """\
<think>
{reasoning}
</think>
<answer>
{answer}
</answer>
"""

def extract_xml_answer(text: str) -> str:
    """Extract the answer from XML tags in the text."""
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return answer.strip()

def verify_output_format(text: str) -> bool:
    """Verify if the output follows the expected format with think and answer tags."""
    pattern = r"<think>.*?</think>\s*<answer>.*?</answer>"
    return bool(re.search(pattern, text, re.DOTALL))

# Load ML QA dataset from Hugging Face
def get_ml_qa_dataset(split="train"):
    # Load the dataset from Hugging Face
    data = load_dataset(DATASET_REPO, split=split)

    # Format it for training
    data = data.map(lambda x: {
        'prompt': [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': x['question']}
        ],
        'reference': x['answer']
    })
    return data

# Replace the GSM8K dataset with ML QA dataset
dataset = get_ml_qa_dataset()

# Reward functions
def semantic_similarity_reward_func(prompts=None, completions=None, **kwargs):
    """
    Reward function based on semantic similarity using sentence embeddings.
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

        # Encode the response and reference
        embeddings = model_embedding.encode([response, reference])
        reward = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
        rewards.append(reward)

    return rewards

def think_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion has think tags."""
    pattern = r"<think>.*?</think>"
    responses = [completion[0]["content"] for completion in completions]
    matches = [bool(re.search(pattern, r, re.DOTALL)) for r in responses]
    return [0.5 if match else 0.0 for match in matches]

# Set up GRPO Trainer configurations
training_args = GRPOConfig(
    use_vllm=True,  # Use vLLM for fast inference!
    learning_rate=5e-6,
    adam_beta1=0.9,
    adam_beta2=0.99,
    weight_decay=0.1,
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    optim="adamw_8bit",
    logging_steps=1,
    bf16=is_bfloat16_supported(),
    fp16=not is_bfloat16_supported(),
    per_device_train_batch_size=1,
    gradient_accumulation_steps=1,  # Increase to 4 for smoother training
    num_generations=8,  # Decrease if out of memory
    max_prompt_length=256,
    max_completion_length=200,
    # num_train_epochs=1,  # Set to 1 for a full training run
    max_steps=250,
    save_steps=250,
    max_grad_norm=0.1,
    report_to="none",  # Can use Weights & Biases
    output_dir=OUTPUT_DIR,
)

# Initialize and run the trainer with our custom reward functions
trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=[
        semantic_similarity_reward_func,
        think_format_reward_func,
    ],
    args=training_args,
    train_dataset=dataset,
)
trainer.train()

# Save the LoRA weights
model.save_lora("grpo_saved_lora")

# Test the model with a sample question
text = tokenizer.apply_chat_template([
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": "What are the advantages of using attention mechanisms in neural networks?"},
], tokenize=False, add_generation_prompt=True)

# Generate a response with the fine-tuned model
sampling_params = SamplingParams(
    temperature=0.8,
    top_p=0.95,
    max_tokens=1024,
)
output = model.fast_generate(
    text,
    sampling_params=sampling_params,
    lora_request=model.load_lora("grpo_saved_lora"),
)[0].outputs[0].text

print(output)

# Save to GGUF format if needed
if HF_TOKEN:
    model.push_to_hub_gguf(
        HF_MODEL_NAME,
        tokenizer,
        quantization_method="q4_k_m",
        token=HF_TOKEN,
    )
