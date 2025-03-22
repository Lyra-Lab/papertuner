"""Simplified GRPO Training Script for QA Model"""
print("Welcome to the ML-QA Trainer!")
# Imports
import re
import os
import torch
import datasets
from sentence_transformers import SentenceTransformer, util
from trl import GRPOConfig, GRPOTrainer
from unsloth import FastLanguageModel, is_bfloat16_supported
from vllm import SamplingParams

# Constants
MODEL_NAME = "unsloth/DeepSeek-R1-Distill-Qwen-1.5B"
MAX_SEQ_LENGTH = 8192
LORA_RANK = 64
SYSTEM_PROMPT = "Respond in the format:\n<reasoning>...</reasoning>\n<answer>...</answer>"

# --- Model Loading ---
def load_model():
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=True,
        fast_inference=True,
        max_lora_rank=LORA_RANK,
        gpu_memory_utilization=0.5,
    )

    return FastLanguageModel.get_peft_model(
        model,
        r=LORA_RANK,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=LORA_RANK,
        use_gradient_checkpointing="unsloth",
        random_state=7,
    ), tokenizer

# --- Data Preparation ---
def load_dataset():
    dataset = datasets.load_dataset("densud2/ml_qa_dataset", split="train")

    def format_example(x):
        return {
            'prompt': [
                {'role': 'system', 'content': SYSTEM_PROMPT},
                {'role': 'user', 'content': x['question']}
            ],
            'answer': x['answer']
        }

    return dataset.map(format_example)

# --- Reward Functions ---
embedding_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

def extract_content(text, tag):
    if f"<{tag}>" in text and f"</{tag}>" in text:
        return text.split(f"<{tag}>")[-1].split(f"</{tag}>")[0].strip()
    return ""

def correctness_reward(prompts, completions, **kwargs):
    """
    Computes a reward based on the similarity between the generated answer
    and the expected answer. The expected answers are passed via the keyword
    'answer' (as produced by your dataset).
    """
    answers = kwargs.get("answer")
    if answers is None:
        raise ValueError("Missing 'answer' keyword argument in reward function.")
    rewards = []
    for completion, answer in zip(completions, answers):
        response = completion[0]['content']
        extracted = extract_content(response, "answer") or response.split("</think>")[-1].strip()
        embeds = embedding_model.encode([extracted, answer], convert_to_tensor=True)
        rewards.append(util.pytorch_cos_sim(embeds[0], embeds[1]).item() * 2)
    return rewards

# Modified to accept the prompts parameter and extra kwargs
def format_reward(prompts, completions, pattern, **kwargs):
    return [0.5 if re.search(pattern, c[0]["content"], re.DOTALL) else 0.0 for c in completions]

# Modified to accept the prompts parameter and extra kwargs
def int_reward(prompts, completions, **kwargs):
    return [0.5 if extract_content(c[0]["content"], "answer").isdigit() else 0.0 for c in completions]

# --- Training Setup ---
def get_trainer(model, tokenizer, dataset):
    return GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[
            # Fixed lambda functions to properly pass prompts parameter
            lambda prompts, completions, **kwargs: format_reward(prompts, completions, r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>"),
            lambda prompts, completions, **kwargs: format_reward(prompts, completions, r"<think>.*?</think>.*"),
            lambda prompts, completions, **kwargs: int_reward(prompts, completions),
            correctness_reward,
        ],
        args=GRPOConfig(
          use_vllm = True, # use vLLM for fast inference!
          learning_rate = 5e-6,
          adam_beta1 = 0.9,
          adam_beta2 = 0.99,
          weight_decay = 0.1,
          warmup_ratio = 0.1,
          lr_scheduler_type = "cosine",
          optim = "adamw_8bit",
          logging_steps = 1,
          bf16 = is_bfloat16_supported(),
          fp16 = not is_bfloat16_supported(),
          per_device_train_batch_size = 1,
          gradient_accumulation_steps = 8, # Increase to 4 for smoother training
          num_generations = 8, # Decrease if out of memory
          max_prompt_length = 256,
          max_completion_length = 200,
          # num_train_epochs = 1, # Set to 1 for a full training run
          max_steps = 250,
          save_steps = 250,
          max_grad_norm = 0.1,
          report_to = "none", # Can use Weights & Biases
          output_dir = "outputs",
        ),
        train_dataset=dataset,  
    )

# --- Main Execution ---
if __name__ == "__main__":

    # 1. Load and prepare dataset
    print("Loading dataset...")
    dataset = load_dataset()

    # 2. Initialize model and tokenizer
    print("Loading model and tokenizer...")
    model, tokenizer = load_model()

    # 3. Initialize trainer
    print("Initializing trainer...")
    trainer = get_trainer(model, tokenizer, dataset)

    # 4. Run training
    print("Starting training...")
    trainer.train()

    # 5. Save trained LoRA weights
    print("Saving LoRA adapter...")
    model.save_lora("trained_lora")

    # 6. Demo inference comparisons
    def run_inference(question, lora=None):
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": question}],
            tokenize=False,
            add_generation_prompt=True
        )
        return model.fast_generate(
            [prompt],
            sampling_params=SamplingParams(
                temperature=0.8,
                top_p=0.95,
                max_tokens=1024,
            ),
            lora_request=model.load_lora(lora) if lora else None,
        )[0].outputs[0].text

    test_question = "How many r's are in strawberry?"

    print("\n=== Pre-training Response ===")
    print(run_inference(test_question))

    print("\n=== Post-training Response ===")
    print(run_inference(test_question, "trained_lora"))

    # 7. Optional: Save full model
    model.push_to_hub_gguf(
        "densud2/ML-assistant",
        tokenizer,
        quantization_method = ["q4_k_m", "q8_0", "q5_k_m",],
        token = os.environ["HF_TOKEN"],
    )
