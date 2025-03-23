"""Simplified GRPO Training Script for ML Research Assistant"""
print("Welcome to the ML-Research Assistant Trainer!")
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
MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
MAX_SEQ_LENGTH = 1024
LORA_RANK = 64
SYSTEM_PROMPT = """You are an AI assistant specialized in machine learning concepts. Follow this response format:
<think>
First, think through the question step-by-step in this section.
Consider what the user is asking, relevant concepts, and how to structure your answer.
This section should contain your analytical process and reasoning.
</think>

After the think section, provide your direct answer without any tags.
Your answer should be clear, concise, and directly address the question.
"""

# --- Model Loading ---
def load_model():
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = MODEL_NAME,
        max_seq_length = MAX_SEQ_LENGTH,
        load_in_4bit = True, # False for LoRA 16bit
        fast_inference = True, # Enable vLLM fast inference
        max_lora_rank = LORA_RANK,
        gpu_memory_utilization = 0.5, # Reduce if out of memory
    )

    return model, tokenizer, FastLanguageModel.get_peft_model(
        model,
        r = LORA_RANK, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
        target_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ], # Remove QKVO if out of memory
        lora_alpha = LORA_RANK,
        use_gradient_checkpointing = "unsloth", # Enable long context finetuning
        random_state = 3407,
    )

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

def extract_answer(text):
    # Extract content after the <think> tag
    return text.split("</think>")[-1].strip() if "</think>" in text else text

def correctness_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
    responses = [completion[0]['content'] for completion in completions]
    q = prompts[0][-1]['content']

    # Extract answers from responses and ground truth
    extracted_responses = [extract_answer(r) for r in responses]
    extracted_answer = extract_answer(answer[0])

    # Compute embeddings
    response_embeddings = embedding_model.encode(extracted_responses, convert_to_tensor=True)
    answer_embedding = embedding_model.encode([extracted_answer], convert_to_tensor=True)

    # Compute similarities and rewards
    rewards = []
    for resp_emb in response_embeddings:
        sim = util.pytorch_cos_sim(resp_emb.unsqueeze(0), answer_embedding).item()
        rewards.append(sim * 2)

    print('-'*20)
    print(f"Question:\n{q}")
    print(f"\nAnswer:\n{extracted_answer}")
    print(f"\nResponse:\n{extracted_responses[0]}")
    print(f"\nSimilarity Score: {rewards[0]/2:.4f}")
    print(f"Reward: {rewards[0]:.4f}")

    return rewards

# --- Training Setup ---
def get_trainer(model, tokenizer, dataset):
    return GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[
            correctness_reward_func
        ],
        args=GRPOConfig(
            use_vllm = True,                             # Enable faster inference using vLLM
            learning_rate = 5e-6,                        # Small learning rate for stable training
            adam_beta1 = 0.9,                            # AdamW optimizer momentum parameter
            adam_beta2 = 0.99,                           # AdamW optimizer second moment parameter
            weight_decay = 0.1,                          # L2 regularization to prevent overfitting
            warmup_ratio = 0.1,                          # Portion of training steps for learning rate warmup
            lr_scheduler_type = "cosine",                # Learning rate decay schedule type
            optim = "adamw_8bit",                        # Use 8-bit AdamW optimizer for memory efficiency
            logging_steps = 1,                           # Log metrics every step
            bf16 = is_bfloat16_supported(),              # Use bfloat16 if hardware supports it
            fp16 = not is_bfloat16_supported(),          # Fallback to float16 if bfloat16 not supported
            per_device_train_batch_size = 1,             # Number of prompts per GPU
            gradient_accumulation_steps = 4,             # Number of steps to accumulate gradients
            num_generations = 8,                         # Number of responses to generate per prompt for GRPO
            max_prompt_length = 256,                     # Maximum length of input prompts in tokens
            max_completion_length = MAX_SEQ_LENGTH - 256, # Maximum length of model responses in tokens
            max_steps = 2000,                           # Total number of training steps
            save_steps = 250,                           # Save checkpoint every 250 steps
            max_grad_norm = 0.1,                        # Gradient clipping threshold
            report_to = "none",                         # Log metrics to Weights & Biases
            output_dir = "outputs",                     # Directory to save model checkpoints
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
    model, tokenizer, model = load_model()

    # 3. Initialize trainer
    print("Initializing trainer...")
    trainer = get_trainer(model, tokenizer, dataset)

    # 4. Run training
    print("Starting training...")
    trainer.train()

    # 5. Save trained LoRA weights
    print("Saving LoRA adapter...")
    model.save_lora("ml_assistant_lora")

    # 6. Demo inference comparisons
    def run_inference(question, lora=None):
        prompt = tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": question}
                ],
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
    print(run_inference(test_question, "ml_assistant_lora"))

    # 7. Optional: Save model to HuggingFace
    if "HG_TOKEN" in os.environ:
        model.push_to_hub_gguf(
                "densud2/ML-researcher",
                tokenizer,
                quantization_method = ["q4_k_m", "q8_0", "q5_k_m"],
                token = os.environ["HG_TOKEN"],
                )
    else:
        print("HG_TOKEN environment variable not found. Skipping HuggingFace upload.")
