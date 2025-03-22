"""Unsloth-based GRPO Training Pipeline for QA Model"""

# Standard Library Imports
import re
from typing import List, Optional

# Third-party Imports
import torch
from datasets import Dataset, load_dataset
from sentence_transformers import SentenceTransformer, util
from trl import GRPOConfig, GRPOTrainer
from unsloth import FastLanguageModel, is_bfloat16_supported
from vllm import SamplingParams

# Constants
MODEL_NAME = "unsloth/DeepSeek-R1-Distill-Qwen-1.5B-unsloth-bnb-4bit"
MAX_SEQ_LENGTH = 8192
LORA_RANK = 64
SYSTEM_PROMPT = """Respond in the following format:
<reasoning>...</reasoning>
<answer>...</answer>"""

class ModelManager:
    """Handles model loading and configuration"""

    @staticmethod
    def load_model_and_tokenizer():
        """Initialize model and tokenizer with optimal settings"""
        device = "cuda" if torch.cuda.is_available() else "cpu"

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=MODEL_NAME,
            max_seq_length=MAX_SEQ_LENGTH,
            load_in_4bit=True,
            fast_inference=True,
            max_lora_rank=LORA_RANK,
            gpu_memory_utilization=0.5,
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
        return model, tokenizer

class DataHandler:
    """Handles dataset loading and preprocessing"""

    @staticmethod
    def load_qa_dataset(split: str = "train") -> Dataset:
        """Load and format the QA dataset"""
        dataset = load_dataset("densud2/ml_qa_dataset", split=split)

        def format_example(x):
            return {
                'prompt': [
                    {'role': 'system', 'content': SYSTEM_PROMPT},
                    {'role': 'user', 'content': x['question']}
                ],
                'answer': x['answer']
            }

        return dataset.map(format_example)

class RewardCalculator:
    """Contains all reward calculation functions"""

    def __init__(self):
        self.embedding_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

    @staticmethod
    def extract_xml_content(text: str, tag: str) -> str:
        """Generic XML tag content extractor"""
        if f"<{tag}>" in text and f"</{tag}>" in text:
            content = text.split(f"<{tag}>")[-1].split(f"</{tag}>")[0]
            return content.strip()
        return ""

    def extract_answer(self, text: str) -> str:
        """Extract answer from XML or fallback format"""
        for tag in ["answer", "think"]:
            content = self.extract_xml_content(text, tag)
            if content:
                return content
        return text.strip()

    def correctness_reward(self, prompts, completions, answers) -> List[float]:
        """Calculate semantic similarity reward"""
        responses = [c[0]['content'] for c in completions]
        extracted = [self.extract_answer(r) for r in responses]

        rewards = []
        for resp, ans in zip(extracted, answers):
            embeds = self.embedding_model.encode([resp, ans], convert_to_tensor=True)
            sim = util.pytorch_cos_sim(embeds[0], embeds[1]).item()
            rewards.append(sim * 2.0)
        return rewards

    @staticmethod
    def format_reward(completions, pattern: str) -> List[float]:
        """Generic format reward checker"""
        return [
            0.5 if re.search(pattern, c[0]["content"], re.DOTALL) else 0.0
            for c in completions
        ]

class TrainingOrchestrator:
    """Manages the training process"""

    @staticmethod
    def get_training_config() -> GRPOConfig:
        """Return configured training parameters"""
        return GRPOConfig(
            use_vllm=True,
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
            gradient_accumulation_steps=1,
            num_generations=8,
            max_prompt_length=256,
            max_completion_length=4096,
            max_steps=250,
            save_steps=250,
            max_grad_norm=0.1,
            report_to="wandb",
            output_dir="outputs",
        )

    def execute_training(self, model, tokenizer, dataset):
        """Run the full training workflow"""
        reward_calculator = RewardCalculator()

        trainer = GRPOTrainer(
            model=model,
            processing_class=tokenizer,
            reward_funcs=[
                reward_calculator.correctness_reward,
                lambda c: self.format_reward(c, r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>"),
                lambda c: self.format_reward(c, r"<think>.*?</think>.*"),
            ],
            args=self.get_training_config(),
            train_dataset=dataset,
        )
        trainer.train()
        return trainer

class InferenceService:
    """Handles model inference"""

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def generate_response(self, prompt: str, lora_path: Optional[str] = None):
        """Generate response with optional LoRA"""
        text = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True
        )

        sampling_params = SamplingParams(
            temperature=0.8,
            top_p=0.95,
            max_tokens=1024,
        )

        return self.model.fast_generate(
            [text],
            sampling_params=sampling_params,
            lora_request=self.model.load_lora(lora_path) if lora_path else None,
        )[0].outputs[0].text

# Main Execution Flow
if __name__ == "__main__":
    # Initialize components
    model, tokenizer = ModelManager.load_model_and_tokenizer()
    dataset = DataHandler.load_qa_dataset()

    # Training
    trainer = TrainingOrchestrator().execute_training(model, tokenizer, dataset)

    # Inference Demo
    inference_engine = InferenceService(model, tokenizer)

    # Base model example
    print("Base Model Response:")
    print(inference_engine.generate_response("How many r's are in strawberry?"))

    # Trained model example
    model.save_lora("grpo_saved_lora")
    print("\nTrained Model Response:")
    print(inference_engine.generate_response(
        "How many r's are in strawberry?",
        lora_path="grpo_saved_lora"
    ))
