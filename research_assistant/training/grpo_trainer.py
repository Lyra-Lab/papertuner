"""
GRPO (Generative Reinforcement from Preference Optimization) trainer implementation
using Unsloth for efficient fine-tuning.
"""

# Import unsloth first
import unsloth
from unsloth import FastLanguageModel, is_bfloat16_supported

import os
import re
from pathlib import Path

import torch
from trl import GRPOConfig, GRPOTrainer
from datasets import Dataset
from vllm import SamplingParams

from configs.config import MODELS_DIR, OUTPUT_DIR

class ResearchGRPOTrainer:
    """
    Trainer class for fine-tuning research assistant models using GRPO with Unsloth.
    """
    
    def __init__(
        self,
        model_name="unsloth/Phi-4",
        max_seq_length=512,
        lora_rank=16,
        output_dir=None,
        system_prompt=None,
    ):
        """
        Initialize the GRPO trainer.
        
        Args:
            model_name (str): Name of the base model to use
            max_seq_length (int): Maximum sequence length for training
            lora_rank (int): Rank for LoRA fine-tuning
            output_dir (str): Directory to save outputs
            system_prompt (str): System prompt to use for training
        """
        self.model_name = model_name
        self.max_seq_length = max_seq_length
        self.lora_rank = lora_rank
        self.output_dir = output_dir or os.path.join(OUTPUT_DIR, "grpo_model")
        
        # Default system prompt if none provided
        self.system_prompt = system_prompt or """
        Respond in the following format:
        <reasoning>
        ...
        </reasoning>
        <answer>
        ...
        </answer>
        """
        
        # Initialize model and tokenizer
        self.model, self.tokenizer = self._initialize_model()
        
        # XML format for structured responses
        self.xml_cot_format = """
        <reasoning>
        {reasoning}
        </reasoning>
        <answer>
        {answer}
        </answer>
        """
    
    def _initialize_model(self):
        """
        Initialize the model and tokenizer with Unsloth.
        
        Returns:
            tuple: (model, tokenizer)
        """
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.model_name,
            max_seq_length=self.max_seq_length,
            load_in_4bit=True,
            fast_inference=True,
            max_lora_rank=self.lora_rank,
            gpu_memory_utilization=0.7,
        )
        
        model = FastLanguageModel.get_peft_model(
            model,
            r=self.lora_rank,
            target_modules=["gate_proj", "up_proj", "down_proj"],
            lora_alpha=self.lora_rank,
            use_gradient_checkpointing="unsloth",
            random_state=3407,
        )
        
        return model, tokenizer
    
    def prepare_dataset(self, dataset, input_column="input", output_column="output"):
        """
        Prepare dataset for GRPO training.
        
        Args:
            dataset: The dataset to prepare
            input_column (str): Column name for input text
            output_column (str): Column name for output text
            
        Returns:
            Dataset: Prepared dataset
        """
        def format_example(example):
            return {
                'prompt': [
                    {'role': 'system', 'content': self.system_prompt},
                    {'role': 'user', 'content': example[input_column]}
                ],
                'answer': self.extract_answer(example[output_column])
            }
        
        return dataset.map(format_example)
    
    def extract_xml_answer(self, text):
        """Extract answer from XML-formatted text."""
        if "<answer>" not in text or "</answer>" not in text:
            return text
        
        answer = text.split("<answer>")[-1]
        answer = answer.split("</answer>")[0]
        return answer.strip()
    
    def extract_answer(self, text):
        """Extract answer from text, handling different formats."""
        # Try XML format first
        if "<answer>" in text and "</answer>" in text:
            return self.extract_xml_answer(text)
        
        # Try hash format (like GSM8K)
        if "####" in text:
            return text.split("####")[1].strip()
        
        # Return as is if no special format detected
        return text.strip()
    
    def train(self, train_dataset, eval_dataset=None, num_train_epochs=1, learning_rate=5e-6, 
              per_device_train_batch_size=1, gradient_accumulation_steps=4, num_generations=6,
              max_steps=None, logging_steps=1):
        """
        Train the model using GRPO.
        
        Args:
            train_dataset: Training dataset
            eval_dataset: Evaluation dataset
            num_train_epochs (int): Number of training epochs
            learning_rate (float): Learning rate
            per_device_train_batch_size (int): Batch size per device
            gradient_accumulation_steps (int): Gradient accumulation steps
            num_generations (int): Number of generations per prompt
            max_steps (int): Maximum number of training steps
            logging_steps (int): Steps between logging
            
        Returns:
            tuple: (model, training_results)
        """
        # Configure GRPO training arguments
        training_args = GRPOConfig(
            use_vllm=True,
            learning_rate=learning_rate,
            adam_beta1=0.9,
            adam_beta2=0.99,
            weight_decay=0.1,
            warmup_ratio=0.1,
            lr_scheduler_type="cosine",
            optim="paged_adamw_8bit",
            logging_steps=logging_steps,
            bf16=is_bfloat16_supported(),
            fp16=not is_bfloat16_supported(),
            per_device_train_batch_size=per_device_train_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            num_generations=num_generations,
            max_prompt_length=256,
            max_completion_length=256,
            num_train_epochs=num_train_epochs,
            max_steps=max_steps,
            save_steps=250,
            max_grad_norm=0.1,
            report_to="wandb",  # Can use Weights & Biases
            output_dir=self.output_dir,
        )
        
        # Initialize GRPO trainer
        trainer = GRPOTrainer(
            model=self.model,
            processing_class=self.tokenizer,
            reward_funcs=[
                self.xmlcount_reward_func,
                self.soft_format_reward_func,
                self.strict_format_reward_func,
                self.correctness_reward_func,
            ],
            args=training_args,
            train_dataset=train_dataset,
        )
        
        # Train the model
        results = trainer.train()
        
        # Save the trained model
        self.save_model()
        
        return self.model, results
    
    def save_model(self, save_path=None, save_method="lora", push_to_hub=False, hub_path="hf/model", token=""):
        """
        Save the trained model with different methods.
        
        Args:
            save_path (str): Path to save the model
            save_method (str): Method to save the model ('merged_16bit', 'merged_4bit', 'lora', 'gguf_8bit', 'gguf_16bit', 'gguf_q4_k_m')
            push_to_hub (bool): Whether to push the model to the Hugging Face Hub
            hub_path (str): Path on the Hugging Face Hub
            token (str): Token for authentication on the Hugging Face Hub
        """
        save_path = save_path or os.path.join(self.output_dir, "final_lora")
        
        if save_method == "merged_16bit":
            self.model.save_pretrained_merged(save_path, self.tokenizer, save_method="merged_16bit")
            if push_to_hub:
                self.model.push_to_hub_merged(hub_path, self.tokenizer, save_method="merged_16bit", token=token)
        elif save_method == "merged_4bit":
            self.model.save_pretrained_merged(save_path, self.tokenizer, save_method="merged_4bit")
            if push_to_hub:
                self.model.push_to_hub_merged(hub_path, self.tokenizer, save_method="merged_4bit", token=token)
        elif save_method == "lora":
            self.model.save_pretrained_merged(save_path, self.tokenizer, save_method="lora")
            if push_to_hub:
                self.model.push_to_hub_merged(hub_path, self.tokenizer, save_method="lora", token=token)
        elif save_method == "gguf_8bit":
            self.model.save_pretrained_gguf(save_path, self.tokenizer)
            if push_to_hub:
                self.model.push_to_hub_gguf(hub_path, self.tokenizer, token=token)
        elif save_method == "gguf_16bit":
            self.model.save_pretrained_gguf(save_path, self.tokenizer, quantization_method="f16")
            if push_to_hub:
                self.model.push_to_hub_gguf(hub_path, self.tokenizer, quantization_method="f16", token=token)
        elif save_method == "gguf_q4_k_m":
            self.model.save_pretrained_gguf(save_path, self.tokenizer, quantization_method="q4_k_m")
            if push_to_hub:
                self.model.push_to_hub_gguf(hub_path, self.tokenizer, quantization_method="q4_k_m", token=token)
        elif save_method == "gguf_multiple":
            if push_to_hub:
                self.model.push_to_hub_gguf(
                    hub_path,
                    self.tokenizer,
                    quantization_method=["q4_k_m", "q8_0", "q5_k_m"],
                    token=token
                )
        else:
            raise ValueError(f"Unsupported save method: {save_method}\nSupported methods: merged_16bit, merged_4bit, lora, gguf_8bit, gguf_16bit, gguf_q4_k_m, gguf_multiple")
        
        return save_path
    
    def generate(self, prompt, lora_path=None, temperature=0.8, top_p=0.95, max_tokens=1024):
        """
        Generate a response using the trained model.
        
        Args:
            prompt (str): Input prompt
            lora_path (str): Path to the saved LoRA weights
            temperature (float): Sampling temperature
            top_p (float): Top-p sampling parameter
            max_tokens (int): Maximum number of tokens to generate
            
        Returns:
            str: Generated response
        """
        text = self.tokenizer.apply_chat_template([
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt},
        ], tokenize=False, add_generation_prompt=True)
        
        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )
        
        lora_request = None
        if lora_path:
            lora_request = self.model.load_lora(lora_path)
        
        output = self.model.fast_generate(
            [text],
            sampling_params=sampling_params,
            lora_request=lora_request,
        )[0].outputs[0].text
        
        return output
    
    # Reward functions
    def correctness_reward_func(self, prompts, completions, answer, **kwargs):
        """Reward function for correctness of the answer."""
        responses = [completion[0]['content'] for completion in completions]
        extracted_responses = [self.extract_xml_answer(r) for r in responses]
        return [2.0 if r == a else 0.0 for r, a in zip(extracted_responses, answer)]
    
    def strict_format_reward_func(self, completions, **kwargs):
        """Reward function that checks if the completion has the exact expected format."""
        pattern = r"^<reasoning>\n.*?\n</reasoning>\n<answer>\n.*?\n</answer>\n$"
        responses = [completion[0]["content"] for completion in completions]
        matches = [re.match(pattern, r, re.DOTALL) is not None for r in responses]
        return [0.5 if match else 0.0 for match in matches]
    
    def soft_format_reward_func(self, completions, **kwargs):
        """Reward function that checks if the completion has the expected tags."""
        pattern = r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>"
        responses = [completion[0]["content"] for completion in completions]
        matches = [re.search(pattern, r, re.DOTALL) is not None for r in responses]
        return [0.5 if match else 0.0 for match in matches]
    
    def xmlcount_reward_func(self, completions, **kwargs):
        """Reward function that counts XML tags and their proper placement."""
        contents = [completion[0]["content"] for completion in completions]
        return [self._count_xml(c) for c in contents]
    
    def _count_xml(self, text):
        """Helper function to count XML tags and their proper placement."""
        count = 0.0
        if "<reasoning>" in text:
            count += 0.125
        if "</reasoning>" in text:
            count += 0.125
        if "<answer>" in text:
            count += 0.125
            count -= len(text.split("</answer>")[-1])*0.001
        if "</answer>" in text:
            count += 0.125
            count -= (len(text.split("</answer>")[-1]) - 1)*0.001
        return count 