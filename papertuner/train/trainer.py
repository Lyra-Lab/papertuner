"""
Model training using GRPO (Guided Reward Preference Optimization).
"""
import os
import torch
from typing import List, Dict, Union, Optional, Tuple

from unsloth import FastLanguageModel, is_bfloat16_supported
from datasets import Dataset
from trl import GRPOConfig, GRPOTrainer
from vllm import SamplingParams

from papertuner.utils.constants import SYSTEM_PROMPT, DEFAULT_MODEL_NAME, DEFAULT_MAX_SEQ_LENGTH, \
    DEFAULT_LORA_RANK, DEFAULT_MAX_STEPS, DEFAULT_SAVE_STEPS, DEFAULT_GPU_MEMORY_UTILIZATION
from papertuner.train.reward_functions import RewardFunctions
from papertuner.data.dataset import get_ml_qa_dataset, get_custom_dataset


class MLAssistantTrainer:
    """Trainer for fine-tuning language models on ML research QA pairs."""
    
    def __init__(self, 
                 model_name: str = DEFAULT_MODEL_NAME, 
                 output_dir: str = "outputs", 
                 max_seq_length: int = DEFAULT_MAX_SEQ_LENGTH,
                 lora_rank: int = DEFAULT_LORA_RANK,
                 max_steps: int = DEFAULT_MAX_STEPS, 
                 save_steps: int = DEFAULT_SAVE_STEPS,
                 gpu_memory_utilization: float = DEFAULT_GPU_MEMORY_UTILIZATION):
        """
        Initialize the trainer.
        
        Args:
            model_name: Name of the base model to fine-tune
            output_dir: Directory to save outputs to
            max_seq_length: Maximum sequence length for the model
            lora_rank: LoRA rank for fine-tuning
            max_steps: Maximum number of training steps
            save_steps: Save checkpoint every X steps
            gpu_memory_utilization: GPU memory utilization (reduce if OOM)
        """
        self.model_name = model_name
        self.output_dir = output_dir
        self.max_seq_length = max_seq_length
        self.lora_rank = lora_rank
        self.max_steps = max_steps
        self.save_steps = save_steps
        self.gpu_memory_utilization = gpu_memory_utilization
        self.model = None
        self.tokenizer = None
        self.dataset = None
        self.reward_funcs = None

    def load_model(self):
        """
        Load and prepare the model for training.
        
        Returns:
            Tuple of (model, tokenizer)
        """
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.model_name,
            max_seq_length=self.max_seq_length,
            load_in_4bit=True,
            fast_inference=True,
            max_lora_rank=self.lora_rank,
            gpu_memory_utilization=self.gpu_memory_utilization,
        )

        self.model = FastLanguageModel.get_peft_model(
            self.model,
            r=self.lora_rank,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
            lora_alpha=self.lora_rank,
            use_gradient_checkpointing="unsloth",
            random_state=3407,
        )

        return self.model, self.tokenizer

    def load_dataset(self, dataset_path: Optional[str] = None, split: str = "train"):
        """
        Load the training dataset.
        
        Args:
            dataset_path: Path to a custom dataset (if None, use the default ML QA dataset)
            split: Dataset split to use
            
        Returns:
            The loaded dataset
        """
        if dataset_path:
            self.dataset = get_custom_dataset(dataset_path, split=split)
        else:
            self.dataset = get_ml_qa_dataset(split=split)
            
        return self.dataset

    def setup_reward_functions(self):
        """
        Set up the reward functions for training.
        
        Returns:
            List of reward functions
        """
        reward_funcs_obj = RewardFunctions()
        self.reward_funcs = reward_funcs_obj.get_all_reward_functions()
        return self.reward_funcs

    def get_training_args(self, report_to: str = "wandb"):
        """
        Configure the training arguments.
        
        Args:
            report_to: Logging platform to report to
            
        Returns:
            GRPOConfig object with training arguments
        """
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
            max_steps=self.max_steps,
            save_steps=self.save_steps,
            max_grad_norm=0.1,
            report_to=report_to,
            output_dir=self.output_dir,
        )

    def train(self, report_to: str = "wandb"):
        """
        Train the model using GRPO.
        
        Args:
            report_to: Logging platform to report to
            
        Returns:
            Tuple of (model, tokenizer)
        """
        if self.model is None:
            self.load_model()

        if self.dataset is None:
            self.load_dataset()

        if self.reward_funcs is None:
            self.setup_reward_functions()

        training_args = self.get_training_args(report_to=report_to)

        trainer = GRPOTrainer(
            model=self.model,
            processing_class=self.tokenizer,
            reward_funcs=self.reward_funcs,
            args=training_args,
            train_dataset=self.dataset,
        )

        trainer.train()

        # Save the trained model
        os.makedirs(self.output_dir, exist_ok=True)
        self.model.save_lora(os.path.join(self.output_dir, "grpo_saved_lora"))
        return self.model, self.tokenizer

    def run_inference(self, query: str, lora_path: Optional[str] = None, 
                     temperature: float = 0.8, top_p: float = 0.95, max_tokens: int = 1024):
        """
        Run inference with the trained model.
        
        Args:
            query: Input query for the model
            lora_path: Path to the LoRA weights to use
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            max_tokens: Maximum number of tokens to generate
            
        Returns:
            Generated text
        """
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model and tokenizer must be loaded before inference")

        text = self.tokenizer.apply_chat_template([
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": query},
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
            text,
            sampling_params=sampling_params,
            lora_request=lora_request,
        )[0].outputs[0].text

        return output

    def save_model(self, model_path: Optional[str] = None, save_method: str = "merged_16bit", 
                  push_to_hub: bool = False, hub_model_id: Optional[str] = None, token: Optional[str] = None):
        """
        Save the model in the specified format.
        
        Args:
            model_path: Path to save the model to (if None, use self.output_dir)
            save_method: Method to use for saving the model
            push_to_hub: Whether to push the model to the HuggingFace Hub
            hub_model_id: Model ID for HuggingFace Hub
            token: HuggingFace token
        """
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model and tokenizer must be loaded before saving")

        model_path = model_path or self.output_dir

        if push_to_hub and hub_model_id:
            self.model.push_to_hub_merged(
                hub_model_id,
                self.tokenizer,
                save_method=save_method,
                token=token
            )
        else:
            self.model.save_pretrained_merged(
                model_path,
                self.tokenizer,
                save_method=save_method
            )

    def save_model_gguf(self, model_path: Optional[str] = None, 
                       quantization_methods: Optional[List[str]] = None,
                       push_to_hub: bool = False, hub_model_id: Optional[str] = None, 
                       token: Optional[str] = None):
        """
        Save the model in GGUF format for llama.cpp compatibility.
        
        Args:
            model_path: Path to save the model to (if None, use self.output_dir)
            quantization_methods: List of quantization methods to use
            push_to_hub: Whether to push the model to the HuggingFace Hub
            hub_model_id: Model ID for HuggingFace Hub
            token: HuggingFace token
        """
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model and tokenizer must be loaded before saving")

        model_path = model_path or self.output_dir
        quantization_methods = quantization_methods or ["q4_k_m"]

        if push_to_hub and hub_model_id:
            self.model.push_to_hub_gguf(
                hub_model_id,
                self.tokenizer,
                quantization_method=quantization_methods,
                token=token
            )
        else:
            self.model.save_pretrained_gguf(
                model_path,
                self.tokenizer,
                quantization_method=quantization_methods
            ) 