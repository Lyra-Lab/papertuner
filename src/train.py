import os
import logging
import torch
import numpy as np
from pathlib import Path
from datasets import load_dataset
from unsloth import FastLanguageModel
from trl import GRPOConfig, GRPOTrainer
import re
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
from tenacity import retry, stop_after_attempt, wait_exponential

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Constants
MODEL_OUTPUT_DIR = Path("data/trained_model")
LORA_OUTPUT_DIR = Path("data/lora_weights")
DATASET_ID = "densud2/ml_qa_dataset"  # Your published dataset

# Initialize sentence embedding model for semantic reward function
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def get_embedding_model():
    try:
        # Using all-mpnet for high-quality embeddings
        model_name = "sentence-transformers/all-mpnet-base-v2"
        embedding_tokenizer = AutoTokenizer.from_pretrained(model_name)
        embedding_model = AutoModel.from_pretrained(model_name)
        if torch.cuda.is_available():
            embedding_model = embedding_model.to("cuda")
        logging.info("Loaded embedding model for semantic similarity rewards")
        return embedding_model, embedding_tokenizer
    except Exception as e:
        logging.error(f"Failed to load embedding model: {e}")
        raise

def mean_pooling(model_output, attention_mask):
    """Calculate mean pooling for sentence transformers."""
    token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def setup_dirs():
    """Create necessary directories for model training."""
    try:
        MODEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        LORA_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        logging.info("Training directories set up successfully.")
        return True
    except OSError as e:
        logging.error(f"Failed to setup training directories: {e}")
        return False

def load_dataset_from_huggingface():
    """Load the research methodology dataset from Hugging Face and analyze its structure."""
    try:
        # Load the dataset directly from Hugging Face
        dataset = load_dataset(DATASET_ID)
        logging.info(f"Successfully loaded dataset from {DATASET_ID}")
        
        # If the dataset has splits, get the training split
        if isinstance(dataset, dict) and "train" in dataset:
            dataset = dataset["train"]
        
        # Analyze dataset to understand its structure
        first_item = dataset[0] if len(dataset) > 0 else {}
        keys = first_item.keys()
        logging.info(f"Dataset fields: {', '.join(keys)}")
        
        # Log some statistical information
        categories = set()
        avg_question_len = 0
        avg_answer_len = 0
        
        for item in dataset:
            if "category" in item:
                categories.add(item["category"])
            if "question" in item:
                avg_question_len += len(item["question"])
            if "answer" in item:
                avg_answer_len += len(item["answer"])
        
        if len(dataset) > 0:
            avg_question_len /= len(dataset)
            avg_answer_len /= len(dataset)
            
        logging.info(f"Dataset size: {len(dataset)} examples")
        logging.info(f"Question categories: {', '.join(categories)}")
        logging.info(f"Average question length: {avg_question_len:.1f} chars")
        logging.info(f"Average answer length: {avg_answer_len:.1f} chars")
        
        # Define system prompt for the research assistant
        SYSTEM_PROMPT = """
        You are a PhD-level research assistant specialized in helping researchers design optimal methodologies.
        Given a research problem, literature review, and hypothesis, predict the most appropriate methodology.

        Respond in the following format:
        <reasoning>
        Step through your thought process, considering multiple methodological approaches, their advantages,
        disadvantages, and appropriateness for the specific research context. Analyze potential confounding
        variables and how different methods might address them.
        </reasoning>
        <approach>
        Clearly state the recommended methodology with justification.
        </approach>
        """
        
        # Transform the dataset into the format required for GRPO training
        formatted_dataset = []
        
        for item in dataset:
            # Format user content from question and any additional context
            question = item["question"]
            
            # Create a structured problem statement from the question
            user_content = f"Problem: {question}\n\n"
            
            # Add any additional context if available in the dataset
            # Use paper title and categories if available
            context_parts = []
            if "paper_title" in item and item["paper_title"]:
                context_parts.append(f"Research Context: {item['paper_title']}")
            
            if "categories" in item and item["categories"]:
                categories = item["categories"]
                if isinstance(categories, list) and categories:
                    context_parts.append(f"Domain: {', '.join(categories)}")
                elif isinstance(categories, str):
                    context_parts.append(f"Domain: {categories}")
            
            if context_parts:
                user_content += "\n".join(context_parts) + "\n\n"
            
            # Add a hypothesis section if not already in the question
            if "hypothesis" not in question.lower():
                user_content += "Hypothesis: Based on the research context, we aim to determine the most effective approach.\n\n"
            
            # Use the answer as the correct approach
            correct_approach = item["answer"]
            
            # Add any category information as metadata
            category = item.get("category", "General")
            
            formatted_dataset.append({
                "prompt": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_content}
                ],
                "correct_approach": correct_approach,
                "category": category
            })
        
        logging.info(f"Created formatted dataset with {len(formatted_dataset)} training examples")
        return formatted_dataset
    
    except Exception as e:
        logging.error(f"Error loading dataset from Hugging Face: {e}")
        raise

def extract_approach(text: str) -> str:
    """Extract approach from formatted model response."""
    if "<approach>" not in text or "</approach>" not in text:
        return ""
    approach = text.split("<approach>")[-1]
    approach = approach.split("</approach>")[0]
    return approach.strip()

def extract_reasoning(text: str) -> str:
    """Extract reasoning from formatted model response."""
    if "<reasoning>" not in text or "</reasoning>" not in text:
        return ""
    reasoning = text.split("<reasoning>")[-1]
    reasoning = reasoning.split("</reasoning>")[0]
    return reasoning.strip()

# Enhanced semantic similarity reward using embeddings
def semantic_similarity_reward(completions, correct_approach, embedding_model, embedding_tokenizer, **kwargs) -> list[float]:
    """
    Calculate semantic similarity between generated approaches and reference approaches.
    This uses embeddings for a more nuanced comparison than simple word overlap.
    """
    responses = [completion[0]['content'] for completion in completions]
    extracted_approaches = [extract_approach(r) for r in responses]
    
    rewards = []
    
    # Get embeddings for correct approach
    with torch.no_grad():
        # Tokenize correct approach
        correct_encoded = embedding_tokenizer(correct_approach, padding=True, truncation=True, 
                                             return_tensors="pt", max_length=512)
        if torch.cuda.is_available():
            correct_encoded = {k: v.to("cuda") for k, v in correct_encoded.items()}
        
        correct_output = embedding_model(**correct_encoded)
        correct_embedding = mean_pooling(correct_output, correct_encoded['attention_mask'])
        
        # Get embeddings for each generated approach
        for approach in extracted_approaches:
            if len(approach) > 50:  # Basic length check
                # Tokenize generated approach
                approach_encoded = embedding_tokenizer(approach, padding=True, truncation=True, 
                                                     return_tensors="pt", max_length=512)
                if torch.cuda.is_available():
                    approach_encoded = {k: v.to("cuda") for k, v in approach_encoded.items()}
                
                approach_output = embedding_model(**approach_encoded)
                approach_embedding = mean_pooling(approach_output, approach_encoded['attention_mask'])
                
                # Calculate cosine similarity
                similarity = cosine_similarity(
                    correct_embedding.cpu().numpy().reshape(1, -1), 
                    approach_embedding.cpu().numpy().reshape(1, -1)
                )[0][0]
                
                # Scale similarity to reward range (0-2)
                reward = similarity * 2.0
                rewards.append(reward)
            else:
                rewards.append(0.0)  # Penalize very short responses
    
    # Log occasionally for debugging
    if rewards and kwargs.get("step", 0) % 20 == 0:
        logging.info(f"Semantic similarity reward: {rewards[0]:.4f}")
    
    return rewards

# Enhanced format compliance reward with graduated scoring
def enhanced_format_reward(completions, **kwargs) -> list[float]:
    """
    Improved format reward that uses graduated scoring:
    - Full points for perfect formatting
    - Partial points for having both sections but minor formatting issues
    - Minimal points for having at least one section
    - Zero points for completely incorrect format
    """
    responses = [completion[0]["content"] for completion in completions]
    rewards = []
    
    for response in responses:
        # Check for perfect format
        perfect_format = bool(re.search(r"<reasoning>.*?</reasoning>\s*<approach>.*?</approach>", response, re.DOTALL))
        
        # Check for both sections but imperfect format
        has_reasoning_tag = "<reasoning>" in response and "</reasoning>" in response
        has_approach_tag = "<approach>" in response and "</approach>" in response
        
        # Check for reasoning content
        reasoning_content = extract_reasoning(response)
        approach_content = extract_approach(response)
        
        # Calculate reward
        if perfect_format and len(reasoning_content) > 100 and len(approach_content) > 50:
            # Perfect format with substantial content
            reward = 0.6
        elif has_reasoning_tag and has_approach_tag:
            # Both sections present but possibly imperfect format
            reward = 0.4
        elif has_reasoning_tag or has_approach_tag:
            # At least one section present
            reward = 0.2
        else:
            # Completely incorrect format
            reward = 0.0
        
        rewards.append(reward)
    
    return rewards

# Enhanced reasoning quality reward with multiple dimensions
def comprehensive_reasoning_reward(completions, **kwargs) -> list[float]:
    """
    Multi-dimensional reasoning quality assessment that evaluates:
    1. Consideration of multiple approaches
    2. Discussion of advantages/disadvantages
    3. Analysis of confounding variables
    4. Methodological comparisons and justifications
    5. Reasoning depth and thoroughness
    """
    responses = [completion[0]["content"] for completion in completions]
    rewards = []
    
    for response in responses:
        # Extract reasoning section
        reasoning = extract_reasoning(response)
        
        if not reasoning:
            rewards.append(0.0)
            continue
            
        # Initialize score components
        score = 0.0
        reasoning_lower = reasoning.lower()
        
        # 1. Multiple approaches consideration (0.0-0.3)
        approach_indicators = ["approach", "method", "technique", "strategy", "framework", "design"]
        approach_score = 0.0
        
        # Count mentions of different approaches
        approach_count = sum(reasoning_lower.count(f"{indicator} ") for indicator in approach_indicators)
        if approach_count >= 3:
            approach_score = 0.3
        elif approach_count >= 2:
            approach_score = 0.2
        elif approach_count >= 1:
            approach_score = 0.1
        
        # 2. Advantages/disadvantages discussion (0.0-0.3)
        tradeoff_terms = ["advantage", "disadvantage", "benefit", "limitation", 
                          "strength", "weakness", "pro", "con", "tradeoff"]
        tradeoff_score = 0.0
        
        # Count tradeoff discussions
        tradeoff_count = sum(reasoning_lower.count(term) for term in tradeoff_terms)
        if tradeoff_count >= 5:
            tradeoff_score = 0.3
        elif tradeoff_count >= 3:
            tradeoff_score = 0.2
        elif tradeoff_count >= 1:
            tradeoff_score = 0.1
        
        # 3. Confounding variables analysis (0.0-0.2)
        confound_terms = ["confound", "control for", "account for", "variable", 
                          "bias", "validity", "reliability", "factor"]
        confound_score = 0.0
        
        # Check for confounding variables discussion
        if any(term in reasoning_lower for term in confound_terms):
            confound_count = sum(reasoning_lower.count(term) for term in confound_terms)
            if confound_count >= 3:
                confound_score = 0.2
            else:
                confound_score = 0.1
        
        # 4. Methodological comparisons (0.0-0.2)
        comparison_terms = ["compared to", "versus", "unlike", "similar to", 
                            "in contrast", "alternatively", "instead of", "rather than"]
        comparison_score = 0.0
        
        # Check for comparison language
        if any(term in reasoning_lower for term in comparison_terms):
            comparison_count = sum(reasoning_lower.count(term) for term in comparison_terms)
            if comparison_count >= 2:
                comparison_score = 0.2
            else:
                comparison_score = 0.1
        
        # 5. Depth and thoroughness (0.0-0.2)
        # Based on reasoning length and paragraph structure
        depth_score = 0.0
        word_count = len(reasoning.split())
        paragraph_count = reasoning.count("\n\n") + 1
        
        if word_count >= 300 and paragraph_count >= 3:
            depth_score = 0.2
        elif word_count >= 200:
            depth_score = 0.15
        elif word_count >= 100:
            depth_score = 0.1
        elif word_count >= 50:
            depth_score = 0.05
        
        # Calculate total score with a slight scaling factor to make the maximum realistic score around 1.0
        total_score = approach_score + tradeoff_score + confound_score + comparison_score + depth_score
        # Cap the score at 1.0
        total_score = min(total_score, 1.0)
        
        rewards.append(total_score)
    
    return rewards

# Practical methodology reward that focuses on actionability
def practical_methodology_reward(completions, **kwargs) -> list[float]:
    """
    Reward function focused on the practical value of the methodology,
    evaluating whether it provides actionable guidance and addresses
    implementation considerations.
    """
    responses = [completion[0]["content"] for completion in completions]
    rewards = []
    
    for response in responses:
        # Extract approach section
        approach = extract_approach(response)
        
        if not approach:
            rewards.append(0.0)
            continue
        
        # Initialize score
        score = 0.0
        approach_lower = approach.lower()
        
        # 1. Actionability (specific steps or procedures)
        action_terms = ["step", "procedure", "implement", "conduct", "measure", 
                        "analyze", "collect", "select", "design", "perform"]
        action_score = 0.0
        
        action_count = sum(approach_lower.count(term) for term in action_terms)
        if action_count >= 5:
            action_score = 0.5
        elif action_count >= 3:
            action_score = 0.3
        elif action_count >= 1:
            action_score = 0.2
        
        # 2. Implementation considerations
        implementation_terms = ["consideration", "requirement", "resource", "time", 
                               "cost", "expertise", "constraint", "limitation", 
                               "feasibility", "practicality"]
        implementation_score = 0.0
        
        implementation_count = sum(approach_lower.count(term) for term in implementation_terms)
        if implementation_count >= 3:
            implementation_score = 0.3
        elif implementation_count >= 1:
            implementation_score = 0.2
        
        # 3. Evaluation criteria
        evaluation_terms = ["evaluate", "assess", "measure", "metric", "criterion", 
                           "success", "performance", "outcome", "result", "validity"]
        evaluation_score = 0.0
        
        evaluation_count = sum(approach_lower.count(term) for term in evaluation_terms)
        if evaluation_count >= 3:
            evaluation_score = 0.2
        elif evaluation_count >= 1:
            evaluation_score = 0.1
        
        # Calculate total practical methodology score (max 1.0)
        total_score = min(action_score + implementation_score + evaluation_score, 1.0)
        rewards.append(total_score)
    
    return rewards

def train_model(dataset_items):
    """Train the research assistant model using GRPO with enhanced reward functions."""
    logging.info("Initializing model training")
    
    # Convert list to Dataset object
    from datasets import Dataset
    dataset = Dataset.from_list(dataset_items)
    
    # Load embedding model for semantic rewards
    embedding_model, embedding_tokenizer = get_embedding_model()
    
    # Model configuration
    max_seq_length = 2048  # Longer for scientific reasoning traces
    lora_rank = 32  # Higher rank for more complex reasoning
    
    # Load the base model - DeepSeek-R1 (already has reasoning capabilities)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "deepseek-ai/deepseek-r1-1.5b",  # Using DeepSeek-R1
        max_seq_length = max_seq_length,
        load_in_4bit = True,  # Use 4bit quantization to save memory
        fast_inference = True,  # Enable vLLM fast inference
        max_lora_rank = lora_rank,
        gpu_memory_utilization = 0.6,  # Adjust based on your GPU
    )
    
    # Set up LoRA for efficient fine-tuning
    model = FastLanguageModel.get_peft_model(
        model,
        r = lora_rank,
        target_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha = lora_rank,
        use_gradient_checkpointing = "unsloth",  # Enable long context fine-tuning
        random_state = 3407,
    )
    
    # Set up training arguments
    max_prompt_length = 512  # Longer prompt length for scientific context
    
    # Define the number of training steps based on dataset size
    num_epochs = 3
    total_steps = len(dataset) * num_epochs
    max_steps = min(total_steps, 1000)  # Cap at 1000 steps
    
    training_args = GRPOConfig(
        learning_rate = 3e-6,  # Slightly lower for more stable training
        adam_beta1 = 0.9,
        adam_beta2 = 0.99,
        weight_decay = 0.05,
        warmup_ratio = 0.1,
        lr_scheduler_type = "cosine",
        optim = "paged_adamw_8bit",
        logging_steps = 10,
        per_device_train_batch_size = 1,
        gradient_accumulation_steps = 2,  # Increase for stability
        num_generations = 4,  # Balance between diversity and memory
        max_prompt_length = max_prompt_length,
        max_completion_length = max_seq_length - max_prompt_length,
        # num_train_epochs = num_epochs,
        max_steps = max_steps,
        save_steps = 100,
        max_grad_norm = 0.3,  # More aggressive gradient clipping
        report_to = "none",  # Can use "wandb" if you want to track with Weights & Biases
        output_dir = str(MODEL_OUTPUT_DIR),
    )
    
    # Create a wrapper for the semantic reward function to pass the embedding model
    def semantic_reward_wrapper(completions, correct_approach, **kwargs):
        return semantic_similarity_reward(completions, correct_approach, embedding_model, embedding_tokenizer, **kwargs)
    
    # Initialize and run GRPO trainer with our enhanced reward functions
    trainer = GRPOTrainer(
        model = model,
        processing_class = tokenizer,
        reward_funcs = [
            enhanced_format_reward,         # Up to 0.6 for proper formatting
            comprehensive_reasoning_reward,  # Up to 1.0 for high-quality reasoning
            practical_methodology_reward,    # Up to 1.0 for practical methodology
            semantic_reward_wrapper,         # Up to 2.0 for semantic similarity
        ],
        # Assign relative weights to different reward functions
        reward_weights = [0.6, 1.0, 1.0, 2.0],  # Total weight: 4.6
        args = training_args,
        train_dataset = dataset,
    )
    
    logging.info(f"Starting training for {max_steps} steps with enhanced reward functions")
    trainer.train()
    
    # Save the trained model
    lora_path = str(LORA_OUTPUT_DIR / "research_assistant_grpo_lora")
    model.save_lora(lora_path)
    logging.info(f"LoRA weights saved to {lora_path}")
    
    # Save to GGUF format for deployment
    gguf_path = str(MODEL_OUTPUT_DIR / "research_assistant_gguf")
    model.save_pretrained_gguf(
        gguf_path,
        tokenizer,
        quantization_method = "q5_k_m"  # Good balance of quality and size
    )
    logging.info(f"GGUF model saved to {gguf_path}")
    
    return {
        "model": model,
        "tokenizer": tokenizer,
        "lora_path": lora_path,
        "gguf_path": gguf_path
    }

def test_model(model_info, test_query):
    """Test the trained model with a sample query."""
    model = model_info["model"]
    tokenizer = model_info["tokenizer"]
    lora_path = model_info["lora_path"]
    
    # System prompt for consistent formatting
    SYSTEM_PROMPT = """
    You are a PhD-level research assistant specialized in helping researchers design optimal methodologies.
    Given a research problem, literature review, and hypothesis, predict the most appropriate methodology.

    Respond in the following format:
    <reasoning>
    Step through your thought process, considering multiple methodological approaches, their advantages,
    disadvantages, and appropriateness for the specific research context. Analyze potential confounding
    variables and how different methods might address them.
    </reasoning>
    <approach>
    Clearly state the recommended methodology with justification.
    </approach>
    """
    
    text = tokenizer.apply_chat_template([
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": test_query},
    ], tokenize=False, add_generation_prompt=True)

    try:
        from vllm import SamplingParams
        sampling_params = SamplingParams(
            temperature=0.7,
            top_p=0.9,
            max_tokens=1536,
        )
        output = model.fast_generate(
            text,
            sampling_params=sampling_params,
            lora_request=model.load_lora(lora_path),
        )[0].outputs[0].text
        
        return output
    except Exception as e:
        logging.error(f"Error during test inference: {e}")
        return "Error generating response"

def evaluate_model(model_info, test_queries):
    """Evaluate the model on multiple test queries and analyze the results."""
    results = []
    
    for query in test_queries:
        response = test_model(model_info, query)
        
        # Extract and analyze the response
        reasoning = extract_reasoning(response)
        approach = extract_approach(response)
        
        result = {
            "query": query,
            "full_response": response,
            "has_reasoning": bool(reasoning),
            "has_approach": bool(approach),
            "reasoning_length": len(reasoning.split()) if reasoning else 0,
            "approach_length": len(approach.split()) if approach else 0,
        }
        results.append(result)
    
    # Compute overall statistics
    format_compliance = sum(1 for r in results if r["has_reasoning"] and r["has_approach"]) / len(results)
    avg_reasoning_length = sum(r["reasoning_length"] for r in results) / len(results)
    avg_approach_length = sum(r["approach_length"] for r in results) / len(results)
    
    logging.info(f"Model Evaluation Results:")
    logging.info(f"Format compliance: {format_compliance*100:.1f}%")
    logging.info(f"Average reasoning length: {avg_reasoning_length:.1f} words")
    logging.info(f"Average approach length: {avg_approach_length:.1f} words")
    
    return results

def main():
    """Main function to train the research assistant model."""
    if not setup_dirs():
        logging.error("Failed to set up directories. Exiting.")
        return
    
    try:
        # Load dataset directly from Hugging Face
        dataset_items = load_dataset_from_huggingface()
        
        # Train the model with enhanced rewards
        model_info = train_model(dataset_items)
        
        # Define test queries covering different research domains
        test_queries = [
            """
            Problem: Examining the effectiveness of different feedback mechanisms on student learning in online programming courses.

            Literature Review: Studies indicate feedback timing, specificity, and delivery method impact learning outcomes. Automated feedback systems have shown promise, but questions remain about their effectiveness compared to human feedback for complex programming concepts.

            Hypothesis: Students receiving a combination of immediate automated feedback and delayed human expert feedback will demonstrate greater improvement in code quality and conceptual understanding compared to students receiving only one type of feedback.
            """,
            
            """
            Problem: Developing a more efficient approach to training large language models that reduces computational requirements while maintaining performance.

            Literature Review: Current approaches to training LLMs require massive computational resources, limiting accessibility. Techniques like distillation, quantization, and pruning show promise for reducing model size after training, but less work has addressed efficiency during the initial training phase.

            Hypothesis: A dynamic sparse training approach with curriculum learning will reduce computational requirements by 40% while achieving performance within 5% of fully-trained dense models.
            """,
            
            """
            Problem: Investigating how different data augmentation techniques affect the robustness of computer vision models to real-world image variations.

            Literature Review: Computer vision models often underperform when deployed in real-world conditions that differ from training data. Recent studies suggest data augmentation may improve generalization, but there's limited understanding of which techniques are most effective for specific types of real-world variations.

            Hypothesis: Models trained with a combination of physics-based and adversarial augmentations will demonstrate greater robustness to real-world lighting and weather variations than models trained with either technique alone.
            """
        ]
        
        # Evaluate the model on test queries
        evaluation_results = evaluate_model(model_info, test_queries)
        
        # Show a sample result
        sample_result = test_model(model_info, test_queries[0])
        logging.info(f"Sample test result:\n{sample_result[:500]}...\n(truncated)")
        
        logging.info("Training and evaluation complete!")
        
    except Exception as e:
        logging.error(f"Training process failed: {e}")
        import traceback
        logging.error(traceback.format_exc())

if __name__ == "__main__":
    main()
