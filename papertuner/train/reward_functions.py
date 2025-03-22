"""
Reward functions for GRPO training.
"""
import re
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer, util

from papertuner.utils.text_processing import extract_xml_answer, extract_reasoning


class RewardFunctions:
    """Collection of reward functions for GRPO training."""
    
    def __init__(self):
        """Initialize reward functions with required models."""
        self.embedding_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

    def correctness_reward_func(self, prompts, completions, answer, **kwargs) -> List[float]:
        """
        Computes a reward based on the semantic similarity between the extracted response and the target answer.
        The reward is computed as the cosine similarity (scaled by 2.0) between the two embeddings.
        
        Args:
            prompts: List of prompt messages
            completions: List of completion messages
            answer: List of reference answers
            **kwargs: Additional keyword arguments
            
        Returns:
            List of reward scores
        """
        # Get the raw responses and extract the inner answer
        responses = [completion[0]['content'] for completion in completions]
        extracted_responses = [extract_xml_answer(r) for r in responses]

        # Compute rewards based on semantic similarity
        rewards = []
        for response_text, target_text in zip(extracted_responses, answer):
            # Compute embeddings for both the model response and the dataset answer
            embeddings = self.embedding_model.encode([response_text, target_text], convert_to_tensor=True)

            # Compute cosine similarity (range [-1, 1])
            sim = util.pytorch_cos_sim(embeddings[0], embeddings[1]).item()

            # Scale similarity to yield a reward (e.g., multiply by 2.0)
            rewards.append(sim * 2.0)

        # Optional: print debug information for the first example
        if len(prompts) > 0 and len(answer) > 0:
            q = prompts[0][-1]['content']
            print('-'*20, f"Question:\n{q}", f"\nAnswer:\n{answer[0]}", 
                  f"\nResponse:\n{responses[0]}", f"\nExtracted:\n{extracted_responses[0]}")
            
        return rewards

    def int_reward_func(self, completions, **kwargs) -> List[float]:
        """
        Reward for numeric answers.
        
        Args:
            completions: List of completion messages
            **kwargs: Additional keyword arguments
            
        Returns:
            List of reward scores
        """
        responses = [completion[0]['content'] for completion in completions]
        extracted_responses = [extract_xml_answer(r) for r in responses]
        return [0.5 if r.isdigit() else 0.0 for r in extracted_responses]

    def strict_format_reward_func(self, completions, **kwargs) -> List[float]:
        """
        Reward function that checks if the completion has a specific format.
        Handles both <reasoning> and <think> tag formats.
        
        Args:
            completions: List of completion messages
            **kwargs: Additional keyword arguments
            
        Returns:
            List of reward scores
        """
        pattern1 = r"^<reasoning>\n.*?\n</reasoning>\n<answer>\n.*?\n</answer>\n$"
        pattern2 = r"^<think>\n.*?\n</think>\n.*$"

        responses = [completion[0]["content"] for completion in completions]
        matches = [re.search(pattern1, r, re.DOTALL) or re.search(pattern2, r, re.DOTALL) for r in responses]

        return [0.5 if match else 0.0 for match in matches]

    def soft_format_reward_func(self, completions, **kwargs) -> List[float]:
        """
        Reward function that checks if the completion has expected tags in any order.
        Handles both <reasoning>/<think> tag formats.
        
        Args:
            completions: List of completion messages
            **kwargs: Additional keyword arguments
            
        Returns:
            List of reward scores
        """
        pattern1 = r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>"
        pattern2 = r"<think>.*?</think>.*"

        responses = [completion[0]["content"] for completion in completions]
        matches = [re.search(pattern1, r, re.DOTALL) or re.search(pattern2, r, re.DOTALL) for r in responses]

        return [0.5 if match else 0.0 for match in matches]

    def _count_xml(self, text) -> float:
        """
        Count XML tags and award points based on formatting.
        Updated to handle both formats.
        
        Args:
            text: The text to analyze
            
        Returns:
            Score based on XML tag formatting
        """
        count = 0.0

        # Handle original format with reasoning/answer tags
        if "<reasoning>" in text and "</reasoning>" in text:
            if text.count("<reasoning>\n") == 1:
                count += 0.125
            if text.count("\n</reasoning>\n") == 1:
                count += 0.125
            if text.count("\n<answer>\n") == 1:
                count += 0.125
                count -= len(text.split("\n</answer>\n")[-1]) * 0.001
            if text.count("\n</answer>") == 1:
                count += 0.125
                count -= (len(text.split("\n</answer>")[-1]) - 1) * 0.001

        # Handle new format with think tags
        elif "<think>" in text and "</think>" in text:
            if text.count("<think>\n") == 1:
                count += 0.25
            if text.count("\n</think>\n") == 1:
                count += 0.25
            # Content after </think> should be the answer
            if "</think>" in text:
                answer_content = text.split("</think>")[-1].strip()
                if answer_content:  # If there's content after </think>
                    count += 0.25
                    # Penalize if there's more tags after the answer section
                    if "<" in answer_content and ">" in answer_content:
                        count -= 0.1

        return count

    def xmlcount_reward_func(self, completions, **kwargs) -> List[float]:
        """
        Reward function based on XML tag formatting.
        
        Args:
            completions: List of completion messages
            **kwargs: Additional keyword arguments
            
        Returns:
            List of reward scores based on XML formatting
        """
        contents = [completion[0]["content"] for completion in completions]
        return [self._count_xml(c) for c in contents]

    def flexible_format_reward_func(self, completions, **kwargs) -> List[float]:
        """
        A more flexible reward function that can handle both the original format
        with <reasoning>/<answer> tags and the new format with <think> tags.
        
        Args:
            completions: List of completion messages
            **kwargs: Additional keyword arguments
            
        Returns:
            List of reward scores
        """
        responses = [completion[0]["content"] for completion in completions]
        rewards = []

        for r in responses:
            reward = 0.0

            # Check if the response has valid reasoning/thinking
            reasoning = extract_reasoning(r)
            if reasoning:
                reward += 0.25

            # Check if the response has a valid answer section
            answer = extract_xml_answer(r)
            if answer and answer != r:  # Make sure we actually extracted something
                reward += 0.25

            # Bonus for proper formatting based on the format detected
            if "<think>" in r and "</think>" in r:
                if r.count("<think>") == 1 and r.count("</think>") == 1:
                    reward += 0.1

                # The text after </think> should be cleanly formatted
                post_think = r.split("</think>")[-1].strip()
                if post_think and not (post_think.startswith("<") and not post_think.startswith("<answer>")):
                    reward += 0.1

            elif "<reasoning>" in r and "</reasoning>" in r and "<answer>" in r and "</answer>" in r:
                if r.count("<reasoning>") == 1 and r.count("</reasoning>") == 1 and r.count("<answer>") == 1 and r.count("</answer>") == 1:
                    reward += 0.2

            rewards.append(reward)

        return rewards
    
    def get_all_reward_functions(self):
        """
        Return all reward functions as a list.
        
        Returns:
            List of reward functions
        """
        return [
            self.xmlcount_reward_func,
            self.soft_format_reward_func,
            self.strict_format_reward_func,
            self.int_reward_func,
            self.correctness_reward_func,
        ] 