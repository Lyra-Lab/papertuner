#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
PaperTuner Inference Dashboard

This marimo notebook provides an interactive dashboard for running inference
with trained models and visualizing results.
"""

import marimo as mo
import os
import json
import torch
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import re
from typing import Dict, List, Any, Optional
import time
from datetime import datetime

from papertuner.train.trainer import MLAssistantTrainer
from papertuner.utils.text_processing import extract_reasoning, extract_xml_answer
from papertuner.utils.constants import DEFAULT_MODEL_NAME, THINK_FORMAT, XML_COT_FORMAT

# Cell 1: Title and introduction
mo.md("""
# 🔍 PaperTuner Inference Dashboard

This interactive dashboard allows you to run inference with trained models,
visualize the results, and compare different prompting strategies.

## Features

- Run inference with different models and LoRA adapters
- Compare base models with fine-tuned versions
- Analyze model reasoning and answers
- Batch processing of multiple queries
- Export results for further analysis
""")

# Cell 2: Model selection UI
with mo.sidebar(open=True):
    mo.md("## Model Configuration")
    
    base_model = mo.ui.text(
        value=DEFAULT_MODEL_NAME,
        label="Base Model",
        placeholder="HuggingFace model name or path"
    )
    
    use_lora = mo.ui.checkbox(
        value=True,
        label="Use LoRA Adapter"
    )
    
    lora_path = mo.ui.text(
        value="trained_model/grpo_saved_lora",
        label="LoRA Path",
        disabled=mo.bind(lambda v: not v, use_lora)
    )
    
    compare_base = mo.ui.checkbox(
        value=True,
        label="Compare with Base Model"
    )
    
    max_seq_length = mo.ui.slider(
        min=1024,
        max=32768,
        value=8192,
        step=1024,
        label="Max Sequence Length"
    )
    
    # Inference parameters
    mo.md("### Inference Parameters")
    
    temperature = mo.ui.slider(
        min=0.1,
        max=1.5,
        value=0.7,
        step=0.1,
        label="Temperature"
    )
    
    top_p = mo.ui.slider(
        min=0.1,
        max=1.0,
        value=0.9,
        step=0.05,
        label="Top P"
    )
    
    max_tokens = mo.ui.slider(
        min=100,
        max=2048,
        value=1024,
        step=100,
        label="Max Tokens"
    )
    
    # Advanced options
    show_advanced = mo.ui.checkbox(label="Show Advanced Options")
    
    @mo.cell
    def advanced_options():
        if not show_advanced.value:
            return ""
        
        format_type = mo.ui.radio(
            options=["XML CoT", "Think/Answer", "Plain Text"],
            value="XML CoT",
            label="Response Format"
        )
        
        system_prompt = mo.ui.text_area(
            value="",
            placeholder="Leave empty to use default prompt",
            label="Custom System Prompt"
        )
        
        return mo.vstack([
            format_type,
            system_prompt
        ])
    
    advanced_options()

# Cell 3: Query input UI
mo.md("## Query Input")

single_query = mo.ui.text_area(
    placeholder="Enter your question here...",
    label="Single Query",
    value="How should I implement attention mechanism in a transformer model?"
)

query_mode = mo.ui.radio(
    options=["Single Query", "Multiple Queries", "Test Set"],
    value="Single Query",
    label="Query Mode"
)

# Cell 4: Multiple queries UI
@mo.cell
def multi_query_ui():
    if query_mode.value != "Multiple Queries":
        return ""
    
    queries = mo.ui.text_area(
        placeholder="Enter one query per line",
        label="Multiple Queries",
        value="""How should I implement attention mechanism in a transformer model?
What's the best way to handle positional encoding?
How do I implement efficient beam search for decoding?
What are the tradeoffs between different tokenization approaches?
How should I approach fine-tuning a large language model?"""
    )
    
    batch_size = mo.ui.slider(
        min=1,
        max=5,
        value=3,
        step=1,
        label="Batch Size"
    )
    
    return mo.vstack([
        queries,
        batch_size
    ])

# Cell 5: Test set UI
@mo.cell
def test_set_ui():
    if query_mode.value != "Test Set":
        return ""
    
    dataset_path = mo.ui.text(
        value="dataset/research_qa_dataset/test",
        placeholder="Path to test dataset",
        label="Test Dataset Path"
    )
    
    num_samples = mo.ui.slider(
        min=1,
        max=20,
        value=5,
        step=1,
        label="Number of Samples"
    )
    
    return mo.vstack([
        dataset_path,
        num_samples
    ])

# Cell 6: Run button
run_button = mo.ui.button("Run Inference", kind="primary")

mo.vstack([
    mo.hstack([
        query_mode,
        run_button
    ]),
    single_query,
    multi_query_ui(),
    test_set_ui()
])

# Cell 7: Helper functions for inference
def get_formatted_prompt(query, format_type, custom_system_prompt):
    """Get a formatted prompt based on the selected format type."""
    system_prompt = custom_system_prompt
    
    if not system_prompt:
        if format_type == "XML CoT":
            system_prompt = """
            Respond in the following format:
            <reasoning>
            Your step-by-step reasoning about the question
            </reasoning>
            <answer>
            Your final answer
            </answer>
            """
        elif format_type == "Think/Answer":
            system_prompt = """
            First think step-by-step about the problem, then provide your answer.
            Use the following format:
            <think>
            Your step-by-step reasoning about the question
            </think>
            Your final answer
            """
        else:  # Plain Text
            system_prompt = "Answer the question in a detailed, step-by-step manner."
    
    return system_prompt, query

def extract_response_parts(response, format_type):
    """Extract reasoning and answer parts from the response."""
    reasoning = ""
    answer = ""
    
    if format_type == "XML CoT":
        reasoning = extract_reasoning(response)
        answer = extract_xml_answer(response)
    elif format_type == "Think/Answer":
        reasoning = extract_reasoning(response)
        answer = extract_xml_answer(response)
    else:  # Plain Text
        answer = response
    
    return reasoning, answer

def calculate_similarity(reference, generated):
    """Calculate simple text similarity."""
    if not reference or not generated:
        return 0.0
        
    # Tokenize by splitting on whitespace
    ref_tokens = set(reference.lower().split())
    gen_tokens = set(generated.lower().split())
    
    # Calculate Jaccard similarity
    if not ref_tokens or not gen_tokens:
        return 0.0
        
    intersection = ref_tokens.intersection(gen_tokens)
    union = ref_tokens.union(gen_tokens)
    
    return len(intersection) / len(union)

# Cell 8: Run inference function
@mo.cell
def run_inference():
    if not run_button.value:
        return mo.md("Click 'Run Inference' to start")
    
    # Reset button
    run_button.update(value=False)
    
    # Get format type and system prompt
    format_type = "XML CoT"
    custom_system_prompt = ""
    
    if show_advanced.value:
        format_values = [child for child in advanced_options().children if hasattr(child, 'value')]
        if len(format_values) >= 1:
            format_type = format_values[0].value
        if len(format_values) >= 2:
            custom_system_prompt = format_values[1].value
    
    # Prepare queries based on mode
    queries = []
    reference_answers = []
    
    if query_mode.value == "Single Query":
        queries = [single_query.value]
        reference_answers = [""]
    elif query_mode.value == "Multiple Queries":
        multi_queries_ui = [child for child in multi_query_ui().children if hasattr(child, 'value')]
        if multi_queries_ui:
            queries = multi_queries_ui[0].value.strip().split("\n")
            reference_answers = [""] * len(queries)
    else:  # Test Set
        test_set_values = [child for child in test_set_ui().children if hasattr(child, 'value')]
        if len(test_set_values) >= 2:
            dataset_path = test_set_values[0].value
            num_samples = test_set_values[1].value
            
            try:
                # Try to load dataset
                from datasets import load_dataset
                try:
                    dataset = load_dataset(dataset_path)
                    split = "test"
                except:
                    dataset = load_from_disk(dataset_path)
                    split = "test" if "test" in dataset else list(dataset.keys())[0]
                
                # Get samples
                samples = dataset[split].select(range(min(num_samples, len(dataset[split]))))
                
                for sample in samples:
                    queries.append(sample["question"])
                    if "answer" in sample:
                        reference_answers.append(sample["answer"])
                    else:
                        reference_answers.append("")
            except Exception as e:
                return mo.md(f"❌ Failed to load test dataset: {str(e)}")
    
    if not queries:
        return mo.md("❌ No queries to process")
    
    # Initialize progress tracking
    num_queries = len(queries)
    progress = mo.ui.progress(value=0, max=100)
    status = mo.md(f"Processing 0/{num_queries} queries")
    
    progress_ui = mo.vstack([
        mo.md("## Processing Queries"),
        progress,
        status
    ])
    
    # Initialize results tracking
    results = []
    
    # Mock for visualization
    def process_queries():
        # Initialize trainers
        with mo.status.spinner("Loading models..."):
            try:
                # Initialize fine-tuned model
                ft_trainer = MLAssistantTrainer(
                    model_name=base_model.value,
                    max_seq_length=max_seq_length.value
                )
                ft_trainer.load_model()
                
                # Initialize base model for comparison if requested
                base_trainer = None
                if compare_base.value:
                    base_trainer = MLAssistantTrainer(
                        model_name=base_model.value,
                        max_seq_length=max_seq_length.value
                    )
                    base_trainer.load_model()
            except Exception as e:
                return mo.md(f"❌ Failed to load models: {str(e)}")
        
        # Process each query
        for i, query in enumerate(queries):
            # Update progress
            progress.update(value=((i) / num_queries) * 100)
            status.update(mo.md(f"Processing query {i+1}/{num_queries}"))
            
            # Format the query
            system_prompt, formatted_query = get_formatted_prompt(
                query, format_type, custom_system_prompt
            )
            
            try:
                # Run inference with fine-tuned model
                ft_response = ft_trainer.run_inference(
                    query=formatted_query,
                    lora_path=lora_path.value if use_lora.value else None,
                    temperature=temperature.value,
                    top_p=top_p.value,
                    max_tokens=max_tokens.value
                )
                
                # Extract parts
                ft_reasoning, ft_answer = extract_response_parts(ft_response, format_type)
                
                # Run comparison with base model if requested
                base_response = ""
                base_reasoning = ""
                base_answer = ""
                
                if compare_base.value and base_trainer:
                    base_response = base_trainer.run_inference(
                        query=formatted_query,
                        temperature=temperature.value,
                        top_p=top_p.value,
                        max_tokens=max_tokens.value
                    )
                    
                    base_reasoning, base_answer = extract_response_parts(base_response, format_type)
                
                # Calculate metrics if reference answer available
                similarity_score = 0.0
                base_similarity_score = 0.0
                
                if reference_answers[i]:
                    similarity_score = calculate_similarity(reference_answers[i], ft_answer)
                    if compare_base.value:
                        base_similarity_score = calculate_similarity(reference_answers[i], base_answer)
                
                # Add to results
                result = {
                    "query": query,
                    "ft_response": ft_response,
                    "ft_reasoning": ft_reasoning,
                    "ft_answer": ft_answer,
                    "base_response": base_response,
                    "base_reasoning": base_reasoning,
                    "base_answer": base_answer,
                    "reference_answer": reference_answers[i],
                    "similarity_score": similarity_score,
                    "base_similarity_score": base_similarity_score
                }
                
                results.append(result)
                
            except Exception as e:
                # Log error but continue processing
                error_result = {
                    "query": query,
                    "ft_response": f"Error: {str(e)}",
                    "ft_reasoning": "",
                    "ft_answer": f"Error: {str(e)}",
                    "base_response": "",
                    "base_reasoning": "",
                    "base_answer": "",
                    "reference_answer": reference_answers[i],
                    "similarity_score": 0.0,
                    "base_similarity_score": 0.0
                }
                results.append(error_result)
        
        # Update progress to complete
        progress.update(value=100)
        status.update(mo.md(f"✅ Completed processing {num_queries} queries"))
        
        return results
    
    # Process queries
    with mo.status.spinner("Initializing..."):
        query_results = process_queries()
    
    # Create a display for single query result
    if query_mode.value == "Single Query" and query_results:
        result = query_results[0]
        
        # Format the display
        if compare_base.value:
            # Side-by-side comparison
            comparison = mo.hstack([
                mo.vstack([
                    mo.md("### Fine-tuned Model Response"),
                    mo.md(result["ft_response"])
                ]),
                mo.vstack([
                    mo.md("### Base Model Response"),
                    mo.md(result["base_response"])
                ])
            ])
            
            return mo.vstack([
                progress_ui,
                mo.md(f"## Query\n\n{result['query']}"),
                comparison
            ])
        else:
            # Just fine-tuned model
            return mo.vstack([
                progress_ui,
                mo.md(f"## Query\n\n{result['query']}"),
                mo.md("### Model Response"),
                mo.md(result["ft_response"])
            ])
    
    # Create a display for multiple queries or test set
    elif query_results:
        # Create a summary table
        summary_data = []
        
        for i, result in enumerate(query_results):
            row = {
                "Query": result["query"][:50] + "..." if len(result["query"]) > 50 else result["query"],
                "FT Answer": result["ft_answer"][:50] + "..." if len(result["ft_answer"]) > 50 else result["ft_answer"]
            }
            
            if compare_base.value:
                row["Base Answer"] = result["base_answer"][:50] + "..." if len(result["base_answer"]) > 50 else result["base_answer"]
            
            if result["reference_answer"]:
                row["Similarity"] = f"{result['similarity_score']:.2f}"
                if compare_base.value:
                    row["Base Similarity"] = f"{result['base_similarity_score']:.2f}"
            
            summary_data.append(row)
        
        summary_table = mo.ui.table(
            pd.DataFrame(summary_data),
            selection="single",
            pagination=True,
            page_size=5
        )
        
        # Display detailed view for selected result
        @mo.cell
        def show_selected_result():
            if not summary_table.value or summary_table.value.empty:
                return mo.md("Select a result to view details")
            
            # Get the selected index
            selected_idx = summary_table.value.index[0]
            result = query_results[selected_idx]
            
            # Create tabs for different views
            if compare_base.value:
                tabs = mo.ui.tabs({
                    "Query": mo.md(f"# Query\n\n{result['query']}"),
                    "Fine-tuned Response": mo.md(f"# Fine-tuned Model Response\n\n{result['ft_response']}"),
                    "Base Response": mo.md(f"# Base Model Response\n\n{result['base_response']}")
                })
                
                if result["reference_answer"]:
                    tabs = mo.ui.tabs({
                        "Query": mo.md(f"# Query\n\n{result['query']}"),
                        "Fine-tuned Response": mo.md(f"# Fine-tuned Model Response\n\n{result['ft_response']}"),
                        "Base Response": mo.md(f"# Base Model Response\n\n{result['base_response']}"),
                        "Reference Answer": mo.md(f"# Reference Answer\n\n{result['reference_answer']}")
                    })
            else:
                tabs = mo.ui.tabs({
                    "Query": mo.md(f"# Query\n\n{result['query']}"),
                    "Model Response": mo.md(f"# Model Response\n\n{result['ft_response']}")
                })
                
                if result["reference_answer"]:
                    tabs = mo.ui.tabs({
                        "Query": mo.md(f"# Query\n\n{result['query']}"),
                        "Model Response": mo.md(f"# Model Response\n\n{result['ft_response']}"),
                        "Reference Answer": mo.md(f"# Reference Answer\n\n{result['reference_answer']}")
                    })
            
            return tabs
        
        # Plot metrics if reference answers available and multiple results
        @mo.cell
        def plot_metrics():
            if not any(r["reference_answer"] for r in query_results):
                return ""
            
            # Extract similarity scores
            ft_scores = [r["similarity_score"] for r in query_results if "similarity_score" in r]
            
            if compare_base.value:
                base_scores = [r["base_similarity_score"] for r in query_results if "base_similarity_score" in r]
                
                # Create a comparison plot
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=list(range(1, len(ft_scores) + 1)),
                    y=ft_scores,
                    name="Fine-tuned Model",
                    marker_color="rgb(55, 83, 109)"
                ))
                fig.add_trace(go.Bar(
                    x=list(range(1, len(base_scores) + 1)),
                    y=base_scores,
                    name="Base Model",
                    marker_color="rgb(26, 118, 255)"
                ))
                
                fig.update_layout(
                    title="Similarity Scores Comparison",
                    xaxis_title="Query",
                    yaxis_title="Similarity Score",
                    barmode="group"
                )
                
                # Calculate average scores
                avg_ft = sum(ft_scores) / len(ft_scores) if ft_scores else 0
                avg_base = sum(base_scores) / len(base_scores) if base_scores else 0
                
                metrics = mo.md(f"""
                ### Average Similarity Scores
                
                - **Fine-tuned Model**: {avg_ft:.3f}
                - **Base Model**: {avg_base:.3f}
                - **Improvement**: {(avg_ft - avg_base):.3f} ({(avg_ft - avg_base) / avg_base * 100:.1f}% relative)
                """)
                
                return mo.vstack([
                    mo.md("## Performance Metrics"),
                    mo.ui.plotly(fig),
                    metrics
                ])
            else:
                # Just plot fine-tuned model metrics
                fig = px.bar(
                    x=list(range(1, len(ft_scores) + 1)),
                    y=ft_scores,
                    labels={"x": "Query", "y": "Similarity Score"},
                    title="Similarity Scores"
                )
                
                avg_ft = sum(ft_scores) / len(ft_scores) if ft_scores else 0
                
                metrics = mo.md(f"""
                ### Average Similarity Score
                
                - **Model**: {avg_ft:.3f}
                """)
                
                return mo.vstack([
                    mo.md("## Performance Metrics"),
                    mo.ui.plotly(fig),
                    metrics
                ])
        
        # Export results button
        export_button = mo.ui.button("Export Results", kind="primary")
        
        @mo.cell
        def handle_export():
            if not export_button.value:
                return ""
            
            export_button.update(value=False)
            
            # Create a DataFrame from results
            export_data = []
            
            for result in query_results:
                row = {
                    "query": result["query"],
                    "ft_answer": result["ft_answer"],
                    "ft_reasoning": result["ft_reasoning"]
                }
                
                if compare_base.value:
                    row["base_answer"] = result["base_answer"]
                    row["base_reasoning"] = result["base_reasoning"]
                
                if result["reference_answer"]:
                    row["reference_answer"] = result["reference_answer"]
                    row["similarity_score"] = result["similarity_score"]
                    
                    if compare_base.value:
                        row["base_similarity_score"] = result["base_similarity_score"]
                
                export_data.append(row)
            
            df = pd.DataFrame(export_data)
            
            # Save to CSV
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"inference_results_{timestamp}.csv"
            df.to_csv(filename, index=False)
            
            return mo.md(f"✅ Results exported to {filename}")
        
        # Combine everything for the results view
        return mo.vstack([
            progress_ui,
            mo.md("## Results Summary"),
            summary_table,
            show_selected_result(),
            plot_metrics(),
            export_button,
            handle_export()
        ])
    
    else:
        return mo.md("❌ No results to display. Check for errors during processing.")

# Cell 9: Main layout
mo.vstack([
    # Title
    mo.md("# 🔍 PaperTuner Inference Dashboard"),
    
    # Query input section
    mo.md("## Query Input"),
    mo.hstack([
        query_mode,
        run_button
    ]),
    
    # Display appropriate input UI based on mode
    single_query,
    multi_query_ui(),
    test_set_ui(),
    
    # Results section
    run_inference()
]) 