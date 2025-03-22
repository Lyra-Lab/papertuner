#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
PaperTuner Paper Processor

This marimo notebook provides an interactive interface for downloading
research papers from arXiv, processing them, and generating QA pairs.
"""

import marimo as mo
import os
import json
import pandas as pd
import plotly.express as px
from pathlib import Path
from typing import Dict, List, Any, Optional
import time
from datetime import datetime

from papertuner.data.processor import PaperProcessor
from papertuner.utils.constants import RAW_DIR, PROCESSED_DIR

# Cell 1: Title and introduction
mo.md("""
# 📄 PaperTuner Paper Processor

This interactive notebook helps you search, download, and process research papers from arXiv.
You can generate QA pairs from the papers using the OpenAI or Gemini API.

## Features

- Search for papers on arXiv using custom queries
- Process papers to extract text and sections
- Generate high-quality QA pairs from paper content
- Create and export datasets for model training
""")

# Cell 2: API configuration
api_key_provider = mo.ui.radio(
    options=["OpenAI", "Gemini", "None (Skip QA Generation)"],
    value="OpenAI" if os.environ.get("OPENAI_API_KEY") else 
          "Gemini" if os.environ.get("GEMINI_API_KEY") else 
          "None (Skip QA Generation)",
    label="API Provider"
)

api_key = mo.ui.text(
    value=os.environ.get("OPENAI_API_KEY", "") or os.environ.get("GEMINI_API_KEY", ""),
    password=True,
    placeholder="Enter your API key",
    label="API Key",
    disabled=mo.bind(lambda p: p == "None (Skip QA Generation)", api_key_provider)
)

model_name = mo.ui.dropdown(
    options=["gpt-3.5-turbo", "gpt-4", "gemini-pro"],
    value="gpt-3.5-turbo" if api_key_provider.value == "OpenAI" else "gemini-pro",
    label="Model for QA Generation",
    disabled=mo.bind(lambda p: p == "None (Skip QA Generation)", api_key_provider)
)

advanced_options = mo.ui.checkbox(label="Show Advanced Options")

with mo.sidebar(open=True):
    mo.md("## API Configuration")
    
    api_config = mo.vstack([
        api_key_provider,
        api_key,
        model_name,
        advanced_options
    ])
    
    mo.md("### Environment Variables")
    env_table = pd.DataFrame([
        {"Variable": "OPENAI_API_KEY", "Status": "✅ Set" if os.environ.get("OPENAI_API_KEY") else "❌ Not Set"},
        {"Variable": "GEMINI_API_KEY", "Status": "✅ Set" if os.environ.get("GEMINI_API_KEY") else "❌ Not Set"},
        {"Variable": "HF_TOKEN", "Status": "✅ Set" if os.environ.get("HF_TOKEN") else "❌ Not Set"}
    ])
    mo.ui.table(env_table)
    
    api_config

# Cell 3: Advanced options (show conditionally)
@mo.cell
def show_advanced_options():
    if not advanced_options.value:
        return ""
    
    data_dir = mo.ui.text(
        value=os.environ.get("PAPERTUNER_DATA_DIR", "data"),
        label="Data Directory"
    )
    
    api_base_url = mo.ui.text(
        value=os.environ.get("API_BASE_URL", ""),
        label="API Base URL (Optional)",
        placeholder="For custom API endpoints"
    )
    
    return mo.hstack([
        data_dir,
        api_base_url
    ])

# Cell 4: Search interface
arxiv_search = mo.ui.text(
    placeholder="Enter search terms",
    label="arXiv Search Query",
    value='"large language models" OR "LLM training" OR "fine-tuning"'
)

max_papers = mo.ui.slider(
    min=1,
    max=50,
    value=5,
    label="Maximum Number of Papers"
)

search_button = mo.ui.button("Search arXiv", kind="primary")

mo.md("## Search for Research Papers")
mo.hstack([
    mo.vstack([
        arxiv_search,
        max_papers
    ]),
    search_button
])

# Cell 5: Initialize paper processor
@mo.cell
def initialize_processor():
    # Determine API key and provider
    if api_key_provider.value == "OpenAI":
        os.environ["OPENAI_API_KEY"] = api_key.value
        model = model_name.value if "gpt" in model_name.value else "gpt-3.5-turbo"
    elif api_key_provider.value == "Gemini":
        os.environ["GEMINI_API_KEY"] = api_key.value
        model = model_name.value if "gemini" in model_name.value else "gemini-pro"
    else:
        model = None
    
    # Set data directory if specified in advanced options
    if advanced_options.value and 'data_dir' in globals():
        os.environ["PAPERTUNER_DATA_DIR"] = data_dir.value
    
    # Set API base URL if specified in advanced options
    api_base = None
    if advanced_options.value and 'api_base_url' in globals() and api_base_url.value:
        api_base = api_base_url.value
    
    processor = PaperProcessor(
        api_key=api_key.value if api_key_provider.value != "None (Skip QA Generation)" else None,
        api_base_url=api_base,
        model_name=model
    )
    
    return processor

# Cell 6: Search and process papers
@mo.cell
def search_papers():
    if not search_button.value:
        return mo.md("Click the Search button to start processing papers")
    
    processor = initialize_processor()
    
    # Create a progress component
    progress = mo.ui.progress(value=0, max=100)
    status_text = mo.md("Initializing search...")
    
    # Display initial status
    status_display = mo.vstack([
        mo.md("## Processing Papers"),
        progress,
        status_text
    ])
    
    # Function to update progress
    def update_progress(i, total, message):
        progress.update(value=(i / total) * 100)
        status_text.update(mo.md(message))
        time.sleep(0.1)  # Allow UI to update
    
    # Process papers with status updates
    try:
        update_progress(0, max_papers.value, "Searching arXiv...")
        
        # Store the result in a list to track progress
        results = []
        
        # Use a custom processing wrapper to update progress
        def process_with_progress():
            update_progress(0, max_papers.value, "Starting paper processing...")
            
            # Monitor the papers directory to track progress
            papers_dir = PROCESSED_DIR / "papers"
            papers_dir.mkdir(parents=True, exist_ok=True)
            initial_count = len(list(papers_dir.glob("paper_*.json")))
            
            # Start the processing
            new_papers = processor.process_papers(
                max_papers=max_papers.value,
                query=arxiv_search.value
            )
            
            # If we have results, return them
            if new_papers:
                return new_papers
            
            # Otherwise, check what files were created
            final_files = list(papers_dir.glob("paper_*.json"))
            new_count = len(final_files) - initial_count
            
            if new_count > 0:
                # Return info about the new files
                return [{"id": f.stem, "filename": f.name, "processed_date": datetime.now().isoformat()} 
                        for f in final_files[-new_count:]]
            
            return []
        
        # Run the processing
        results = process_with_progress()
        
        # Update progress based on results
        if results:
            for i, paper in enumerate(results):
                update_progress(i + 1, max_papers.value, 
                               f"Processed paper {i+1}/{len(results)}: {paper.get('title', paper.get('id', ''))}")
        
        update_progress(max_papers.value, max_papers.value, f"Completed! Processed {len(results)} papers.")
        
        # Return success message and results
        return mo.vstack([
            mo.md(f"## Successfully processed {len(results)} papers"),
            mo.ui.table(
                pd.DataFrame(results),
                pagination=True,
                page_size=5
            )
        ])
        
    except Exception as e:
        return mo.md(f"## ❌ Error during processing\n\n```\n{str(e)}\n```")

# Cell 7: Dataset creation interface
mo.md("## Create Dataset from Processed Papers")

dataset_name = mo.ui.text(
    placeholder="dataset name (optional)",
    value="research_qa_dataset",
    label="Dataset Name"
)

split_ratios = mo.ui.slider(
    min=0.5,
    max=0.9,
    value=0.8,
    step=0.05,
    label="Training Split Ratio"
)

create_dataset_button = mo.ui.button("Create Dataset", kind="primary")

mo.hstack([
    mo.vstack([
        dataset_name,
        split_ratios
    ]),
    create_dataset_button
])

# Cell 8: Create dataset
@mo.cell
def create_dataset():
    if not create_dataset_button.value:
        return ""
    
    processor = initialize_processor()
    
    with mo.status.spinner("Creating dataset..."):
        # Calculate split ratios
        train_ratio = split_ratios.value
        val_ratio = (1 - train_ratio) / 2
        test_ratio = (1 - train_ratio) / 2
        
        # Create the dataset
        output_path = Path("dataset") / dataset_name.value
        dataset_dict = processor.create_dataset_from_processed_papers(
            output_path=str(output_path),
            split_ratios=(train_ratio, val_ratio, test_ratio)
        )
    
    if not dataset_dict:
        return mo.md("### ❌ Failed to create dataset. Make sure you've processed papers first.")
    
    # Display dataset statistics
    stats = {
        "Split": ["Train", "Validation", "Test", "Total"],
        "Samples": [
            len(dataset_dict["train"]),
            len(dataset_dict["validation"]),
            len(dataset_dict["test"]),
            len(dataset_dict["train"]) + len(dataset_dict["validation"]) + len(dataset_dict["test"])
        ],
        "Percentage": [
            f"{train_ratio*100:.1f}%",
            f"{val_ratio*100:.1f}%",
            f"{test_ratio*100:.1f}%",
            "100%"
        ]
    }
    
    # Plot distribution
    fig = px.pie(
        values=stats["Samples"][:3],
        names=stats["Split"][:3],
        title="Dataset Split Distribution"
    )
    
    return mo.vstack([
        mo.md(f"### ✅ Dataset Created Successfully\n\nSaved to: `{output_path}`"),
        mo.ui.table(pd.DataFrame(stats)),
        mo.ui.plotly(fig),
        mo.md("### Sample QA Pairs"),
        mo.ui.table(
            pd.DataFrame([
                {"question": sample["question"], "answer_preview": sample["answer"][:100] + "..."}
                for sample in dataset_dict["train"][:5]
            ]),
            pagination=True,
            page_size=5
        )
    ])

# Cell 9: Upload to HuggingFace interface
mo.md("## Upload Dataset to Hugging Face Hub")

hf_token = mo.ui.text(
    value=os.environ.get("HF_TOKEN", ""),
    password=True,
    label="HuggingFace Token"
)

hf_repo_id = mo.ui.text(
    value=os.environ.get("HF_REPO_ID", ""),
    placeholder="username/dataset-name",
    label="Repository ID"
)

upload_button = mo.ui.button("Upload to HuggingFace", kind="primary")

mo.hstack([
    mo.vstack([
        hf_token,
        hf_repo_id
    ]),
    upload_button
])

# Cell 10: Upload to HuggingFace
@mo.cell
def upload_to_hf():
    if not upload_button.value:
        return ""
    
    if not hf_token.value or not hf_repo_id.value:
        return mo.md("### ❌ HuggingFace token and repository ID are required")
    
    processor = initialize_processor()
    
    with mo.status.spinner("Uploading dataset to HuggingFace Hub..."):
        # Calculate split ratios
        train_ratio = split_ratios.value
        val_ratio = (1 - train_ratio) / 2
        test_ratio = (1 - train_ratio) / 2
        
        # Set environment variables
        os.environ["HF_TOKEN"] = hf_token.value
        os.environ["HF_REPO_ID"] = hf_repo_id.value
        
        # Upload dataset
        success = processor.upload_to_hf(
            hf_token=hf_token.value,
            hf_repo_id=hf_repo_id.value,
            split_ratios=(train_ratio, val_ratio, test_ratio)
        )
    
    if success:
        return mo.vstack([
            mo.md(f"### ✅ Dataset Uploaded Successfully\n\nRepository: [https://huggingface.co/datasets/{hf_repo_id.value}](https://huggingface.co/datasets/{hf_repo_id.value})"),
            mo.md(f"""
            ```python
            # You can now load this dataset with:
            from datasets import load_dataset
            
            dataset = load_dataset("{hf_repo_id.value}")
            ```
            """)
        ])
    else:
        return mo.md("### ❌ Failed to upload dataset to HuggingFace Hub")

# Cell 11: Main layout
mo.vstack([
    # Header
    mo.md("# 📄 PaperTuner Paper Processor"),
    
    # Advanced options
    show_advanced_options(),
    
    # Search interface
    mo.md("## Search for Research Papers"),
    mo.hstack([
        mo.vstack([
            arxiv_search,
            max_papers
        ]),
        search_button
    ]),
    
    # Search results
    search_papers(),
    
    # Dataset creation interface
    mo.md("## Create Dataset from Processed Papers"),
    mo.hstack([
        mo.vstack([
            dataset_name,
            split_ratios
        ]),
        create_dataset_button
    ]),
    
    # Dataset creation results
    create_dataset(),
    
    # Upload to HuggingFace interface
    mo.md("## Upload Dataset to Hugging Face Hub"),
    mo.hstack([
        mo.vstack([
            hf_token,
            hf_repo_id
        ]),
        upload_button
    ]),
    
    # Upload results
    upload_to_hf()
]) 