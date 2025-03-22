#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
PaperTuner Dataset Explorer

This marimo notebook allows you to explore ML research paper datasets
created with PaperTuner, visualize their contents, and analyze the
distribution of topics and question types.
"""

import marimo as mo
import os
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from typing import Dict, List, Union, Optional

# Cell 1: Title and Introduction
mo.md("""
# 📚 PaperTuner Dataset Explorer

This interactive dashboard allows you to explore datasets created with PaperTuner,
visualize the distribution of research domains, analyze question categories,
and explore individual QA pairs from the dataset.

Select a dataset source below to get started.
""")

# Cell 2: Dataset selection UI
dataset_mode = mo.ui.radio(
    options=["Local Directory", "HuggingFace Dataset"],
    value="Local Directory",
    label="Dataset Source"
)

local_path = mo.ui.text(
    value=os.environ.get("PAPERTUNER_DATA_DIR", "data"),
    placeholder="Path to dataset directory",
    label="Local Dataset Path",
    disabled=mo.bind(lambda m: m != "Local Directory", dataset_mode)
)

hf_dataset = mo.ui.text(
    value="densud2/ml_qa_dataset",
    placeholder="HuggingFace dataset ID (user/dataset)",
    label="HuggingFace Dataset ID",
    disabled=mo.bind(lambda m: m != "HuggingFace Dataset", dataset_mode)
)

dataset_selector_ui = mo.hstack([
    mo.vstack([
        dataset_mode,
        mo.ui.button("Load Dataset", kind="primary")
    ]),
    mo.vstack([
        local_path,
        hf_dataset
    ])
], justify="space-between")

dataset_selector_ui

# Cell 3: Load dataset function
def load_dataset(mode: str, local_path: str, hf_id: str) -> pd.DataFrame:
    """Load dataset from local directory or HuggingFace Hub."""
    
    if mode == "Local Directory":
        path = Path(local_path) / "processed_dataset" / "papers"
        
        if not path.exists():
            return pd.DataFrame()
            
        qa_pairs = []
        
        for file in path.glob("paper_*.json"):
            try:
                with open(file, "r") as f:
                    data = json.load(f)
                    
                # Handle the case with multiple QA pairs
                if "qa_pairs" in data and isinstance(data["qa_pairs"], list):
                    for qa in data["qa_pairs"]:
                        if qa.get("question") and qa.get("answer"):
                            qa_pairs.append({
                                "question": qa["question"],
                                "answer": qa["answer"],
                                "category": qa.get("category", "General"),
                                "paper_id": data["metadata"]["id"],
                                "paper_title": data["metadata"]["title"],
                                "categories": data["metadata"]["categories"]
                            })

                # Handle the legacy case with a single QA pair
                elif "qa" in data and data["qa"].get("question") and data["qa"].get("answer"):
                    qa_pairs.append({
                        "question": data["qa"]["question"],
                        "answer": data["qa"]["answer"],
                        "category": "General",
                        "paper_id": data["metadata"]["id"],
                        "paper_title": data["metadata"]["title"],
                        "categories": data["metadata"]["categories"]
                    })
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Error reading {file}: {e}")
                
        return pd.DataFrame(qa_pairs)
    
    elif mode == "HuggingFace Dataset":
        try:
            from datasets import load_dataset
            dataset = load_dataset(hf_id, split="train")
            return pd.DataFrame(dataset)
        except Exception as e:
            print(f"Error loading HuggingFace dataset: {e}")
            return pd.DataFrame()
    
    return pd.DataFrame()

# Cell 4: Load dataset button
@mo.cell
def _load_dataset():
    with mo.status.busy("Loading dataset..."):
        df = load_dataset(dataset_mode.value, local_path.value, hf_dataset.value)
        
    if df.empty:
        return mo.md("### ❌ No dataset found or dataset is empty")
    
    # Return dataset stats and preview
    stats_md = f"""
    ### Dataset Statistics
    
    - **Total QA pairs**: {len(df)}
    - **Unique papers**: {df['paper_id'].nunique()}
    - **Question categories**: {len(df['category'].unique())}
    - **Research domains**: {len([cat for cats in df['categories'] for cat in (cats if isinstance(cats, list) else [])])}
    """
    
    return mo.vstack([
        mo.md(stats_md),
        mo.ui.table(
            df.head(5),
            selection="single",
            pagination=True,
            page_size=5
        )
    ])

# Cell 5: Process loaded data
@mo.cell
def process_dataset(df):
    if df is None or isinstance(df, str) or not isinstance(df, pd.DataFrame) or df.empty:
        return None
    
    # Extract all unique domains/categories from the papers
    all_domains = []
    for cats in df['categories']:
        if isinstance(cats, list):
            all_domains.extend(cats)
        elif isinstance(cats, str):
            all_domains.append(cats)
    
    domain_counts = Counter(all_domains)
    
    # Count question categories
    category_counts = Counter(df['category'])
    
    # Create word frequencies from questions
    question_text = ' '.join(df['question'].tolist())
    
    # Generate plots
    domain_fig = px.bar(
        x=list(domain_counts.keys()),
        y=list(domain_counts.values()),
        labels={'x': 'Domain', 'y': 'Count'},
        title='Research Domains Distribution'
    )
    
    category_fig = px.pie(
        values=list(category_counts.values()),
        names=list(category_counts.keys()),
        title='Question Categories Distribution'
    )
    
    # Generate word cloud
    wordcloud = WordCloud(
        width=800, height=400,
        background_color='white',
        max_words=100,
        contour_width=3
    ).generate(question_text)
    
    # Convert wordcloud to image
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    wordcloud_fig = plt.gcf()
    
    # Return all visualizations as a dictionary
    return {
        'df': df,
        'domain_fig': domain_fig,
        'category_fig': category_fig,
        'wordcloud': wordcloud_fig
    }

# Cell 6: Visualizations
@mo.cell
def display_visualizations(data):
    if data is None:
        return mo.md("### Please load a dataset first")
    
    domain_chart = mo.ui.plotly(data['domain_fig'])
    category_chart = mo.ui.plotly(data['category_fig'])
    
    tabs = mo.ui.tabs({
        "Domains": domain_chart,
        "Categories": category_chart,
        "Word Cloud": mo.as_html(data['wordcloud'])
    })
    
    return mo.vstack([
        mo.md("## Dataset Visualizations"),
        tabs
    ])

# Cell 7: QA Explorer
@mo.cell
def qa_explorer(data):
    if data is None:
        return mo.md("### Please load a dataset first")
    
    df = data['df']
    
    # Create filters
    category_filter = mo.ui.dropdown(
        options=["All"] + sorted(df['category'].unique().tolist()),
        value="All",
        label="Filter by Category"
    )
    
    search_input = mo.ui.text(
        placeholder="Search in questions",
        label="Search"
    )
    
    # Filter function
    def filter_qa(df, category, search_term):
        filtered_df = df.copy()
        
        if category != "All":
            filtered_df = filtered_df[filtered_df['category'] == category]
            
        if search_term:
            filtered_df = filtered_df[filtered_df['question'].str.contains(search_term, case=False)]
            
        return filtered_df
    
    # Reactive filtered dataframe
    filtered_df = mo.ref(
        filter_qa(df, category_filter.value, search_input.value)
    )
    
    # Update reference when filters change
    @mo.cell
    def _update_filtered():
        filtered_df.set(filter_qa(df, category_filter.value, search_input.value))
    
    # Display selected QA pair
    qa_selector = mo.ui.table(
        filtered_df.value.reset_index()[['index', 'question', 'category', 'paper_title']],
        selection="single",
        pagination=True,
        page_size=10
    )
    
    @mo.cell
    def show_selected_qa():
        if not qa_selector.value or qa_selector.value.empty:
            return mo.md("### Select a question to view details")
        
        selected_idx = qa_selector.value['index'].iloc[0]
        selected_row = df.loc[selected_idx]
        
        return mo.accordion({
            "Question": mo.md(f"### {selected_row['question']}"),
            "Answer": mo.md(selected_row['answer']),
            "Metadata": mo.md(f"""
                **Category**: {selected_row['category']}
                
                **Paper**: {selected_row['paper_title']}
                
                **Paper ID**: {selected_row['paper_id']}
                
                **Categories**: {', '.join(selected_row['categories']) if isinstance(selected_row['categories'], list) else selected_row['categories']}
            """)
        }, active=["Question", "Answer"])
    
    return mo.vstack([
        mo.md("## QA Pair Explorer"),
        mo.hstack([category_filter, search_input]),
        mo.md(f"Showing **{len(filtered_df.value)}** QA pairs"),
        mo.hstack([
            mo.vstack([mo.md("### Select a question"), qa_selector]),
            show_selected_qa()
        ])
    ])

# Cell 8: Export button
@mo.cell
def export_options(data):
    if data is None:
        return ""
    
    export_format = mo.ui.dropdown(
        options=["CSV", "JSON", "Excel"],
        value="CSV",
        label="Export Format"
    )
    
    export_button = mo.ui.button("Export Dataset", kind="primary")
    
    @mo.cell
    def handle_export():
        if not export_button.value or data is None:
            return ""
        
        df = data['df']
        export_path = f"exported_dataset.{export_format.value.lower()}"
        
        if export_format.value == "CSV":
            df.to_csv(export_path, index=False)
        elif export_format.value == "JSON":
            df.to_json(export_path, orient="records", indent=2)
        elif export_format.value == "Excel":
            df.to_excel(export_path, index=False)
            
        return mo.md(f"Dataset exported to `{export_path}`")
    
    return mo.vstack([
        mo.md("## Export Dataset"),
        mo.hstack([export_format, export_button]),
        handle_export()
    ])

# Cell 9: Layout
mo.vstack([
    # Header
    mo.md("# 📚 PaperTuner Dataset Explorer"),
    
    # Dataset selection
    mo.md("## Select Dataset Source"),
    dataset_selector_ui,
    
    # Dataset loading status and preview
    _load_dataset(),
    
    # Tabs for different views
    mo.ui.tabs({
        "Visualizations": display_visualizations(process_dataset(_load_dataset())),
        "QA Explorer": qa_explorer(process_dataset(_load_dataset())),
        "Export": export_options(process_dataset(_load_dataset()))
    })
])