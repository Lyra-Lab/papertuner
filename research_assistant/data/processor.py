"""
Dataset processing utilities.
"""

import os
from pathlib import Path

import pandas as pd
from datasets import Dataset, DatasetDict

from configs.config import RAW_DATA_DIR, PROCESSED_DATA_DIR

def load_raw_data(filename, format="csv"):
    """
    Load raw data from the raw data directory.
    
    Args:
        filename (str): Name of the file to load.
        format (str): Format of the file ('csv', 'json', 'parquet').
        
    Returns:
        pd.DataFrame: The loaded data.
    """
    file_path = Path(RAW_DATA_DIR) / filename
    
    if format == "csv":
        return pd.read_csv(file_path)
    elif format == "json":
        return pd.read_json(file_path)
    elif format == "parquet":
        return pd.read_parquet(file_path)
    else:
        raise ValueError(f"Unsupported format: {format}")

def save_processed_data(df, filename, format="csv"):
    """
    Save processed data to the processed data directory.
    
    Args:
        df (pd.DataFrame): The data to save.
        filename (str): Name of the file to save.
        format (str): Format to save the file ('csv', 'json', 'parquet').
    """
    file_path = Path(PROCESSED_DATA_DIR) / filename
    
    if format == "csv":
        df.to_csv(file_path, index=False)
    elif format == "json":
        df.to_json(file_path, orient="records", lines=True)
    elif format == "parquet":
        df.to_parquet(file_path, index=False)
    else:
        raise ValueError(f"Unsupported format: {format}")

def prepare_research_dataset(df, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    """
    Prepare a research dataset from a DataFrame.
    
    Args:
        df (pd.DataFrame): DataFrame containing 'problem', 'literature', 
                           'hypothesis', and 'methodology' columns.
        train_ratio (float): Ratio of data to use for training.
        val_ratio (float): Ratio of data to use for validation.
        test_ratio (float): Ratio of data to use for testing.
        
    Returns:
        DatasetDict: A dataset dictionary with train, validation, and test splits.
    """
    # Verify that ratios sum to 1
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1"
    
    # Create input-output pairs
    df["input"] = (
        "Research Problem: " + df["problem"] + "\n\n" +
        "Literature Review: " + df["literature"] + "\n\n" +
        "Hypothesis: " + df["hypothesis"] + "\n\n" +
        "What methodology should be used for this research?"
    )
    
    df["output"] = "Methodology: " + df["methodology"]
    
    # Split the data
    n = len(df)
    train_size = int(n * train_ratio)
    val_size = int(n * val_ratio)
    
    # Shuffle the data
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    train_df = df.iloc[:train_size]
    val_df = df.iloc[train_size:train_size+val_size]
    test_df = df.iloc[train_size+val_size:]
    
    # Convert to Hugging Face datasets
    train_dataset = Dataset.from_pandas(train_df[["input", "output"]])
    val_dataset = Dataset.from_pandas(val_df[["input", "output"]])
    test_dataset = Dataset.from_pandas(test_df[["input", "output"]])
    
    # Create a DatasetDict
    dataset_dict = DatasetDict({
        "train": train_dataset,
        "validation": val_dataset,
        "test": test_dataset
    })
    
    return dataset_dict