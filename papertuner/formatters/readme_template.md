# {{ dataset_name }}

## Dataset Description

{{ dataset_description }}

### Dataset Summary

This dataset contains scientific papers processed with PaperTuner. It includes full text,
metadata, and summaries that can be used for training language models on scientific text tasks.

### Languages

The dataset primarily contains papers in English.

## Dataset Structure

### Data Instances

Each instance contains:
- Paper ID
- Title
- Authors
- Published date
- Categories
- Summary
- Full text of the paper
- Additional metadata

### Data Fields

- `paper_id`: Unique identifier for the paper
- `title`: Title of the paper
- `authors`: List of authors
- `published_date`: Publication date in ISO format
- `categories`: List of subject categories
- `summary`: Abstract or summary of the paper
- `full_text`: Full text content of the paper
- `metadata`: Additional metadata about the paper (if available)
- `input`: Example prompt for LLM training
- `output`: Expected response (summary)

### Data Splits

The dataset {{ data_splits_description }}.

## Dataset Creation

### Source Data

The papers were sourced from research repositories and processed using PaperTuner.

## Considerations for Using the Data

### License

This dataset is provided under the {{ dataset_license }} license.

### Citation Information

Please cite the original papers if you use this dataset in your research.
