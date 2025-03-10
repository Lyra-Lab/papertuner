from setuptools import setup, find_packages

setup(
    name="papertuner",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "requests",
        "arxiv",
        "datasets",
        "huggingface-hub",
        "google-generativeai",
        "tqdm",
        "pyyaml",
        "PyMuPDF",  # Adding PyMuPDF to the install_requires list
        "fitz"
    ],
    author="Your Name",
    author_email="your.email@example.com",
    description="A simple tool to generate datasets for fine-tuning LLMs from academic papers",
    keywords="machine learning, dataset, arxiv, huggingface",
    url="https://github.com/yourusername/papertuner",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
)
