from setuptools import setup, find_packages

setup(
    name="papertuner",
    version="0.1.0",
    packages=find_packages(),
    description="A tool for fine-tuning language models on research papers",
    author="Denis",
    install_requires=[
        "torch",
        "unsloth",
        "datasets",
        "sentence-transformers",
        "trl",
        "vllm",
        "arxiv",
        "openai",
        "PyMuPDF",
        "huggingface_hub",
        "tenacity",
        "tqdm",
        "requests"
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
    entry_points={
        "console_scripts": [
            "papertuner-train=papertuner.train.cli:main",
            "papertuner-dataset=papertuner.data.cli:main",
        ],
    },
)
