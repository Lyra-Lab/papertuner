from setuptools import setup, find_packages

setup(
    name="papertuner",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "arxiv",
        "datasets",
        "ollama",
        "pdf2image",
        "pillow",
        "requests",
        "tqdm",
        "PyPDF2",
    ],
    extras_require={
        "huggingface": [
            "torch>=1.10.0",
            "transformers>=4.25.0",
        ],
    },
    description="A tool for creating fine-tuning datasets from scientific papers",
    author="Your Name",
    author_email="your.email@example.com",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)
