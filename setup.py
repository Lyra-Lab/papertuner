from setuptools import setup, find_packages

setup(
    name="paper-dataset-generator",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "mistralai",
        "arxiv",
        "tqdm",
        "pydantic",
    ],
    author="Your Name",
    author_email="your.email@example.com",
    description="A tool to create LLM fine-tuning datasets from research papers",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/paper-dataset-generator",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
) 