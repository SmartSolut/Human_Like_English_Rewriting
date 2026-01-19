"""
Setup script for Human-Like English Rewriting System
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="paraphrase-system",
    version="1.0.0",
    author="Your Name",
    description="Human-Like English Rewriting System",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/paraphrase",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.10",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "paraphrase-download=src.data.downloader:main",
            "paraphrase-preprocess=src.data.preprocessor:main",
            "paraphrase-train=src.training.trainer:main",
            "paraphrase-evaluate=src.evaluation.evaluator:main",
            "paraphrase-api=src.api.main:main",
        ],
    },
)








