from setuptools import find_packages, setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="simpletransformers-le",  # le for longformer-electra
    version="0.0.1.dev7",
    author="Joel Niklaus",
    author_email="me@joelniklaus.ch",
    description="An easy-to-use wrapper library for the Transformers library.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/JoelNiklaus/simpletransformers/",
    packages=find_packages(),
    scripts=["bin/simple-viewer"],
    classifiers=[
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.6",
    install_requires=[
        "numpy",
        "requests",
        "tqdm>=4.47.0",
        "regex",
        "transformers>=4.6.0",
        "datasets",
        "scipy",
        "scikit-learn",
        "seqeval",
        "tensorboard",
        "pandas",
        "tokenizers",
        "wandb>=0.10.32",
        "streamlit",
        "sentencepiece",
    ],
)
