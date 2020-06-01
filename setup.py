from setuptools import find_packages, setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="simpletransformers",
    version="0.32.1",
    author="Thilina Rajapakse",
    author_email="chaturangarajapakshe@gmail.com",
    description="An easy-to-use wrapper library for the Transformers library.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ThilinaRajapakse/simpletransformers/",
    packages=find_packages(),
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
        "tqdm",
        "regex",
        "transformers>=2.9.1",
        "scipy",
        "scikit-learn",
        "seqeval",
        "tensorboardx",
        "pandas",
        "tokenizers",
    ],
)
