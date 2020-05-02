---
title: Installation
permalink: /docs/installation/
excerpt: "Instructions for installing the Simple Transformers library."
last_modified_at: 2020-05-02 17:56:52
toc: true
---

It's a good idea to always use virtual environments when working with Python packages. 
Anaconda/Miniconda is a package manager that lets you create virtual environments and manage package installations smoothly.

Follow the instructions given below to install Simple Transformers using with Anaconda (or miniconda, a lighter version of anaconda).

## Installation steps

1. Install Anaconda or Miniconda Package Manager from [here](https://www.anaconda.com/distribution/).
2. Create a new virtual environment and install packages.
   ```shell
   conda create -n transformers python pandas tqdm
   conda activate transformers
   ```
3. Using a CUDA capable GPU is recommended.  
   To install Pytorch with CUDA support:  
      ```shell
      conda install pytorch cudatoolkit=10.1 -c pytorch 
      ```
   CPU only:  
      ```shell
      conda install pytorch cpuonly -c pytorch
      ```

4. Install Apex to use FP16 training. Please follow the instructions [here](https://github.com/NVIDIA/apex). (Installing Apex from pip has caused issues for several people.)

5. Install simpletransformers.  
`pip install simpletransformers` 

## Optional

1. Install Weights and Biases (wandb) for experiment tracking and visualizing training in a web browser.  
`pip install wandb`

---
