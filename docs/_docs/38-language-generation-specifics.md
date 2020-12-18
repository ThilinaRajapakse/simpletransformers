---
title: Language Generation Specifics
permalink: /docs/language-generation-specifics/
excerpt: "Specific notes for Language Generation"
last_modified_at: 2020/12/08 00:04:09
toc: true
---

The Language Generation model provides an easy way to use a trained Transformer model for language generation. Unlike the other models in Simple Transformers, the `LanguageGenerationModel` does not support training of any kind. If you wish to train or fine-tune models for language generatin, please see the [Language Modeling](/docs/lm-specifics/) section.

**Tip:** This [Medium article](https://towardsdatascience.com/understanding-electra-and-training-an-electra-language-model-3d33e3a9660d?source=friends_link&sk=2b4b4a79954e3d7c84ab863efaea8c65) provides more information on fine-tuning language models and language generation.
{: .notice--success}


## Usage Steps

The process of performing Language Generation in Simple Transformers consists of initializing a model and generating sequences.

1. Initialize a `LanguageGenerationModel`
2. Generate text with `generate()`


## Supported Model Types

New model types are regularly added to the library. Language Modeling tasks currently supports the model types given below.

| Model          | Model code for `LanguageGenerationModel` |
| -------------- | ---------------------------------------- |
| CTRL           | ctrl                                     |
| GPT-2          | gpt2                                     |
| OpenAI GPT     | openai-gpt                               |
| Transformer-XL | transfo-xl                               |
| XLM            | xlm                                      |
| XLNet          | xlnet                                    |

**Tip:** The model code is used to specify the `model_type` in a Simple Transformers model.
{: .notice--success}
