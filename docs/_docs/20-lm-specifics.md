---
title: Language Modeling Specifics
permalink: /docs/lm-specifics/
excerpt: "Specific notes for Language Modeling tasks."
last_modified_at: 2021/07/24 13:16:18
toc: true
---
The idea of (probabilistic) language modeling is to calculate the probability of a sentence (or sequence of words). This can be used to find the probabilities for the next word in a sequence, or the probabilities for possible words at a given (masked) position.

The commonly used *pre-training* strategies reflect this idea. For example;

- Masked Language Modeling - Predict the randomly masked (hidden) words in a sequence of text (E.g. BERT).
- Next word prediction - Predict the next word, given all the previous words (E.g. GPT-2).
- ELECTRA - Predict whether each word has been replaced by a generated word or whether it is an original.

To perform these tasks successfully, the model has to learn the probabilities of a sequence of words, i.e. language modeling.

**Tip:** This [Medium article](https://towardsdatascience.com/understanding-electra-and-training-an-electra-language-model-3d33e3a9660d?source=friends_link&sk=2b4b4a79954e3d7c84ab863efaea8c65) provides more information on pre-training and language modeling.
{: .notice--success}

**Tip:** This [Medium article](https://towardsdatascience.com/understanding-electra-and-training-an-electra-language-model-3d33e3a9660d?source=friends_link&sk=2b4b4a79954e3d7c84ab863efaea8c65) provides more information on fine-tuning language models and language generation.
{: .notice--success}

## Language Model Fine-Tuning vs Training a Language Model From Scratch

There are two main uses of the Language Modeling task. The overall process is the same with the key difference being that *language model fine-tuning* starts from a pre-trained model whereas *training a language model from scratch* starts with an untrained, randomly initialized model. The `LanguageModelingModel` is used for both sub-tasks.

### Language Model Fine-Tuning

When *fine-tuning* a language model, an existing pre-trained model (e.g. `bert-base-cased`, `roberta-base`, etc.) is pre-trained further on a new unlabelled text corpus (using the original, pre-trained tokenizer). Generally, this is valuable when you wish to use a pre-trained for a particular task where the language used may be highly technical and/or specialized. This technique was successfully employed in the [SciBERT paper](https://www.aclweb.org/anthology/D19-1371.pdf).

### Training a Language Model From Scratch

Here, an untrained, randomly initialized model is pre-trained on a large corpus of text *from scratch*. This will also train a tokenizer optimized for the given corpus of text. This is particularly useful when training a language model for languages which do not have publicly available pre-trained models.

This also gives you the option to create a Transformer model with a custom architecture.

## Usage Steps

The process of performing Language Modeling in Simple Transformers follows the [standard pattern](/docs/usage/#task-specific-models). However, there is no predict functionality.

1. Initialize a `LanguageModelingModel`
2. Train the model with `train_model()`
3. Evaluate the model with `eval_model()`

## Supported Model Types

New model types are regularly added to the library. Language Modeling tasks currently supports the model types given below.

| Model      | Model code for `LanguageModelingModel` |
| ---------- | -------------------------------------- |
| BERT       | bert                                   |
| BigBird    | bigbird                                |
| CamemBERT  | camembert                              |
| DistilBERT | distilbert                             |
| ELECTRA    | electra                                |
| GPT-2      | gpt2                                   |
| Longformer | longformer                             |
| OpenAI GPT | openai-gpt                             |
| RemBERT    | rembert                                |
| RoBERTa    | roberta                                |
| XLMRoBERTa | xlmroberta                             |

**Tip:** The model code is used to specify the `model_type` in a Simple Transformers model.
{: .notice--success}

## ELECTRA Models

The ELECTRA model consists of a generator model and a discriminator model.

### Configuring an ELECTRA model

You can configure an ELECTRA model in several ways by using the options below.

- `model_type` must be set to `electra`.
- To load a saved ELECTRA model, you can provide the path to the save files as `model_name`.
- However, the pre-trained ELECTRA models made public by Google are available as separate generator and discriminator models. When starting from these models (Language Model fine-tuning), set `model_name` to `electra` and provide the pre-trained models as `generator_name` and `discriminator_name`. These two parameters can also be used to load locally saved generator and/or discriminator models.

  ```python
  model = LanguageModelingModel(
      "electra",
      "electra",
      generator_name="outputs/generator_model",
      discriminator_name="outputs/disciminator_model",
  )

  ```
- When training an ELECTRA language model from scratch, you can define the architecture by using the `generator_config` and `discriminator_config` in the `args` dict. The [default values](https://huggingface.co/transformers/model_doc/electra.html#electraconfig) will be used for any config parameters that aren't specified.

  ```python
  model_args = {
      "vocab_size": 52000,
      "generator_config": {
          "embedding_size": 128,
          "hidden_size": 256,
          "num_hidden_layers": 3,
      },
      "discriminator_config": {
          "embedding_size": 128,
          "hidden_size": 256,
      },
  }

  train_file = "data/train_all.txt"

  model = LanguageModelingModel(
      "electra",
      None,
      args=model_args,
      train_files=train_file,
  )

  ```

Refer to the [Language Modeling Minimal Start](/docs/lm-minimal-start/) for full (minimal) examples.

### Saving ELECTRA models

When using ELECTRA models for downstream tasks, the ELECTRA developers recommend using the discriminator model only. Because of this, Simple Transformers will save the generator and discriminator models separately at the end of training. The discriminator model can then be used for downstream tasks.

E.g.:

```python
model = ClassificationModel("electra", "outputs/discriminator_model")
```

The discriminator and generator models are not saved separately for any intermediate checkpoints as it is not necessary to save them separately unless they are to be used for a downstream task. However, you can manually save the discriminator and/or generator model separately from any checkpoint by using the `save_discriminator()` and `save_generator()` methods.

E.g.:

```python
lm_model = LanguageModelingModel("electra", "outputs/checkpoint-1-epoch-1")
lm_model.save_discriminator("outputs/checkpoint-1-epoch-1")

classification_model = ClassificationModel("electra", "outputs/checkpoint-1-epoch-1/discriminator_model")
```

**Note:** Both `save_discriminator()` and `save_generator()` methods takes in an optional `output_dir` argument which specifies where the model should be saved.
{: .notice--info}

## Distributed Training

Simple Transformers supports distributed language model training.

**Tip:** You can find an example script [here](https://github.com/ThilinaRajapakse/simpletransformers/blob/master/examples/language_generation/train_new_lm.py).
{: .notice--success}

You can launch distributed training as shown below.

```bash
python -m torch.distributed.launch --nproc_per_node=4 train_new_lm.py
```
