---
title: Language Modeling Specifics
permalink: /docs/lm-specifics/
excerpt: "Specific notes for Language Modeling tasks."
last_modified_at: 2020-05-02 17:58:53
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


## Language Model Fine-Tuning vs Training a Language Model From Scratch

There are two main uses of the Language Modeling task. The overall process is the same with the key difference being that *language model fine-tuning* starts from a pre-trained model whereas *training a language model from scratch* starts with an untrained, randomly initialized model. The `LanguageModelingModel` is used for both sub-tasks.

### Language Model Fine-Tuning

When *fine-tuning* a language model, an existing pre-trained model (e.g. `bert-base-cased`, `roberta-base`, etc.) is pre-trained further on a new unlabelled text corpus (using the original, pre-trained tokenizer). Generally, this is valuable when you wish to use a pre-trained for a particular task where the language used may be highly technical and/or specialized. This technique was successfully employed in the [SciBERT paper](https://www.aclweb.org/anthology/D19-1371.pdf).

### Training a Language Model From Scratch

Here, an untrained, randomly initialized model is pre-trained on a large corpus of text *from scratch*. This will also train a tokenizer optimized for the given corpus of text. This is particularly useful when training a language model for languages which do not have publicly available pre-trained models.

## Usage Steps

The process of performing Language Modeling in Simple Transformers does not deviate from the [standard pattern](/docs/usage/#task-specific-models).

1. Initialize a `QuestionAnsweringModel`
2. Train the model with `train_model()`
3. Evaluate the model with `eval_model()`
4. Make predictions on (unlabelled) data with `predict()`


## Supported Model Types

New model types are regularly added to the library. Language Modeling tasks currently supports the model types given below.

| Model      | Model code for `NERModel` |
| ---------- | ------------------------- |
| ALBERT     | albert                    |
| BERT       | bert                      |
| DistilBERT | distilbert                |
| ELECTRA    | electra                   |
| RoBERTa    | roberta                   |
| XLM        | xlm                       |
| XLNet      | xlnet                     |

**Tip:** The model code is used to specify the `model_type` in a Simple Transformers model.
{: .notice--success}


## ELECTRA Models

The ELECTRA model consists of a generator model and a discriminator model. Because of this, you can configure an ELECTRA model in several ways by using the options below.

- `model_type` must be set to `"electra"`.
- To load a saved ELECTRA model, you can provide the path to the save files as `model_name`.
- However, the pre-trained ELECTRA models made public by Google are available as separate generator and discriminator models. When starting from these models (Language Model fine-tuning), set `model_name` to `None` and provide the pre-trained models as `generator_name` and `discriminator_name`. These two parameters can also be used to load locally saved generator and/or discriminator models.

    ```python
    model = LanguageModelingModel(
        "electra",
        None,
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