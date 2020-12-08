---
title: Language Generation Model
permalink: /docs/language-generation-model/
excerpt: "LanguageGenerationModel for Language Generation"
last_modified_at: 2020/12/08 00:21:52
toc: true
---


## `LanguageGenerationModel`

The `LanguageGenerationModel` class is used for Language Generation.

To create a `LanguageGenerationModel`, you must specify a `model_type` and a `model_name`.


**Note:** `model_name` is set to `None` to train a Language Model from scratch.
{: .notice--info}


- `model_type` should be one of the model types from the [supported models](/docs/language-generation-specifics/#supported-model-types)
- `model_name` specifies the exact architecture and trained weights to use. This may be a Hugging Face Transformers compatible pre-trained model, a community model, or the path to a directory containing model files.

    **Note:** For a list of standard pre-trained models, see [here](https://huggingface.co/transformers/pretrained_models.html).
    {: .notice--info}

    **Note:** For a list of community models, see [here](https://huggingface.co/models).
    {: .notice--info}

    You may use any of these models provided the `model_type` is supported.


```python
from simpletransformers.language_generation import (
    LanguageGenerationModel,
)

model = LanguageGenerationModel(
    "gpt2", "gpt2"
)
```

**Note:** For more information on working with Simple Transformers models, please refer to the [General Usage section](/docs/usage/#creating-a-task-specific-model).
{: .notice--info}


### Configuring a `LanguageGenerationModel`

`LanguageGenerationModel` has several task-specific configuration options.


| Argument                    | Type    | Default | Description                                                                                                                                   |
| --------------------------- | ------- | ------- | --------------------------------------------------------------------------------------------------------------------------------------------- |
| do_sample                   | bool    | False   | If set to False greedy decoding is used. Otherwise sampling is used. Defaults to False as defined in configuration_utils.PretrainedConfig.    |
| early_stopping              | bool    | True    | if set to True beam search is stopped when at least num_beams sentences finished per batch.                                                   |
| evaluate_generated_text     | bool    | False   | Generate sequences for evaluation.                                                                                                            |
| length_penalty              | float   | 2.0     | Exponential penalty to the length. Default to 2.                                                                                              |
| max_length                  | int     | 20      | The max length of the sequence to be generated. Between 0 and infinity. Default to 20.                                                        |
| max_steps                   | int     | -1      | Maximum number of training steps. Will override the effect of num_train_epochs.                                                               |
| num_beams                   | int     | 1       | Number of beams for beam search. Must be between 1 and infinity. 1 means no beam search. Default to 1.                                        |
| num_return_sequences        | int     | 1       | The number of samples to generate.                                                                                                            |
| repetition_penalty          | float   | 1.0     | The parameter for repetition penalty. Between 1.0 and infinity. 1.0 means no penalty. Default to 1.0.                                         |
| top_k                       | float   | None    | Filter top-k tokens before sampling (<=0: no filtering)                                                                                       |
| top_p                       | float   | None    | Nucleus filtering (top-p) before sampling (<=0.0: no filtering)                                                                               |
| prompt                    | str     | ""   | A prompt text for the model..                                                                                   |
| stop_token                    | str     | None   | Token at which text generation is stopped.                                                                                   |
| temperature                       | float   | 1.0    | Temperature of 1.0 is the default. Lowering this makes the sampling greedier                                                                               |
| padding_text                    | str     | ""   | Padding text for Transfo-XL and XLNet.                                                                               |
| xlm_language                    | str     | ""   | Optional language when used with the XLM model..                                                                               |
| config_name                    | str     | None   | Name of a pre-trained config or path to a directory containing a saved config.                                                                               |
| tokenizer_name                    | str     | None   | Name of a pre-trained tokenizer or path to a directory containing a saved tokenizer.                                                                               |



**Note:** For configuration options common to all Simple Transformers models, please refer to the [Configuring a Simple Transformers Model section](/docs/usage/#configuring-a-simple-transformers-model).
{: .notice--info}


## `Class LanguageGenerationModel`

> *simpletransformers.language_generation.LanguageGenerationModel*{: .function-name}(self, model_type, model_name, args=None, use_cuda=True, cuda_device=-1, **kwargs,)

Initializes a LanguageGenerationModel model.
{: .function-text}

> Parameters
{: .parameter-blockquote}

* **model_type** *(`str`)* - The type of model to use ([model types](/docs/language-generation-specifics/#supported-model-types))

* **model_name** *(`str`)* - The exact architecture and trained weights to use. This may be a Hugging Face Transformers compatible pre-trained model, a community model, the path to a directory containing model files, or None to train a Language Model from scratch.

* **args** *(`dict`, optional)* - [Default args](/docs/usage/#configuring-a-simple-transformers-model) will be used if this parameter is not provided. If provided, it should be a dict containing the args that should be changed in the default args.

* **use_cuda** *(`bool`, optional)* - Use GPU if available. Setting to False will force model to use CPU only. (See [here](/docs/usage/#to-cuda-or-not-to-cuda))

* **cuda_device** *(`int`, optional)* - Specific GPU that should be used. Will use the first available GPU by default. (See [here](/docs/usage/#selecting-a-cuda-device))

* **kwargs** *(optional)* - For providing proxies, force_download, resume_download, cache_dir and other options specific to the 'from_pretrained' implementation where this will be supplied. (See [here](/docs/usage/#options-for-downloading-pre-trained-models))
{: .parameter-list}

> Returns
{: .returns}

* `None`
{: .return-list}


**Note:** For configuration options common to all Simple Transformers models, please refer to the [Configuring a Simple Transformers Model section](/docs/usage/#configuring-a-simple-transformers-model).
{: .notice--info}


## Generating text with a `LanguageGenerationModel`

The `generate()`  method is used to generate text.

```python
model.generate()
```

> *simpletransformers.language_generation.LanguageGenerationModel*{: .function-name}(self, prompt=None, args=None, verbose=True)

Generate text
{: .function-text}

> Parameters
{: .parameter-blockquote}

* **prompt** *(`str`)* - A prompt text for the model. If given, will override args.prompt


* **args** *(`dict`, optional)* - A dict of configuration options for the `LanguageGenerationModel`. Any changes made will persist for the model.

* **verbose** *(optional)* - If verbose, generated text will be logged to the console. Default is `True`.
{: .parameter-list}

> Returns
{: .returns}

* **generated_sequences** *(`list`)* - Sequences of text generated by the model.
{: .return-list}
