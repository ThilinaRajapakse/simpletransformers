---
title: Language Modeling Model
permalink: /docs/lm-model/
excerpt: "LanguageModelingModel for Language Modeling tasks."
last_modified_at: 2020/08/10 01:15:58
toc: true
---


## `LanguageModelingModel`

The `LanguageModelingModel` class is used for Language Modeling. This can be used for both Language Model fine-tuning and for training a Language Model from scratch.

To create a `LanguageModelingModel`, you must specify a `model_type` and a `model_name`.


**Note:** `model_name` is set to `None` to train a Language Model from scratch.
{: .notice--info}


- `model_type` should be one of the model types from the [supported models](/docs/lm-specifics/#supported-model-types) (e.g. bert, electra, gpt2)
- `model_name` specifies the exact architecture and trained weights to use. This may be a Hugging Face Transformers compatible pre-trained model, a community model, the path to a directory containing model files, or `None` to train a Language Model from scratch.

    **Note:** For a list of standard pre-trained models, see [here](https://huggingface.co/transformers/pretrained_models.html).
    {: .notice--info}

    **Note:** For a list of community models, see [here](https://huggingface.co/models).
    {: .notice--info}

    You may use any of these models provided the `model_type` is supported.


### Language Model fine-tuning

```python
from simpletransformers.language_modeling import (
    LanguageModelingModel,
)


model = LanguageModelingModel("bert", "bert-base-cased")

```

### Language Model training from scratch

```python
from simpletransformers.language_modeling import (
    LanguageModelingModel,
)

model = LanguageModelingModel(
    "bert", None
)
```

**Note:** For more information on working with Simple Transformers models, please refer to the [General Usage section](/docs/usage/#creating-a-task-specific-model).
{: .notice--info}


### Configuring a `LanguageModelingModel`

`LanguageModelingModel` has several task-specific configuration options.


| Argument             | Type                        | Default                                            | Description                                                                                                                                                                                                                        |
| -------------------- | --------------------------- | -------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| block_size           | int                         | `-1`                                               | Optional input sequence length after tokenization. The training dataset will be truncated in block of this size for training. Default to the model max input length for single sentence inputs (take into account special tokens). |
| clean_text           | bool                        | `True`                                             | Performs invalid character removal and whitespace cleanup on text.                                                                                                                                                                 |
| config_name          | str                         | `None`                                             | Name of pretrained config or path to a directory containing a config.json file.                                                                                                                                                    |
| dataset_class        | Subclass of Pytorch Dataset | `None`                                             | A custom dataset class to use instead of `dataset_type`.                                                                                                                                                                           |
| dataset_type         | str                         | `"simple"`                                         | Choose between `simple`, `line_by_line`, and `text` dataset types. (See Dataset types below)                                                                                                                                       |
| discriminator_config | dict                        | `{}`                                               | Key-values given here will override the default values used in an Electra discriminator model Config.  (See [ELECTRA models](/docs/lm-specifics/#electra-models))                                                                  |
| generator_config     | dict                        | `{}`                                               | Key-values given here will override the default values used in an Electra generator model Config.  (See [ELECTRA models](/docs/lm-specifics/#electra-models))                                                                      |
| handle_chinese_chars | bool                        | `True`                                             | Whether to tokenize Chinese characters. If `False`, Chinese text will not be tokenized properly.                                                                                                                                   |
| max_steps            | int                         | `-1`                                               | If `max_steps` > 0: set total number of training steps to perform. Supersedes num_train_epochs.                                                                                                                                    |
| min_frequency        | int                         | `2`                                                | Minimum frequency required for a word to be added to the vocabulary.                                                                                                                                                               |
| mlm                  | bool                        | `True`                                             | Train with masked-language modeling loss instead of language modeling. Set to `False` for models which don't use Masked Language Modeling.                                                                                         |
| mlm_probability      | float                       | `0.15`                                             | Ratio of tokens to mask for masked language modeling loss.                                                                                                                                                                         |
| sliding_window       | bool                        | `False`                                            | Whether sliding window technique should be used when preparing data. Only works with SimpleDataset.                                                                                                                                |
| special_tokens       | list                        | *Defaults to the special_tokens of the model used* | List of special tokens to be used when training a new tokenizer.                                                                                                                                                                   |
| stride               | float                       | `0.8`                                              | A fraction of the max_seq_length to use as the stride when using a sliding window                                                                                                                                                  |
| strip_accents        | bool                        | `True`                                             | Strips accents from a piece of text.                                                                                                                                                                                               |
| tokenizer_name       | str                         | `None`                                             | Name of pretrained tokenizer or path to a directory containing tokenizer files.                                                                                                                                                    |
| vocab_size           | int                         | `None`                                             | The maximum size of the vocabulary of the tokenizer. Required when training a tokenizer.                                                                                                                                           |


**Note:** For configuration options common to all Simple Transformers models, please refer to the [Configuring a Simple Transformers Model section](/docs/usage/#configuring-a-simple-transformers-model).
{: .notice--info}


{% capture notice-text %}
- `simple` (or None) - Each line in the train file is considered to be a single, separate sample. `sliding_window` can be set to True to automatically split longer sequences into samples of length `max_seq_length`. Uses multiprocessing for significantly improved performance on multi-core systems.

- `line_by_line` - Treats each line in the train file as a separate sample. Uses tokenizers from the Hugging Face tokenizers library.

- `text` - Treats each line in the train file as a separate sample. Uses default tokenizers.

*Using `simple` is recommended.*

{% endcapture %}

<div class="notice--info">
  <strong>Dataset types</strong>
  {{ notice-text | markdownify }}
</div>

### Configuring the architecture of a Language Model

When training a Language Model from scratch, you are free to define your own architecture. For all model types except ELECTRA, this is controlled through the `config` entry in the model `args` dict. For ELECTRA, the generator and the discriminator architectures can be specified through the `generator_config`, and `discriminator_config` entries respectively.

If not specified, the default configurations (the base architecture) for the given model will be used. For all available parameters and their default values, please refer to the Hugging Face docs for the relevant config class (E.g. [BERT config](https://huggingface.co/transformers/model_doc/bert.html#bertconfig)).

A custom BERT architecture:

```python
from simpletransformers.language_modeling import LanguageModelingModel, LanguageModelingArgs


model_args = LanguageModelingArgs()
model_args.config = {
    "num_hidden_layers": 2
}
 model_args.vocab_size = 5000

model = LanguageModelingModel(
    "bert", None, args=model_args, train_files=train_file
)

```

A custom ELECTRA architecture:

```python
from simpletransformers.language_modeling import LanguageModelingModel, LanguageModelingArgs


model_args = LanguageModelingArgs()
model_args.vocab_size = 30000
model_args.generator_config = {
    "embedding_size": 512,
    "hidden_size": 256,
    "num_hidden_layers": 4,
}
model_args.discriminator_config = {
    "embedding_size": 512,
    "hidden_size": 256,
    "num_hidden_layers": 16,
}

model = LanguageModelingModel(
    "electra",
    None,
    args=model_args,
    train_files=train_file
)
```


## `Class LanguageModelingModel`

> *simpletransformers.language_modeling.LanguageModelingModel*{: .function-name}(self, model_type, model_name, args=None, use_cuda=True, cuda_device=-1, **kwargs,)

Initializes a LanguageModelingModel model.
{: .function-text}

> Parameters
{: .parameter-blockquote}

* **model_type** *(`str`)* - The type of model to use ([model types](/docs/lm-specifics/#supported-model-types))

* **model_name** *(`str`)* - The exact architecture and trained weights to use. This may be a Hugging Face Transformers compatible pre-trained model, a community model, the path to a directory containing model files, or None to train a Language Model from scratch.

* **generator_name** *(`str`, optional)* - A pretrained model name or path to a directory containing an ELECTRA generator model. (See [ELECTRA models](/docs/lm-specifics/#electra-models))

* **discriminator_name** *(`str`, optional)* - A pretrained model name or path to a directory containing an ELECTRA discriminator model. (See [ELECTRA models](/docs/lm-specifics/#electra-models))

* **train_files** *(`str` or `List`, optional)* - A file or a List of files to be used when training the tokenizer. Required if the tokenizer is being trained from scratch.

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


## Training a `LanguageModelingModel`

The `train_model()`  method is used to train the model.

```python
model.train_model(train_file)
```

> *simpletransformers.language_modeling.LanguageModelingModel*{: .function-name}(self, train_file, output_dir=None, show_running_loss=True, args=None, eval_file=None, verbose=True, **kwargs)

Trains the model using 'train_file'
{: .function-text}

> Parameters
{: .parameter-blockquote}

* **train_file** *(`str`)* - Path to text file containing the text to train the language model on. The model will be trained on this data. Refer to the [Language Modeling Data Formats](/docs/lm-data-formats) section for the correct formats.

* **output_dir** *(`str`, optional)* - The directory where model files will be saved. If not given, `self.args['output_dir']` will be used.

* **show_running_loss** *(`bool`, optional)* - If True, the running loss (training loss at current step) will be logged to the console.

* **args** *(`dict`, optional)* - A dict of configuration options for the `LanguageModelingModel`. Any changes made will persist for the model.

* **eval_file** *(`str`, optional)* - Evaluation data (same format as train_file) against which evaluation will be performed when evaluate_during_training is enabled. Is required if evaluate_during_training is enabled.

* **kwargs** *(optional)* - Additional metrics are not currently supported for Language Modeling.
{: .parameter-list}

> Returns
{: .returns}

* `None`
{: .return-list}

**Note:** For more details on training models with Simple Transformers, please refer to the [Tips and Tricks](/docs/tips-and-tricks) section.
{: .notice--info}


## Evaluating a `LanguageModelingModel`

The `eval_model()`  method is used to evaluate the model.

The following metrics will be calculated by default:

* `perplexity` - Perplexity is a score used to evaluate language models.
* `eval_loss` - Cross Entropy Loss for eval_file


```python
result = model.eval_model(eval_file)
```

> *simpletransformers.language_modeling.LanguageModelingModel.eval_model*{: .function-name}(self, eval_file,
> output_dir=None, verbose=True, silent=False)

Evaluates the model using 'eval_file'
{: .function-text}

> Parameters
{: .parameter-blockquote}

* **eval_file** *(`str`)* - Path to text file containing the text to evaluate the language model on. The model will be evaluated on this data. Refer to the [Language Modeling Data Formats](/docs/lm-data-formats) section for the correct formats.

* **output_dir** *(`str`, optional)* - The directory where model files will be saved. If not given, `self.args['output_dir']` will be used.

* **verbose** *(`bool`, optional)* - If verbose, results will be printed to the console on completion of evaluation.

* **silent** *(`bool`, optional)* - If silent, tqdm progress bars will be hidden.
{: .parameter-list}

> Returns
{: .returns}

* **result** *(`dict`)* - Dictionary containing evaluation results.

* **texts** *(`list`)* - A dictionary containing the 3 dictionaries `correct_text`, `similar_text`, and `incorrect_text`.
{: .return-list}

**Note:** For more details on evaluating models with Simple Transformers, please refer to the [Tips and Tricks](/docs/tips-and-tricks) section.
{: .notice--info}
