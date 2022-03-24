---
title: Conversational AI Model
permalink: /docs/convAI-model/
excerpt: "Conversational AI Model"
last_modified_at: 2020/09/06 21:33:07
toc: true
---

## `ConvAIModel`

The `ConvAIModel` class is used for Conversational AI.

To create a `ConvAIModel`, you must specify a `model_type` and a `model_name`.

- `model_type` should be one of the model types from the [supported models](/docs/convAI-specifics/) (e.g. gpt2, gpt)
- `model_name` specifies the exact architecture and trained weights to use. This may be a Hugging Face Transformers compatible pre-trained model, a community model, or the path to a directory containing model files.

    **Note:** For a list of standard pre-trained models, see [here](https://huggingface.co/transformers/pretrained_models.html).
    {: .notice--info}

    **Note:** For a list of community models, see [here](https://huggingface.co/models).
    {: .notice--info}

    You may use any of these models provided the `model_type` is supported.

**Tip:** A GPT model trained for conversation is available from Hugging Face [here](https://s3.amazonaws.com/models.huggingface.co/transfer-learning-chatbot/gpt_personachat_cache.tar.gz). You can use it by downloading the model and extracting it to `gpt_personachat_cache`.
{: .notice--success}

```python
from simpletransformers.conv_ai import ConvAIModel


model = ConvAIModel(
    "gpt", "gpt_personachat_cache"
)
```

**Note:** For more information on working with Simple Transformers models, please refer to the [General Usage section](/docs/usage/#creating-a-task-specific-model).
{: .notice--info}


### Configuring a `ConvAIModel`

`ConvAIModel` has several task-specific configuration options.

| Argument                 | Type  | Default | Description                                                                                |
|--------------------------|-------|---------|--------------------------------------------------------------------------------------------|
| num_candidates           | int   | 2       | Number of candidates for training                                                          |
| personality_permutations | int   | 1       | Number of permutations of personality sentences                                            |
| max_history              | int   | 2       | Number of previous exchanges to keep in history                                            |
| lm_coef                  | float | 2.0     | Language Model loss coefficient                                                            |
| mc_coef                  | float | 1.0     | Multiple-choice loss coefficient                                                           |
| do_sample                | bool  | 20      | If set to False greedy decoding is used. Otherwise sampling is used.                       |
| max_length               | int   | -1      | The maximum length of the sequence to be generated. Between 0 and infinity. Default to 20. |
| min_length               | int   | 1       | The minimum length of the sequence to be generated. Between 0 and infinity. Default to 20. |
| temperature              | float | 0.7     | Sampling softmax temperature                                                               |
| top_k                    | int   | 0       | Filter top-k tokens before sampling (<=0: no filtering)                                    |
| top_p                    | float | 0.9     | Nucleus filtering (top-p) before sampling (<=0.0: no filtering)                            |

```python
from simpletransformers.conv_ai import ConvAIModel, ConvAIArgs


model_args = ConvAIArgs()
model_args.max_history = 5

model = ConvAIModel(
    "gpt",
    "gpt_personachat_cache",
    args=model_args
)
```

**Note:** For configuration options common to all Simple Transformers models, please refer to the [Configuring a Simple Transformers Model section](/docs/usage/#configuring-a-simple-transformers-model).
{: .notice--info}


## `Class ConvAIModel`

> *simpletransformers.conv_ai.ConvAIModel*{: .function-name}(self, model_name, args=None, use_cuda=True, cuda_device=-1, **kwargs,)

Initializes a ConvAIModel model.
{: .function-text}

> Parameters
{: .parameter-blockquote}

* **model_type** *(`str`)* - The type of model to use ([model types](/docs/convAI-specifics/#supported-model-types))

* **model_name** *(`str`)* - The exact architecture and trained weights to use. This may be a Hugging Face Transformers compatible pre-trained model, a community model, or the path to a directory containing model files.

* **args** *(`dict`, optional)* - [Default args](/docs/usage/#configuring-a-simple-transformers-model) will be used if this parameter is not provided. If provided, it should be a dict containing the args that should be changed in the default args or a `ConvAIArgs` object.

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


## Training a `ConvAIModel`

The `train_model()`  method is used to train the model.

```python
model.train_model(train_file)
```

> *simpletransformers.conv_ai.ConvAIModel*{: .function-name}(self, train_file, output_dir=None, show_running_loss=True, args=None, eval_file=None, verbose=True, **kwargs)

Trains the model using 'train_file'
{: .function-text}

> Parameters
{: .parameter-blockquote}

* **train_file** - Path to a JSON file containing the training data.
If not given, train dataset from PERSONA-CHAT will be used. The model will be trained on this data. Refer to the [Conversational AI Data Formats](/docs/convAI-data-formats) section for the correct formats.

* **output_dir** *(`str`, optional)* - The directory where model files will be saved. If not given, `self.args['output_dir']` will be used.

* **show_running_loss** *(`bool`, optional)* - If True, the running loss (training loss at current step) will be logged to the console.

* **args** *(`dict`, optional)* - A dict of configuration options for the `ConvAIModel`. Any changes made will persist for the model.

* **eval_file** *(optional)* - Evaluation data (same format as train_file) against which evaluation will be performed when evaluate_during_training is enabled. If not given when evaluate_during_training is enabled, the evaluation data from PERSONA-CHAT will be used.

* **kwargs** *(optional)* - Additional metrics that should be calculated. Pass in the metrics as keyword arguments *(name of metric: function to calculate metric)*. Refer to the [additional metrics](/docs/tips-and-tricks/#additional-evaluation-metrics) section.
E.g. `f1=sklearn.metrics.f1_score`.
A metric function should take in two parameters. The first parameter will be the true labels, and the second parameter will be the predictions.
{: .parameter-list}

> Returns
{: .returns}

* `None`
{: .return-list}

**Note:** For more details on training models with Simple Transformers, please refer to the [Tips and Tricks](/docs/tips-and-tricks) section.
{: .notice--info}



## Evaluating a `ConvAIModel`

The `eval_model()`  method is used to evaluate the model.

The following metrics will be calculated by default:

* `language_model_loss`
* `f1_score`


```python
result, model_outputs, wrong_preds = model.eval_model(eval_file)
```

> *simpletransformers.conv_ai.ConvAIModel.eval_model*{: .function-name}(self, eval_file,
> output_dir=None, verbose=True, silent=False, **kwargs)

Evaluates the model using 'eval_file'
{: .function-text}

> Parameters
{: .parameter-blockquote}

* **eval_file** - Path to JSON file containing evaluation data OR list of Python dicts in the correct format. The model will be evaluated on this data. Refer to the [Conversational AI Data Formats](/docs/convAI-data-formats) section for the correct formats.

* **output_dir** *(`str`, optional)* - The directory where model files will be saved. If not given, `self.args['output_dir']` will be used.

* **verbose** *(`bool`, optional)* - If verbose, results will be printed to the console on completion of evaluation.

* **verbose_logging** *(`bool`, optional)* - Log info related to feature conversion and writing predictions.

* **silent** *(`bool`, optional)* - If silent, tqdm progress bars will be hidden.

* **kwargs** *(optional)* - Additional metrics that should be calculated. Pass in the metrics as keyword arguments *(name of metric: function to calculate metric)*. Refer to the [additional metrics](/docs/tips-and-tricks/#additional-evaluation-metrics) section.
E.g. `f1=sklearn.metrics.f1_score`.
A metric function should take in two parameters. The first parameter will be the true labels, and the second parameter will be the predictions.
{: .parameter-list}

> Returns
{: .returns}

* **result** *(`dict`)* - Dictionary containing evaluation results. (f1_score, language_model_loss)
{: .return-list}

**Note:** For more details on evaluating models with Simple Transformers, please refer to the [Tips and Tricks](/docs/tips-and-tricks) section.
{: .notice--info}


## Interacting with a `ConvAIModel`

Two methods are available for interacting with a ConvAIModel.

1. `interact()` - Used to start an interactive terminal session with the model
2. `interact_single` - Used to communicate with the model through single messages, i.e. by providing the current message and the history of the conversation.

```python
personality=[
    "My name is Geralt.",
    "I hunt monsters.",
    "I say hmm a lot.",
]

# Interactive session (looped)
model.interact(
    personality=personality
)

# Single interaction
history = [
    "Hello, what's your name?",
    "Geralt",
    "What do you do for a living?",
    "I hunt monsters",
]

response, history = model.interact_single(
    "Is it dangerous?",
    history,
    personality=personality
)

```

> *simpletransformers.conv_ai.ConvAIModel.interact*{: .function-name}(self, personality=None)

Interact with a model in the terminal.
{: .function-text}

> Parameters
{: .parameter-blockquote}

- **personality** *(`list`, optional)*: A list of sentences that the model will use to build a personality.
If not given, a random personality from PERSONA-CHAT will be picked.
{: .parameter-list}

> Returns
{: .returns}

* `None`
{: .return-list}


> *simpletransformers.conv_ai.ConvAIModel.interact_single*{: .function-name}(self, message, history, personality=None, encode_history=True)`

Get Response from the model based on the history and message
{: .function-text}

> Parameters
{: .parameter-blockquote}

- **message** *(`str`)*: A message to be sent to the model.

- **history** *(`list`)*: A list of sentences that repersents the interaction history between the model and the user.

- **personality** *(`list`, optional)*: A list of sentences that the model will use to build a personality.

- **encode_history** *(`bool`, optional)*: If True, the history should be in text (string) form. The history will be tokenized and encoded.
{: .parameter-list}

> Returns
{: .returns}

* **out_text** *(`str`)* - The response generated by the model based on the personality, history and message.

* **history** *(`list`)* - The updated history of the conversation. If encode_history is True, this will be in text form. If not, it will be in encoded form.
{: .return-list}


