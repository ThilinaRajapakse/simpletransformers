---
title: T5 Model
permalink: /docs/t5-model/
excerpt: "T5Model for T5 tasks."
last_modified_at: 2020/12/29 17:01:36
toc: true
---


## `T5Model`

The `T5Model` class is used for any NLP task performed with a T5 model or a mT5 model.

To create a `T5Model`, you must specify the `model_type` and `model_name`.

- `model_type` should be one of the model types from the [supported models](/docs/t5-specifics/#supported-model-types) (`t5` or `mt5`)
- `model_name` specifies the exact architecture and trained weights to use. This may be a Hugging Face Transformers compatible pre-trained model, a community model, or the path to a directory containing model files.

    **Note:** For a list of standard pre-trained models, see [here](https://huggingface.co/transformers/pretrained_models.html).
    {: .notice--info}

    **Note:** For a list of community models, see [here](https://huggingface.co/models).
    {: .notice--info}

    You may use any of these models provided they are a T5 model.

```python
from simpletransformers.t5 import T5Model


model = T5Model(
    "t5",
    "t5-base"
)
```

**Note:** For more information on working with Simple Transformers models, please refer to the [General Usage section](/docs/usage/#creating-a-task-specific-model).
{: .notice--info}


### Configuring a `T5Model`

`T5Model` has the following task-specific configuration options.

| Argument                    | Type    | Default | Description                                                                                                                                       |
|-----------------------------|---------|---------|---------------------------------------------------------------------------------------------------------------------------------------------------|
| dataset_class               | Dataset | None    | A custom dataset class to use. (Subclass of Pytorch Dataset)                                                                                      |
| do_sample                   | bool    | False   | If set to False greedy decoding is used. Otherwise sampling is used. Defaults to False as defined in configuration_utils.PretrainedConfig.        |
| early_stopping              | bool    | True    | if set to True beam search is stopped when at least num_beams sentences finished per batch.                                                       |
| evaluate_generated_text     | bool    | False   | Generate sequences for evaluation.                                                                                                                |
| length_penalty              | float   | 2.0     | Exponential penalty to the length. Default to 2.                                                                                                  |
| max_length                  | int     | 20      | The max length of the sequence to be generated. Between 0 and infinity. Default to 20.                                                            |
| max_steps                   | int     | -1      | Maximum number of training steps. Will override the effect of num_train_epochs.                                                                   |
| num_beams                   | int     | 1       | Number of beams for beam search. Must be between 1 and infinity. 1 means no beam search. Default to 1.                                            |
| num_return_sequences        | int     | 1       | The number of samples to generate.                                                                                                                |
| preprocess_inputs           | bool    | True    | Automatically add : and < /s> tokens to train_model() and eval_model() inputs. Automatically add < /s> to each string in to_predict in predict(). |
| repetition_penalty          | float   | 1.0     | The parameter for repetition penalty. Between 1.0 and infinity. 1.0 means no penalty. Default to 1.0.                                             |
| special_tokens_list     | list      | []      | The list of special tokens to be added to the model tokenizer                                                                                         |
| top_k                       | int   | None    | Filter top-k tokens before sampling (<=0: no filtering)                                                                                           |
| top_p                       | float   | None    | Nucleus filtering (top-p) before sampling (<=0.0: no filtering)                                                                                   |
| use_multiprocessed_decoding | bool    | True    | Use multiprocessing when decoding outputs. Significantly speeds up decoding (CPU intensive).                                                      |                                               |

```python
from simpletransformers.t5 import T5Model, T5Args


model_args = T5Args()
model_args.num_train_epochs = 3

model = T5Model(
    "t5-base",
    args=model_args,
)
```

**Note:** For configuration options common to all Simple Transformers models, please refer to the [Configuring a Simple Transformers Model section](/docs/usage/#configuring-a-simple-transformers-model).
{: .notice--info}


## `Class T5Model`

> *simpletransformers.t5.T5Model*{: .function-name}(self, model_name, args=None, use_cuda=True, cuda_device=-1, **kwargs,)

Initializes a T5Model model.
{: .function-text}

> Parameters
{: .parameter-blockquote}

* **model_name** *(`str`)* - The exact architecture and trained weights to use. This may be a Hugging Face Transformers compatible pre-trained model, a community model, or the path to a directory containing model files.

* **args** *(`dict`, optional)* - [Default args](/docs/usage/#configuring-a-simple-transformers-model) will be used if this parameter is not provided. If provided, it should be a dict containing the args that should be changed in the default args or a `T5Args` object.

* **use_cuda** *(`bool`, optional)* - Use GPU if available. Setting to False will force model to use CPU only. (See [here](/docs/usage/#to-cuda-or-not-to-cuda))

* **cuda_device** *(`int`, optional)* - Specific GPU that should be used. Will use the first available GPU by default. (See [here](/docs/usage/#selecting-a-cuda-device))

* **kwargs** *(optional)* - For providing proxies, force_download, resume_download, cache_dir and other options specific to the 'from_pretrained' implementation where this will be supplied. (See [here](/docs/usage/#options-for-downloading-pre-trained-models))
{: .parameter-list}

> Returns
{: .returns}

* `None`
{: .return-list}


## Training a `T5Model`

The `train_model()`  method is used to train the model.

```python
model.train_model(train_data)
```

> *simpletransformers.t5.T5Model.train_model*{: .function-name}(self, train_data, output_dir=None, show_running_loss=True, args=None, eval_data=None, verbose=True, **kwargs)

Trains the model using 'train_data'
{: .function-text}

> Parameters
{: .parameter-blockquote}

* **train_data** - Pandas DataFrame containing the 3 columns - `prefix`, `input_text`, `target_text`.
    - `prefix`: A string indicating the task to perform. (E.g. `"question"`, `"stsb"`)
    - `input_text`: The input text sequence. `prefix` is automatically prepended to form the full input. (<prefix>: <input_text>)
    - `target_text`: The target sequence

* **output_dir** *(`str`, optional)* - The directory where model files will be saved. If not given, `self.args['output_dir']` will be used.

* **show_running_loss** *(`bool`, optional)* - If True, the running loss (training loss at current step) will be logged to the console.

* **args** *(`dict`, optional)* - A dict of configuration options for the `T5Model`. Any changes made will persist for the model.

* **eval_data** *(optional)* - Evaluation data (same format as train_data) against which evaluation will be performed when `evaluate_during_training` is enabled. Is required if `evaluate_during_training` is enabled.

* **kwargs** *(optional)* - Additional metrics that should be calculated. Pass in the metrics as keyword arguments *(name of metric: function to calculate metric)*. Refer to the [additional metrics](/docs/usage/#additional-evaluation-metrics) section.
E.g. `f1=sklearn.metrics.f1_score`.
A metric function should take in two parameters. The first parameter will be the true labels, and the second parameter will be the predictions.
{: .parameter-list}

> Returns
{: .returns}

* `None`
{: .return-list}

**Note:** For more details on evaluating T5 models with custom metrics, please refer to the [Evaluating Generated Sequences](/docs/t5-specifics/#evaluating-generated-sequences) section.
{: .notice--info}

**Note:** For more details on training models with Simple Transformers, please refer to the [Tips and Tricks](/docs/tips-and-tricks) section.
{: .notice--info}


## Evaluating a `T5Model`

The `eval_model()`  method is used to evaluate the model.

The following metrics will be calculated by default:

* `eval_loss` - Model loss over the evaluation data


```python
result = model.eval_model(eval_data)
```

> *simpletransformers.t5.T5Model.eval_model*{: .function-name}(self, eval_data,
> output_dir=None, verbose=True, silent=False, **kwargs)

Evaluates the model using 'eval_data'
{: .function-text}

> Parameters
{: .parameter-blockquote}

* **eval_data** - Pandas DataFrame containing the 3 columns - `prefix`, `input_text`, `target_text`.
    - `prefix`: A string indicating the task to perform. (E.g. `"question"`, `"stsb"`)
    - `input_text`: The input text sequence. `prefix` is automatically prepended to form the full input. (<prefix>: <input_text>)
    - `target_text`: The target sequence

* **output_dir** *(`str`, optional)* - The directory where model files will be saved. If not given, `self.args['output_dir']` will be used.

* **verbose** *(`bool`, optional)* - If verbose, results will be printed to the console on completion of evaluation.

* **silent** *(`bool`, optional)* - If silent, tqdm progress bars will be hidden.

* **kwargs** *(optional)* - Additional metrics that should be calculated. Pass in the metrics as keyword arguments *(name of metric: function to calculate metric)*. Refer to the [additional metrics](/docs/usage/#additional-evaluation-metrics) section.
E.g. `f1=sklearn.metrics.f1_score`.
A metric function should take in two parameters. The first parameter will be the true labels, and the second parameter will be the predictions.
{: .parameter-list}

> Returns
{: .returns}

* **result** *(`dict`)* - Dictionary containing evaluation results.

**Note:** For more details on evaluating T5 models with custom metrics, please refer to the [Evaluating Generated Sequences](/docs/t5-specifics/#evaluating-generated-sequences) section.
{: .notice--info}

**Note:** For more details on evaluating models with Simple Transformers, please refer to the [Tips and Tricks](/docs/tips-and-tricks) section.
{: .notice--info}

## Making Predictions With a `T5Model`

The `predict()`  method is used to make predictions with the model.

```python
to_predict = [
    "binary classification: Luke blew up the first Death Star",
    "generate question: In 1971, George Lucas wanted to film an adaptation of the Flash Gordon serial, but could not obtain the rights, so he began developing his own space opera.",
]

predictions = model.predict(to_predict)
```

**Note:** The input **must** be a List even if there is only one sentence.
{: .notice--info}


> *simpletransformers.t5.T5Model.predict*{: .function-name}(to_predict)

Performs predictions on a list of text `to_predict`.
{: .function-text}

> Parameters
{: .parameter-blockquote}

**to_predict** - A python list of text (str) to be sent to the model for prediction.
{: .parameter-list}

> Returns
{: .returns}

* **preds** *(`list`)* - A python list of the generated sequences.
{: .return-list}
