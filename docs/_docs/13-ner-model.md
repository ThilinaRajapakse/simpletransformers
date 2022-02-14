---
title: NER Model
permalink: /docs/ner-model/
excerpt: "NERModel for named entity recognition."
last_modified_at: 2021/10/02 12:54:24
toc: true
---


## `NERModel`

The `NERModel` class is used for Named Entity Recognition (token classification).

To create a `NERModel`, you must specify a `model_type` and a `model_name`.

- `model_type` should be one of the model types from the [supported models](/docs/ner-specifics/) (e.g. bert, electra, xlnet)
- `model_name` specifies the exact architecture and trained weights to use. This may be a Hugging Face Transformers compatible pre-trained model, a community model, or the path to a directory containing model files.

    **Note:** For a list of standard pre-trained models, see [here](https://huggingface.co/transformers/pretrained_models.html).
    {: .notice--info}

    **Note:** For a list of community models, see [here](https://huggingface.co/models).
    {: .notice--info}

    You may use any of these models provided the `model_type` is supported.

```python
from simpletransformers.ner import NERModel


model = NERModel(
    "roberta", "roberta-base"
)
```

**Note:** For more information on working with Simple Transformers models, please refer to the [General Usage section](/docs/usage/#creating-a-task-specific-model).
{: .notice--info}


### Configuring a `NERModel`

`NERModel` has the following task-specific configuration options.

| Argument              | Type | Default                                                                            | Description                                                                  |
| --------------------- | ---- | ---------------------------------------------------------------------------------- | ---------------------------------------------------------------------------- |
| classification_report | bool | `True`                                                                             | If True, a sklearn classification report will be written to the `output_dir` |
| labels_list           | list | `["O", "B-MISC", "I-MISC",  "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"]` | A list of all token labels.                                                  |
| special_tokens_list     | list      | []      | The list of special tokens to be added to the model tokenizer                                                                                         |

```python
from simpletransformers.ner import NERModel, NERArgs


model_args = NERArgs()
model_args.labels_list = ["PERSON", "LOCATION", "CAREER"]

model = NERModel(
    "roberta",
    "roberta-base",
    args=model_args,
)
```

**Note:** For configuration options common to all Simple Transformers models, please refer to the [Configuring a Simple Transformers Model section](/docs/usage/#configuring-a-simple-transformers-model).
{: .notice--info}


## `Class NERModel`

> *simpletransformers.ner.NERModel*{: .function-name}(self, model_type, model_name, labels=None, weight=None, args=None, use_cuda=True, cuda_device=-1, onnx_execution_provider=None, **kwargs,)

Initializes a NERModel model.
{: .function-text}

> Parameters
{: .parameter-blockquote}

* **model_type** *(`str`)* - The type of model to use ([model types](/docs/ner-specifics/#supported-model-types))

* **model_name** *(`str`)* - The exact architecture and trained weights to use. This may be a Hugging Face Transformers compatible pre-trained model, a community model, or the path to a directory containing model files.

* **labels** *(`int`, optional)* - A list of all Named Entity labels.  If not given, ["O", "B-MISC", "I-MISC",  "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"] will be used. (See [here](/docs/ner-specifics/#custom-labels))

* **weight** *(`list`, optional)* - A `torch.Tensor`, `numpy.ndarray` or list.  The weight to be applied to each class when computing the loss of the model.

* **args** *(`dict`, optional)* - [Default args](/docs/usage/#configuring-a-simple-transformers-model) will be used if this parameter is not provided. If provided, it should be a dict containing the args that should be changed in the default args.

* **use_cuda** *(`bool`, optional)* - Use GPU if available. Setting to False will force model to use CPU only. (See [here](/docs/usage/#to-cuda-or-not-to-cuda))

* **cuda_device** *(`int`, optional)* - Specific GPU that should be used. Will use the first available GPU by default. (See [here](/docs/usage/#selecting-a-cuda-device))

* **onnx_execution_provider** *(`str`, optional)* - The execution provider to use for ONNX export.

* **kwargs** *(optional)* - For providing proxies, force_download, resume_download, cache_dir and other options specific to the 'from_pretrained' implementation where this will be supplied. (See [here](/docs/usage/#options-for-downloading-pre-trained-models))
{: .parameter-list}

> Returns
{: .returns}

* `None`
{: .return-list}


## Training a `NERModel`

The `train_model()`  method is used to train the model.

```python
model.train_model(train_data)
```

> *simpletransformers.ner.NERModel.train_model*{: .function-name}(self, train_data, output_dir=None, show_running_loss=True, args=None, eval_data=None, verbose=True, **kwargs)

Trains the model using 'train_data'
{: .function-text}

> Parameters
{: .parameter-blockquote}

* **train_data** - train_data should be the path to a .txt file containing the training data OR a pandas DataFrame with 3 columns. If a text file is given the data should be in the CoNLL format. i.e. One word per line, with sentences separated by an empty line. The first word of the line should be a word, and the last should be a Name Entity Tag. If a DataFrame is given, each sentence should be split into words, with each word assigned a tag, and with all words from the same sentence given the same sentence_id.

* **output_dir** *(`str`, optional)* - The directory where model files will be saved. If not given, `self.args['output_dir']` will be used.

* **show_running_loss** *(`bool`, optional)* - If True, the running loss (training loss at current step) will be logged to the console.

* **args** *(`dict`, optional)* - A dict of configuration options for the `NERModel`. Any changes made will persist for the model.

* **eval_data** *(optional)* - Evaluation data (same format as train_data) against which evaluation will be performed when evaluate_during_training is enabled. Is required if evaluate_during_training is enabled.

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


## Evaluating a `NERModel`

The `eval_model()`  method is used to evaluate the model.

The following metrics will be calculated by default:

* `precision` - Precision
* `recall` - Recall
* `f1_score` - F1 score
* `eval_loss` - Cross Entropy Loss for eval_data


```python
result, model_outputs, wrong_preds = model.eval_model(eval_data)
```

> *simpletransformers.ner.NERModel.eval_model*{: .function-name}(self, eval_data,
> output_dir=None, verbose=True, silent=False, **kwargs)

Evaluates the model using 'eval_data'
{: .function-text}

> Parameters
{: .parameter-blockquote}

* **eval_data** - eval_data should be the path to a .txt file containing the training data OR a pandas DataFrame with 3 columns. If a text file is given the data should be in the CoNLL format. i.e. One word per line, with sentences seperated by an empty line. The first word of the line should be a word, and the last should be a Name Entity Tag. If a DataFrame is given, each sentence should be split into words, with each word assigned a tag, and with all words from the same sentence given the same sentence_id.

* **output_dir** *(`str`, optional)* - The directory where model files will be saved. If not given, `self.args['output_dir']` will be used.

* **verbose** *(`bool`, optional)* - If verbose, results will be printed to the console on completion of evaluation.

* **silent** *(`bool`, optional)* - If silent, tqdm progress bars will be hidden.

* **kwargs** *(optional)* - Additional metrics that should be calculated. Pass in the metrics as keyword arguments *(name of metric: function to calculate metric)*. Refer to the [additional metrics](/docs/tips-and-tricks/#additional-evaluation-metrics) section.
E.g. `f1=sklearn.metrics.f1_score`.
A metric function should take in two parameters. The first parameter will be the true labels, and the second parameter will be the predictions.
{: .parameter-list}

> Returns
{: .returns}

* **result** *(`dict`)* - Dictionary containing evaluation results.

* **model_outputs** *(`list`)* - List of model outputs for each row in eval_data

* **preds_list** *(`list`)* - List of predicted tags
{: .return-list}

**Note:** For more details on evaluating models with Simple Transformers, please refer to the [Tips and Tricks](/docs/tips-and-tricks) section.
{: .notice--info}


## Making Predictions With a `NERModel`

The `predict()`  method is used to make predictions with the model.

```python
predictions, raw_outputs = model.predict(["Sample sentence 1", "Sample sentence 2"])
```

**Note:** The input **must** be a List even if there is only one sentence.
{: .notice--info}


> *simpletransformers.ner.NERModel.predict*{: .function-name}(to_predict, split_on_space=True)

Performs predictions on a list of text `to_predict`.
{: .function-text}

> Parameters
{: .parameter-blockquote}

* **to_predict** - A python list of text (str) to be sent to the model for prediction.

* **split_on_space** *(`bool`, optional)* - If True, each sequence will be split by spaces for assigning labels. If set to `False`, `to_predict` must be a a list of lists, with the inner list being a list of strings consisting of the split sequences. The outer list is the list of sequences to predict on.
{: .parameter-list}

> Returns
{: .returns}

* **preds** *(`list`)* - A Python list of lists of dicts containing each word mapped to its NER tag.

* **model_outputs** *(`list`)* - A Python list of lists with dicts containing each word mapped to its list with raw model output.
{: .return-list}

**Tip:** You can also make predictions using the Simple Viewer web app. Please refer to the [Simple Viewer](/docs/tips-and-tricks/#simple-viewer-visualizing-model-predictions-with-streamlit) section.
{: .notice--success}
