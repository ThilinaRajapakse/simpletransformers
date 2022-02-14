---
title: Question Answering Model
permalink: /docs/qa-model/
excerpt: "QuestionAnsweringModel for Question Answering tasks."
last_modified_at: 2020/12/29 17:01:27
toc: true
---


## `QuestionAnsweringModel`

The `QuestionAnsweringModel` class is used for Question Answering.

To create a `QuestionAnsweringModel`, you must specify a `model_type` and a `model_name`.

- `model_type` should be one of the model types from the [supported models](/docs/qa-specifics/) (e.g. bert, electra, xlnet)
- `model_name` specifies the exact architecture and trained weights to use. This may be a Hugging Face Transformers compatible pre-trained model, a community model, or the path to a directory containing model files.

    **Note:** For a list of standard pre-trained models, see [here](https://huggingface.co/transformers/pretrained_models.html).
    {: .notice--info}

    **Note:** For a list of community models, see [here](https://huggingface.co/models).
    {: .notice--info}

    You may use any of these models provided the `model_type` is supported.

```python
from simpletransformers.question_answering import QuestionAnsweringModel


model = QuestionAnsweringModel(
    "roberta", "roberta-base"
)
```

**Note:** For more information on working with Simple Transformers models, please refer to the [General Usage section](/docs/usage/#creating-a-task-specific-model).
{: .notice--info}


### Configuring a `QuestionAnsweringModel`

`QuestionAnsweringModel` has several task-specific configuration options.


| Argument                  | Type  | Default | Description                                                                                          |
| ------------------------- | ----- | ------- | ---------------------------------------------------------------------------------------------------- |
| doc_stride                | int   | `384`   | When splitting up a long document into chunks, how much stride to take between chunks.               |
| max_query_length          | int   | `64`    | Maximum token length for questions. Any questions longer than this will be truncated to this length. |
| n_best_size               | int   | `20`    | The number of predictions given per question.                                                        |
| max_answer_length         | int   | `100`   | The maximum token length of an answer that can be generated.                                         |
| null_score_diff_threshold | float | `0.0`   | If `(null_score - best_non_null)` is greater than the threshold predict null.                            |
| special_tokens_list     | list      | []      | The list of special tokens to be added to the model tokenizer                                                                                         |

```python
from simpletransformers.question_answering import QuestionAnsweringModel, QuestionAnsweringArgs


model_args = QuestionAnsweringModel(n_best_size=2)

model = QuestionAnsweringModel(
    "roberta",
    "roberta-base",
    args=model_args,
)
```

**Note:** For configuration options common to all Simple Transformers models, please refer to the [Configuring a Simple Transformers Model section](/docs/usage/#configuring-a-simple-transformers-model).
{: .notice--info}


## `Class QuestionAnsweringModel`

> *simpletransformers.question_answering.QuestionAnsweringModel*{: .function-name}(self, model_type, model_name, args=None, use_cuda=True, cuda_device=-1, **kwargs,)

Initializes a QuestionAnsweringModel model.
{: .function-text}

> Parameters
{: .parameter-blockquote}

* **model_type** *(`str`)* - The type of model to use ([model types](/docs/qa-specifics/#supported-model-types))

* **model_name** *(`str`)* - The exact architecture and trained weights to use. This may be a Hugging Face Transformers compatible pre-trained model, a community model, or the path to a directory containing model files.

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


## Training a `QuestionAnsweringModel`

The `train_model()`  method is used to train the model.

```python
model.train_model(train_data)
```

> *simpletransformers.question_answering.QuestionAnsweringModel*{: .function-name}(self, train_data, output_dir=None, show_running_loss=True, args=None, eval_data=None, verbose=True, **kwargs)

Trains the model using 'train_data'
{: .function-text}

> Parameters
{: .parameter-blockquote}

* **train_data** - Path to JSON file containing training data OR list of Python dicts in the correct format. The model will be trained on this data. Refer to the [Question Answering Data Formats](/docs/qa-data-formats) section for the correct formats.

* **output_dir** *(`str`, optional)* - The directory where model files will be saved. If not given, `self.args['output_dir']` will be used.

* **show_running_loss** *(`bool`, optional)* - If True, the running loss (training loss at current step) will be logged to the console.

* **args** *(`dict`, optional)* - A dict of configuration options for the `QuestionAnsweringModel`. Any changes made will persist for the model.

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


## Evaluating a `QuestionAnsweringModel`

The `eval_model()`  method is used to evaluate the model.

The following metrics will be calculated by default:

* `correct` - Number of predicted answers matching the true answer exactly.
* `similar` - Number of predicted answers that are a substring of the true answer or vice versa.
* `incorrect` - Number of predicted answers that does not meet the criteria for `correct` or `similar`.
* `eval_loss` - Cross Entropy Loss for eval_data


```python
result, model_outputs, wrong_preds = model.eval_model(eval_data)
```

> *simpletransformers.question_answering.QuestionAnsweringModel.eval_model*{: .function-name}(self, eval_data,
> output_dir=None, verbose=True, silent=False, **kwargs)

Evaluates the model using 'eval_data'
{: .function-text}

> Parameters
{: .parameter-blockquote}

* **eval_data** - Path to JSON file containing evaluation data OR list of Python dicts in the correct format. The model will be evaluated on this data. Refer to the [Question Answering Data Formats](/docs/qa-data-formats) section for the correct formats.

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

* **result** *(`dict`)* - Dictionary containing evaluation results.

* **texts** *(`list`)* - A dictionary containing the 3 dictionaries `correct_text`, `similar_text`, and `incorrect_text`.
{: .return-list}

**Note:** For more details on evaluating models with Simple Transformers, please refer to the [Tips and Tricks](/docs/tips-and-tricks) section.
{: .notice--info}


## Making Predictions With a `QuestionAnsweringModel`

The `predict()`  method is used to make predictions with the model.

```python
context_text = "Mistborn is a series of epic fantasy novels written by American author Brandon Sanderson."

predictions, raw_outputs = model.predict(
    [
        {
            "context": context_text,
            "qas": [
                {
                    "question": "Who was the author of Mistborn?",
                    "id": "0",
                }
            ],
        }
    ]
)

```

**Note:** The input **must** be a List even if there is only one sentence.
{: .notice--info}


> *simpletransformers.question_answering.QuestionAnsweringModel.predict*{: .function-name}(to_predict, n_best_size=None)

Performs predictions on a list of text `to_predict`.
{: .function-text}

> Parameters
{: .parameter-blockquote}

* **to_predict** - A python list of python dicts in the correct format to be sent to the model for prediction. Refer to the [Question Answering Data Formats](/docs/qa-data-formats) section for the correct formats.

* **n_best_size** *(`int`, optional)* - Number of predictions to return. args['n_best_size'] will be used if not specified.
{: .parameter-list}

> Returns
{: .returns}

* **answer_list** *(`list`)* - A Python list of dicts containing each question id mapped to its answer (or a list of answers if `n_best_size > 1`).
* **probability_list** *(`list`)* - A Python list of dicts containing each question id mapped to the probability score for the answer (or a list of probability scores if `n_best_size > 1`).
{: .return-list}

**Tip:** You can also make predictions using the Simple Viewer web app. Please refer to the [Simple Viewer](/docs/tips-and-tricks/#simple-viewer-visualizing-model-predictions-with-streamlit) section.
{: .notice--success}
