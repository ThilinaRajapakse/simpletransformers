---
title: "Classification Models"
permalink: /docs/classification-models/
excerpt: "Model class used for text classification"
last_modified_at: 2020/12/29 17:01:00
toc: true
---

There are two task-specific Simple Transformers classification models, `ClassificationModel` and `MultiLabelClassificationModel`. The two are mostly identical except for the specific use-case and a few other minor differences detailed below.


## `ClassificationModel`

The `ClassificationModel` class is used for all text classification tasks except for multi label classification.

To create a `ClassificationModel`, you must specify a `model_type` and a `model_name`.

- `model_type` should be one of the model types from the [supported models](/docs/classification-specifics/#supported-model-types) (e.g. bert, electra, xlnet)
- `model_name` specifies the exact architecture and trained weights to use. This may be a Hugging Face Transformers compatible pre-trained model, a community model, or the path to a directory containing model files.

    **Note:** For a list of standard pre-trained models, see [here](https://huggingface.co/transformers/pretrained_models.html).
    {: .notice--info}

    **Note:** For a list of community models, see [here](https://huggingface.co/models).
    {: .notice--info}

    You may use any of these models provided the `model_type` is supported.

```python
from simpletransformers.classification import ClassificationModel


model = ClassificationModel(
    "roberta", "roberta-base"
)
```

**Note:** For more information on working with Simple Transformers models, please refer to the [General Usage section](/docs/usage/#creating-a-task-specific-model).
{: .notice--info}


### `Class ClassificationModel`

> *simpletransformers.classification.ClassificationModel*{: .function-name}(self, model_type, model_name, num_labels=None, weight=None, args=None, use_cuda=True, cuda_device=-1, **kwargs,)

Initializes a ClassificationModel model.
{: .function-text}

> Parameters
{: .parameter-blockquote}

* **model_type** *(`str`)* - The type of model to use ([model types](/docs/classification-specifics/#supported-model-types))

* **model_name** *(`str`)* - The exact architecture and trained weights to use. This may be a Hugging Face Transformers compatible pre-trained model, a community model, or the path to a directory containing model files.

* **num_labels** *(`int`, optional)* - The number of labels or classes in the dataset. (See [here](/docs/classification-models/#specifying-the-number-of-classeslabels))

* **weight** *(`list`, optional)* - A list of length num_labels containing the weights to assign to each label for loss calculation. (See [here](/docs/classification-models/#setting-class-weights))

* **args** *(`dict`, optional)* - [Default args](/docs/usage/#configuring-a-simple-transformers-model) will be used if this parameter is not provided. If provided, it should be a dict containing the args that should be changed in the default args.

* **use_cuda** *(`bool`, optional)* - Use GPU if available. Setting to False will force model to use CPU only. (See [here](/docs/usage/#to-cuda-or-not-to-cuda))

* **cuda_device** *(`int`, optional)* - Specific GPU that should be used. Will use the first available GPU by default. (See [here](/docs/usage/#selecting-a-cuda-device))

* **kwargs** *(optional)* - For providing proxies, force_download, resume_download, cache_dir and other options specific to the 'from_pretrained' implementation where this will be supplied. (See [here](/docs/usage/#options-for-downloading-pre-trained-models))
{: .parameter-list}

> Returns
{: .returns}

* `None`
{: .return-list}


### Specifying the number of classes/labels

By default, a `ClassificationModel` will behave as a binary classifier.
You can specify the number of classes/labels to use it as a multi-class classifier or as a regression model.

#### Binary classification
```python
model = ClassificationModel(
    "roberta", "roberta-base"
)
```

#### Multi-class classification
```python
model = ClassificationModel(
    "roberta", "roberta-base", num_labels=4
)
```

#### Regression
```python
model = ClassificationModel(
    "roberta",
    "roberta-base",
    num_labels=1,
    args={
        "regression": True
    }
)
```

**Note:** When performing regression, you must configure the model's args dict and set `regression` to `True` in addition to specifying `num_labels=1`.
{: .notice--warning}


### Setting class weights

A commonly used tactic to deal with imbalanced datasets is to assign weights to each label.
This can be done by passing in a list of weights. The list must contain a weight value for each label.

```python
model = ClassificationModel(
    "roberta",
    "roberta-base",
    num_labels=4,
    weight=[1, 0.5, 1, 2]
)
```

### Configuring a Classification model

`ClassificationModel` has the following task-specific configuration options.


| Argument                | Type      | Default | Description                                                                                                                                           |
| ----------------------- | --------- | ------- | ----------------------------------------------------------------------------------------------------------------------------------------------------- |
| lazy_delimiter          | str       | `\t`    | The delimiter used to separate column in the file containing the lazy loading dataset                                                                 |
| lazy_loading_start_line | int       | 1       | The line number where the dataset starts (`1` means header row is skipped)                                                                            |
| lazy_labels_column      | int       | `0`     | The column (based on the delimiter) containing the labels for lazy loading single sentence datasets                                                   |
| lazy_text_a_column      | int       | `None`  | The column (based on the delimiter) containing the first sentence (text_a) for lazy loading sentence-pair datasets                                    |
| lazy_text_b_column      | int       | `None`  | The column (based on the delimiter) containing the second sentence (text_a) for lazy loading sentence-pair datasets                                   |
| lazy_text_column        | int       | `0`     | The column (based on the delimiter) containing text for lazy loading single sentence datasets                                                         |
| regression              | int       | `False` | Set True when doing regression. `num_labels` parameter in the model must also be set to `1`.                                                          |
| sliding_window          | bool      | `False` | Whether to use sliding window technique to prevent truncating longer sequences                                                                        |
| special_tokens_list     | list      | []      | The list of special tokens to be added to the model tokenizer                                                                                         |
| stride                  | float/int | `0.8`   | The distance to move the window when generating sub-sequences using a sliding window. Can be a fraction of the `max_seq_length` OR a number of tokens |
| tie_value               | int       | `1`     | The tie_value will be used as the prediction label for any samples where the sliding window predictions are tied                                      |


```python
from simpletransformers.classification import ClassificationModel, ClassificationArgs


model_args = ClassificationArgs(sliding_window=True)

model = ClassificationModel(
    "roberta",
    "roberta-base",
    args=model_args,
)
```

**Note:** For configuration options common to all Simple Transformers models, please refer to the [Configuring a Simple Transformers Model section](/docs/usage/#configuring-a-simple-transformers-model).
{: .notice--info}


## `MultiLabelClassificationModel`

The `MultiLabelClassificationModel` is used for multi-label classification tasks.

To create a `MultiLabelClassificationModel`, you must specify a `model_type` and a `model_name`.

- `model_type` should be one of the model types from the [supported models](/docs/classification-specifics/#supported-model-types) (e.g. bert, electra, xlnet)
- `model_name` specifies the exact architecture and trained weights to use. This may be a Hugging Face Transformers compatible pre-trained model, a community model, or the path to a directory containing model files.

    **Note:** For a list of standard pre-trained models, see [here](https://huggingface.co/transformers/pretrained_models.html).
    {: .notice--info}

    **Note:** For a list of community models, see [here](https://huggingface.co/models).
    {: .notice--info}

    You may use any of these models provided the `model_type` is supported.

```python
from simpletransformers.classification import MultiLabelClassificationModel


model = MultiLabelClassificationModel(
    "roberta", "roberta-base"
)
```

**Note:** For more information on working with Simple Transformers models, please refer to the [General Usage section](/docs/usage/#creating-a-task-specific-model).
{: .notice--info}


### `Class MultiLabelClassificationModel`

> *simpletransformers.classification.MultiLabelClassificationModel*{: .function-name}(self, model_type, model_name, num_labels=None, pos_weight=None, args=None, use_cuda=True, cuda_device=-1, **kwargs,)

Initializes a MultiLabelClassification model.
{: .function-text}

> Parameters
{: .parameter-blockquote}

* **model_type** *(`str`)* - The type of model to use ([model types](/docs/classification-specifics/#supported-model-types))

* **model_name** *(`str`)* - The exact architecture and trained weights to use. This may be a Hugging Face Transformers compatible pre-trained model, a community model, or the path to a directory containing model files.

* **num_labels** *(`int`, optional)* - The number of labels or classes in the dataset. (See [here](/docs/classification-models/#specifying-the-number-of-labels))

* **pos_weight** *(`list`, optional)* - A list of length num_labels containing the weights to assign to each label for loss calculation. (See [here](/docs/classification-models/#setting-class-weights-1))

* **args** *(`dict`, optional)* - [Default args](/docs/usage/#configuring-a-simple-transformers-model) will be used if this parameter is not provided. If provided, it should be a dict containing the args that should be changed in the default args.

* **use_cuda** *(`bool`, optional)* - Use GPU if available. Setting to False will force model to use CPU only. (See [here](/docs/usage/#to-cuda-or-not-to-cuda))

* **cuda_device** *(`int`, optional)* - Specific GPU that should be used. Will use the first available GPU by default. (See [here](/docs/usage/#selecting-a-cuda-device))

* **kwargs** *(optional)* - For providing proxies, force_download, resume_download, cache_dir and other options specific to the 'from_pretrained' implementation where this will be supplied. (See [here](/docs/usage/#options-for-downloading-pre-trained-models))
{: .parameter-list}

> Returns
{: .returns}

* `None`
{: .return-list}


### Specifying the number of labels

The default number of labels in a `MultiLabelClassificationModel` is `2`. This can be changed by passing in the number of values to `num_labels`.

```python
model = MultiLabelClassificationModel(
    "roberta", "roberta-base", num_labels=4
)
```

### Setting class weights

Setting class weights in the `MultiLabelClassificationModel` is done through the `pos_weight` parameter.

```python
model = MultiLabelClassificationModel(
    "roberta",
    "roberta-base",
    num_labels=4,
    pos_weight=[1, 0.5, 1, 2]
)
```
### Configuring a Multi-Label Classification Model

`MultiLabelClassificationModel` has the following task-specific configuration options.

| Argument  | Type  | Default | Description                                                                                                                                                                                                                                                 |
| --------- | ----- | ------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| threshold | float | `0.5`   | The threshold is the value at which a given label flips from 0 to 1 when predicting. The threshold may be a single value or a list of value with the same length as the number of labels. This enables the use of separate threshold values for each label. |

```python
model_args = {
    "threshold": 0.8
}

model = MultiLabelClassificationModel(
    "roberta",
    "roberta-base",
    args=model_args,
)
```

**Note:** For configuration options common to all Simple Transformers models, please refer to the [Configuring a Simple Transformers Model section](/docs/usage/#configuring-a-simple-transformers-model).
{: .notice--info}


## Training a Classification Model

The `train_model()`  method is used to train the model. The `train_model()` method is identical for `ClassificationModel` and `MultiLabelClassificationModel`, except for the `multi_label` argument being `True` by default for the latter.

```python
model.train_model(train_df)
```

> *simpletransformers.classification.ClassificationModel.train_model*{: .function-name}(self, train_df, multi_label=False,
> output_dir=None, show_running_loss=True, args=None, eval_df=None, verbose=True, **kwargs)

Trains the model using 'train_df'
{: .function-text}

> Parameters
{: .parameter-blockquote}

* **train_df** - Pandas DataFrame containing the train data. Refer to [Data Format](/docs/classification-data-formats/).

* **output_dir** *(`str`, optional)* - The directory where model files will be saved. If not given, `self.args['output_dir']` will be used.

* **show_running_loss** *(`bool`, optional)* - If True, the running loss (training loss at current step) will be logged to the console.

* **args** *(`dict`, optional)* - A dict of configuration options for the `ClassificationModel`. Any changes made will persist for the model.

* **eval_df** *(`dataframe`, optional)* - A DataFrame against which evaluation will be performed when evaluate_during_training is enabled. Is required if evaluate_during_training is enabled.

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


## Evaluating a Classification Model

The `eval_model()`  method is used to evaluate the model. The `eval_model()` method is identical for `ClassificationModel` and `MultiLabelClassificationModel`, except for the `multi_label` argument being `True` by default for the latter.

The following metrics will be calculated by default:

* Binary classification
  * `mcc` - [Matthews correlation coefficient](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.matthews_corrcoef.html)
  * `tp` - True positives
  * `tn` - True negatives
  * `fp` - False positives
  * `fn` - False negatives
  * `eval_loss` - Cross Entropy Loss for eval_df
* Multi-class classification
  * `mcc` - [Matthews correlation coefficient](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.matthews_corrcoef.html)
  * `eval_loss` - Cross Entropy Loss for eval_df
* Regression
  * `eval_loss` - Cross Entropy Loss for eval_df
* Multi-label classification
  * `LRAP` - [Label ranking average precision](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.label_ranking_average_precision_score.html?highlight=lrap)
  * `eval_loss` - Binary Cross Entropy Loss for eval_df


```python
result, model_outputs, wrong_predictions = model.eval_model(eval_df)
```

> *simpletransformers.classification.ClassificationModel.eval_model*{: .function-name}(self, eval_df, multi_label=False,
> output_dir=None, verbose=True, silent=False, **kwargs)

Evaluates the model using 'eval_df'
{: .function-text}

> Parameters
{: .parameter-blockquote}

* **eval_df** - Pandas DataFrame containing the evaluation data. Refer to [Data Format](/docs/classification-data-formats/).

* **output_dir** *(`str`, optional)* - The directory where model files will be saved. If not given, `self.args['output_dir']` will be used.

* **verbose** *(`bool`, optional)* - If verbose, results will be printed to the console on completion of evaluation.

* **silent** *(`bool`, optional)* - If silent, tqdm progress bars will be hidden.

* **kwargs** *(optional)* - Additional metrics that should be calculated. Pass in the metrics as keyword arguments *(name of metric: function to calculate metric)*. Refer to the [additional metrics](/docs/tips-and-tricks/#additional-evaluation-metrics).
E.g. `f1=sklearn.metrics.f1_score` section.
A metric function should take in two parameters. The first parameter will be the true labels, and the second parameter will be the predictions.
{: .parameter-list}

> Returns
{: .returns}

* **result** *(`dict`)* - Dictionary containing evaluation results.

* **model_outputs** *(`list`)* - List of model outputs for each row in eval_df

* **wrong_preds** *(`list`)* - List of InputExample objects corresponding to each incorrect prediction by the model
{: .return-list}

**Note:** For more details on evaluating models with Simple Transformers, please refer to the [Tips and Tricks](/docs/tips-and-tricks) section.
{: .notice--info}


## Making Predictions With a Classification Model

The `predict()`  method is used to make predictions with the model. The `predict()` method is identical for `ClassificationModel` and `MultiLabelClassificationModel`, except for the `multi_label` argument being `True` by default for the latter.

```python
predictions, raw_outputs = model.predict(["Sample sentence 1", "Sample sentence 2"])
# For LayoutLM
predictions, raw_outputs = model.predict(
    [
        ["Sample page text 1", [1, 2, 3, 4], [11, 12, 13, 14], [21, 22, 23, 24], [31, 32, 33, 34]],
        ["Sample page text 2", [1, 2, 3, 4], [11, 12, 13, 14], [21, 22, 23, 24], [31, 32, 33, 34]],
    ]
)
```

**Note:** The input **must** be a List (or list of lists) even if there is only one sentence.
{: .notice--info}


> *simpletransformers.classification.ClassificationModel.predict*{: .function-name}(to_predict, multi_label=False)

Performs predictions on a list of text (list of lists for model types `layoutlm` and `layoutlmv2`) `to_predict`.
{: .function-text}

> Parameters
{: .parameter-blockquote}

* **to_predict** - A python list of text (str) to be sent to the model for prediction. For `layoutlm` and `layoutlmv2` model types, this should be a list of lists: 
                        [
                            [text1, [x0], [y0], [x1], [y1]],
                            [text2, [x0], [y0], [x1], [y1]],
                            ...
                            [text3, [x0], [y0], [x1], [y1]]
                        ]
                        
{: .parameter-list}

> Returns
{: .returns}

* **preds** *(`list`)* - A python list of the predictions (0 or 1) for each text.
* **model_outputs** *(`list`)* - A python list of the raw model outputs for each text.
{: .return-list}

**Tip:** You can also make predictions using the Simple Viewer web app. Please refer to the [Simple Viewer](/docs/tips-and-tricks/#simple-viewer-visualizing-model-predictions-with-streamlit) section.
{: .notice--success}
