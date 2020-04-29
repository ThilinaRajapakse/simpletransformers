---
title: "Classification Models"
permalink: /docs/models-classification/
excerpt: "Model class used for text classifcation"
last_modified_at: 2018-03-20T15:59:31-04:00
toc: true
---

There are two task-specific Simple Transformers classification models, `ClassificationModel` and `MultiLabelClassificationModel`. The two are mostly identical except for the specific use-case and a few other minor differences detailed below.

## `ClassificationModel`

The `ClassificationModel` class is used for all text classification tasks except for multi label classification.

To create a `ClassificationModel`, you must specify a `model_type` and a `model_name`.

- `model_type` should be one of the model types from the [supported models](/docs/intro-classification/) (e.g. bert, electra, xlnet)
- `model_name` specifies the exact architecture and trained weights to use. This may be a Hugging Face Transformers compatible pre-trained model or it could be the path to a directory containing model files.

    **Note:** For a list of standard pre-trained models, see [here](https://huggingface.co/transformers/pretrained_models.html).
    {: .notice--info}

    **Note:** For a list of community models, see [here](https://huggingface.co/models).
    {: .notice--info}

    You may use any of these models provided the `model_type` is supported.


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
    "roberta", "roberta-base", num_labels=1, args={"regression": True}
)
```

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

### Configuring a model

`ClassificationModel` has the following task-specific configuration options.


| Argument       | Type       | Default | Description                                                                                                                                           |
| -------------- | ---------- | ------- | ----------------------------------------------------------------------------------------------------------------------------------------------------- |
| sliding_window | bool       | `False` | Whether to use sliding window technique to prevent truncating longer sequences                                                                        |
| tie_value      | int        | `1`     | The tie_value will be used as the prediction label for any samples where the sliding window predictions are tied                                      |
| stride         | float/int  | `0.8`   | The distance to move the window when generating sub-sequences using a sliding window. Can be a fraction of the `max_seq_length` OR a number of tokens |
| regression     | regression | `False` | Set True when doing regression. `num_labels` parameter in the model must also be set to `1`.                                                          |



## `MultiLabelClassificationModel`

The `MultiLabelClassificationModel` is used for multi-label classification tasks.

To create a `MultiLabelClassificationModel`, you must specify a `model_type` and a `model_name`.

- `model_type` should be one of the model types from the [supported models](/docs/intro-classification/) (e.g. bert, electra, xlnet)
- `model_name` specifies the exact architecture and trained weights to use. This may be a Hugging Face Transformers compatible pre-trained model or it could be the path to a directory containing model files.

    **Note:** For a list of standard pre-trained models, see [here](https://huggingface.co/transformers/pretrained_models.html).
    {: .notice--info}

    **Note:** For a list of community models, see [here](https://huggingface.co/models).
    {: .notice--info}

    You may use any of these models provided the `model_type` is supported.

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
### Configuring a model

`MultiLabelClassificationModel` has the following task-specific configuration options.

| Argument  | Type  | Default | Description                                                                                                                                                                                                                                                 |
|-----------|-------|---------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| threshold | float | `0.5`     | The threshold is the value at which a given label flips from 0 to 1 when predicting. The threshold may be a single value or a list of value with the same length as the number of labels. This enables the use of separate threshold values for each label. |
