---
title: Binary Classification
permalink: /docs/binary-classification/
excerpt: "Binary text classification."
last_modified_at: 2020-04-27T20:45:49.398Z
toc: true
---

The goal of binary text classification is to classify a text sequence into one of two classes. A transformer-based binary text classification model typically consists of a transformer model with a classification layer on top of it. The classification layer will have two output neurons, corresponding to each class.

## Process

The process of performing Binary Classification in Simple Transformers does not deviate from the [standard pattern](/docs/usage/#task-specific-models).

1. Initialize a `ClassificationModel`
2. Train the model with `train_model()`
3. Evaluate the model with `eval_model()`
4. Make predictions on (unlabelled) data with `predict()`

## Data formats

### Train data format

Used with `train_model()`

The train data should be contained in a Pandas Dataframe with at least two columns. One column should contain the text and the other should contain the labels. The text column should be of datatype `str`, while the labels column should be of datatype `int` (taking the values 0 or 1).

If the dataframe has a header row, the text column should have the heading `text` and the labels column should have the heading `labels`.

| text                           | labels |
| ------------------------------ | ------ |
| Aragorn is the heir of Isildur | 1      |
| Frodo is the heir of Isildur   | 0      |
| ...                            | ...    |

Alternatively, you may also use a dataframe without a header row. In this case, the first column **must** contain the text and the second column **must** contain the labels.

```python
train_data = [
    ["Aragorn was the heir of Isildur", 1],
    ["Frodo was the heir of Isildur", 0],
]
train_df = pd.DataFrame(train_data)
train_df.columns = ["text", "labels"]
```

### Evaluation data format

Used with `eval_model()`

The evaluation data format is identical to the train data format.

```python
eval_data = [
    ["Theoden was the king of Rohan", 1],
    ["Merry was the king of Rohan", 0],
]
eval_df = pd.DataFrame(eval_data)
eval_df.columns = ["text", "labels"]
```

### Prediction data format

Used with `predict()`

The prediction data must be a list of strings.

```python
to_predict = [
    "Gandalf was a Wizard",
    "Sam was a Wizard",
]
```

## Usage

### Initializing a model

```python
from simpletransformers.classification import ClassificationModel


model = ClassificationModel(
    "bert",
    "bert-base-cased",
)
```

**Note:** For more details on model initialization, please refer to this [section](/docs/models-classification/#classificationmodel).
{: .notice--info}

### Training a model

The `ClassificationModel.train_model()`  method is used to train the model.

> simpletransformers.classification.ClassificationModel.train_model()


## Minimal Start

```python
from simpletransformers.classification import ClassificationModel
import pandas as pd
import logging


logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

# Preparing train data
train_data = [
    ["Aragorn was the heir of Isildur", 1],
    ["Frodo was the heir of Isildur", 0],
]
train_df = pd.DataFrame(train_data)
train_df.columns = ["text", "labels"]

# Preparing eval data
eval_data = [
    ["Theoden was the king of Rohan", 1],
    ["Merry was the king of Rohan", 0],
]
eval_df = pd.DataFrame(eval_data)
eval_df.columns = ["text", "labels"]

# Create a ClassificationModel
model = ClassificationModel(
    "roberta", "roberta-base"
)  # You can set class weights by using the optional weight argument

# Train the model
model.train_model(train_df)

# Evaluate the model
result, model_outputs, wrong_predictions = model.eval_model(eval_df)

```
