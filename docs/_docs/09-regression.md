---
title: Regression
permalink: /docs/regression/
excerpt: "Regression tasks."
last_modified_at: 2020-05-02 17:58:39
---

The goal of regression  in natural language processing is to predict a single, continuous target value for each example in the dataset. A transformer-based regression model typically consists of a transformer model with a fully-connected layer on top of it. The fully-connected layer will have a single output neuron which predicts the target.

**Note:** You must configure the model's args dict and set `regression` to `True`.
{: .notice--info}

**Note:** You must set `num_labels` to `1`.
{: .notice--info}

**Tip:** Refer to [ClassificationModel](/docs/classification-models/#classificationmodel) for details on configuring a classification model.
{: .notice--success}

## Minimal Start

```python
from simpletransformers.classification import ClassificationModel, ClassificationArgs
import pandas as pd
import logging


logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

# Preparing train data
train_data = [
    ["Aragorn was the heir of Isildur", 1.0],
    ["Frodo was the heir of Isildur", 0.0],
    ["Pippin is stronger than Merry", 0.3],
]
train_df = pd.DataFrame(train_data)
train_df.columns = ["text", "labels"]

# Preparing eval data
eval_data = [
    ["Theoden was the king of Rohan", 1.0],
    ["Merry was the king of Rohan", 0.0],
    ["Aragorn is stronger than Boromir", 0.5],
]
eval_df = pd.DataFrame(eval_data)
eval_df.columns = ["text", "labels"]

# Enabling regression
# Setting optional model configuration
model_args = ClassificationArgs()
model_args.num_train_epochs = 1
model_args.regression = True

# Create a ClassificationModel
model = ClassificationModel(
    "roberta",
    "roberta-base",
    num_labels=1,
    args=model_args
)

# Train the model
model.train_model(train_df)

# Evaluate the model
result, model_outputs, wrong_predictions = model.eval_model(eval_df)

# Make predictions with the model
predictions, raw_outputs = model.predict(["Sam was a Wizard"])

```
