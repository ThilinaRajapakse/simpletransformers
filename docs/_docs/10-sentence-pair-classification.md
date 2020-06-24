---
title: Sentence Pair Classification
permalink: /docs/sentence-pair-classification/
excerpt: "Sentence pair classification."
last_modified_at: 2020-05-02 17:58:44
---

In sentence-pair classification, each example in a dataset has *two* sentences along with the appropriate target variable. E.g. Sentence similarity, entailment, etc.

Sentence pairs are supported in all classification subtasks.

**Note:** Input dataframes must contain the three columns, `text_a`, `text_b`, and `labels`. See [Sentence-Pair Data Format](/docs/classification-data-formats/#sentence-pair-data-format).
{: .notice--info}

**Note:** The `predict()` function expects a list of lists. A single sample input should also be a list of lists like [[text_a, text_b]]. See [Sentence-Pair Data Format](/docs/classification-data-formats/#sentence-pair-data-format).
{: .notice--info}

**Tip:** Refer to [ClassificationModel](/docs/classification-models/#classificationmodel) for details on configuring a classification model.
{: .notice--success}

## Minimal Start

```python
from simpletransformers.classification import (
    ClassificationModel, ClassificationArgs
)
import pandas as pd
import logging


logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

# Preparing train data
train_data = [
    [
        "Aragorn was the heir of Isildur",
        "Gimli fought with a battle axe",
        1,
    ],
    [
        "Frodo was the heir of Isildur",
        "Legolas was an expert archer",
        0,
    ],
]
train_df = pd.DataFrame(train_data)
train_df.columns = ["text_a", "text_b", "labels"]

# Preparing eval data
eval_data = [
    [
        "Theoden was the king of Rohan",
        "Gimli's preferred weapon was a battle axe",
        1,
    ],
    [
        "Merry was the king of Rohan",
        "Legolas was taller than Gimli",
        0,
    ],
]
eval_df = pd.DataFrame(eval_data)
eval_df.columns = ["text_a", "text_b", "labels"]

# Optional model configuration
model_args = ClassificationArgs(num_train_epochs=1)

# Create a ClassificationModel
model = ClassificationModel("roberta", "roberta-base")

# Train the model
model.train_model(train_df)

# Evaluate the model
result, model_outputs, wrong_predictions = model.eval_model(
    eval_df
)

# Make predictions with the model
predictions, raw_outputs = model.predict(
    [
        [
            "Legolas was an expert archer",
            "Legolas was taller than Gimli",
        ]
    ]
)

```

## Guides

- [Semantic Textual Similarity Benchmark - Sentence Pair](https://medium.com/@chaturangarajapakshe/solving-sentence-pair-tasks-using-simple-transformers-2496fe79d616?source=friends_link&sk=fbf7439e9c31f7aefa1613d423a0fd40)