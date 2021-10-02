---
title: Multi-Label Classification
permalink: /docs/multi-label-classification/
excerpt: "Multi-label text classification."
last_modified_at: 2021/10/02 12:31:00
---

In multi-label text classification, the target for a single example from the dataset is a list of `n` distinct binary labels. A transformer-based multi-label text classification model typically consists of a transformer model with a classification layer on top of it. The classification layer will have `n` output neurons, corresponding to each label. Each output neuron (and by extension, each label) are considered to be independent of each other.

{% capture notice-text %}
To illustrate the difference between multi-class classification and multi-label classification, consider the following:

Multi-class: Out of the four races Men, Elves, Dwarves, and Hobbits;

* Aragorn is a Man.
* Frodo is a Hobbit.
* Gimli is a Dwarf.

*Each sample can only belong to one of each class.*

Multi-label: A character could be a formidable warrior, a Ringbearer, and of short stature.

* Aragorn is a formidable warrior, but neither a Ringbearer nor short.
* Frodo is not a formidable warrior, but is a Ringbearer, and is short.
* Gimli is a formidable warrior, not a Ringbearer, and is short.

*Each sample has a binary value for each label.*

{% endcapture %}

<div class="notice--success">
  <h4>Multi-label vs Multi-class:</h4>
  {{ notice-text | markdownify }}
</div>


**Tip:** Refer to [MultiLabelClassificationModel](/docs/classification-models/#multilabelclassificationmodel) for details on configuring a multi-label classification model.
{: .notice--success}


## Minimal Start

```python
from simpletransformers.classification import (
    MultiLabelClassificationModel, MultiLabelClassificationArgs
)
import pandas as pd
import logging


logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

# Preparing train data
train_data = [
    ["Aragorn", [1, 0, 0]],
    ["Frodo", [0, 1, 1]],
    ["Gimli", [1, 0, 1]],
]
train_df = pd.DataFrame(train_data)
train_df.columns = ["text", "labels"]

# Preparing eval data
eval_data = [
    ["Legolas", [1, 0, 0]],
    ["Merry", [0, 0, 1]],
    ["Eomer", [1, 0, 0]],
]
eval_df = pd.DataFrame(eval_data)
eval_df.columns = ["text", "labels"]

# Optional model configuration
model_args = MultiLabelClassificationArgs(num_train_epochs=1)

# Create a MultiLabelClassificationModel
model = MultiLabelClassificationModel(
    "roberta",
    "roberta-base",
    num_labels=3,
    args=model_args,
)

# Train the model
model.train_model(train_df)

# Evaluate the model
result, model_outputs, wrong_predictions = model.eval_model(
    eval_df
)

# Make predictions with the model
predictions, raw_outputs = model.predict(["Sam"])

```

## Guides

- [Toxic Comments Dataset - Multilabel Classification](https://towardsdatascience.com/multi-label-classification-using-bert-roberta-xlnet-xlm-and-distilbert-with-simple-transformers-b3e0cda12ce5?source=friends_link&sk=354e688fe238bfb43e9a575216816219)
