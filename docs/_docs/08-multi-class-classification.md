---
title: Multi-Class Classification
permalink: /docs/multi-class-classification/
excerpt: "Multi-Class text classification."
last_modified_at: 2020-05-02 17:58:31
---

The goal of multi-class classification is to classify a text sequence into one of `n` classes. A transformer-based multi-class text classification model typically consists of a transformer model with a classification layer on top of it. The classification layer will have `n` output neurons, corresponding to each class.

The minimal start given below uses a `n` value of `3`. You can change `n` by changing the `num_labels` parameter.

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
    ["Aragorn was the heir of Isildur", 1],
    ["Frodo was the heir of Isildur", 0],
    ["Pippin is stronger than Merry", 2],
]
train_df = pd.DataFrame(train_data)
train_df.columns = ["text", "labels"]

# Preparing eval data
eval_data = [
    ["Aragorn was the heir of Elendil", 1],
    ["Sam was the heir of Isildur", 0],
    ["Merrry is stronger than Pippin", 2],
]
eval_df = pd.DataFrame(eval_data)
eval_df.columns = ["text", "labels"]

# Optional model configuration
model_args = ClassificationArgs(num_train_epochs=1)

# Create a ClassificationModel
model = ClassificationModel(
    'bert',
    'bert-base-cased',
    num_labels=3,
    args=model_args
) 

# Train the model
model.train_model(train_df)

# Evaluate the model
result, model_outputs, wrong_predictions = model.eval_model(eval_df)

# Make predictions with the model
predictions, raw_outputs = model.predict(["Sam was a Wizard"])

```

## Guides

- [AG News Dataset - Multiclass Classification](https://medium.com/swlh/simple-transformers-multi-class-text-classification-with-bert-roberta-xlnet-xlm-and-8b585000ce3a?source=friends_link&sk=90e1c97255b65cedf4910a99041d9dfc)
- [AG News Dataset - BERT (base and distilled), RoBERTa (base and distilled), and XLNet compared](https://towardsdatascience.com/to-distil-or-not-to-distil-bert-roberta-and-xlnet-c777ad92f8?source=friends_link&sk=6a3c7940b18066ded94aeee95e354ed1)
- [Comparing ELECTRA, BERT, RoBERTa, and XLNET](https://medium.com/@chaturangarajapakshe/battle-of-the-transformers-electra-bert-roberta-or-xlnet-40607e97aba3?sk=fe857841d15d5202d94a58ba166c240b)