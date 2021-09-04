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
import pandas as pd

from simpletransformers.classification import ClassificationModel

# Train and Evaluation data needs to be in a Pandas Dataframe containing at least two columns. If the Dataframe has a header, it should contain a 'text' and a 'labels' column. If no header is present, the Dataframe should contain at least two columns, with the first column is the text with type str, and the second column in the label with type int.
train_data = [
    ["Example sentence belonging to class 1", 1],
    ["Example sentence belonging to class 0", 0],
    ["Example eval senntence belonging to class 2", 2],
]
train_df = pd.DataFrame(train_data)

eval_data = [
    ["Example eval sentence belonging to class 1", 1],
    ["Example eval sentence belonging to class 0", 0],
    ["Example eval senntence belonging to class 2", 2],
]
eval_df = pd.DataFrame(eval_data)

# Create a ClassificationModel
model = ClassificationModel(
    "bert",
    "bert-base-cased",
    num_labels=3,
    args={"reprocess_input_data": True, "overwrite_output_dir": True},
)

# Train the model
model.train_model(train_df)

# Evaluate the model
result, model_outputs, wrong_predictions = model.eval_model(eval_df)

predictions, raw_outputs = model.predict(["Some arbitary sentence"])
```

## Guides

- [AG News Dataset - Multiclass Classification](https://medium.com/swlh/simple-transformers-multi-class-text-classification-with-bert-roberta-xlnet-xlm-and-8b585000ce3a?source=friends_link&sk=90e1c97255b65cedf4910a99041d9dfc)
- [AG News Dataset - BERT (base and distilled), RoBERTa (base and distilled), and XLNet compared](https://towardsdatascience.com/to-distil-or-not-to-distil-bert-roberta-and-xlnet-c777ad92f8?source=friends_link&sk=6a3c7940b18066ded94aeee95e354ed1)
- [Comparing ELECTRA, BERT, RoBERTa, and XLNET](https://medium.com/@chaturangarajapakshe/battle-of-the-transformers-electra-bert-roberta-or-xlnet-40607e97aba3?sk=fe857841d15d5202d94a58ba166c240b)
