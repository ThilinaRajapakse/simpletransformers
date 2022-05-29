---
title: Classification Specifics
permalink: /docs/classification-specifics/
excerpt: "Specific notes for text classification tasks."
last_modified_at: 2021/02/02 02:03:09
toc: true
---

This section describes how Text Classification tasks are organized and conducted with Simple Transformers.

## Sub-Tasks Falling Under Text Classification

| Task                                       | Model                           |
| ------------------------------------------ | ------------------------------- |
| Binary and multi-class text classification | `ClassificationModel`           |
| Multi-label text classification            | `MultiLabelClassificationModel` |
| Regression                                 | `ClassificationModel`           |
| Sentence-pair classification               | `ClassificationModel`           |


## Usage Steps

The process of performing text classification in Simple Transformers does not deviate from the [standard pattern](/docs/usage/#task-specific-models).

1. Initialize a `ClassificationModel` or a `MultiLabelClassificationModel`
2. Train the model with `train_model()`
3. Evaluate the model with `eval_model()`
4. Make predictions on (unlabelled) data with `predict()`


## Supported Model Types

New model types are regularly added to the library. Text classification tasks currently supports the model types given below.

| Model        | Model code for `ClassificationModel` |
| ------------ | ------------------------------------ |
| ALBERT       | albert                               |
| BERT         | bert                                 |
| BERTweet     | bertweet                             |
| *BigBird     | bigbird                              |
| CamemBERT    | camembert                            |
| *DeBERTa     | deberta                              |
| DistilBERT   | distilbert                           |
| ELECTRA      | electra                              |
| FlauBERT     | flaubert                             |
| HerBERT      | herbert                              |
| LayoutLM     | layoutlm                             |
| LayoutLMv2   | layoutlmv2                           |
| *Longformer  | longformer                           |
| *MPNet       | mpnet                                |
| MobileBERT   | mobilebert                           |
| RemBERT      | rembert                              |
| RoBERTa      | roberta                              |
| *SqueezeBert | squeezebert                          |
| XLM          | xlm                                  |
| XLM-RoBERTa  | xlmroberta                           |
| XLNet        | xlnet                                |

\* *Not available with Multi-label classification*

**Tip:** The model code is used to specify the `model_type` in a Simple Transformers model.
{: .notice--success}


## Dealing With Long Text

Transformer models typically have a restriction on the maximum length allowed for a sequence. This is defined in terms of the number of tokens, where a token is any of the "words" that appear in the model vocabulary.

**Note:** Each Transformer model has a vocabulary which consists of *tokens* mapped to a numeric ID. The input sequence to a Transformer consists of a tensor of numeric values found in the vocabulary.
{: .notice--info}

The `max_seq_length` is the maximum number of such tokens (technically token IDs) that a sequence can contain. Any tokens that appear after the `max_seq_length` will be truncated when working with Transformer models. Unfortunately, each model type also has an upper bound for the `max_seq_length` itself, with it most commonly being 512.

While there is currently no standard method of circumventing this issue, a plausible strategy is to use the sliding window approach. Here, any sequence exceeding the `max_seq_length` will be split into several windows (sub-sequences), each of length `max_seq_length`.

The *windows* will typically overlap each other to a certain degree to minimize any information loss that may be caused by hard cutoffs. The amount of overlap between the windows is determined by the `stride`. The stride is the distance (in terms of number of tokens) that the window will be, well, *slid* to obtain the next sub-sequence.

The `stride` can be specified in terms of either a fraction of the `max_seq_length`, or as an absolute number of tokens. The default `stride` is set to `0.8 * max_seq_length`, which results in about 20% overlap between the sub-sequences.

```python
model_args = ClassificationArgs(sliding_window=True)

model = ClassificationModel(
    "roberta",
    "roberta-base",
    args=model_args,
)
```

### Training with sliding window

When training a model with `sliding_window` enabled, each sub-sequence will be assigned the label from the original sequence. The model will then be trained on the full set of sub-sequences. Depending on the number of sequences and how much each sequence exceeds the `max_seq_length`, the total number of training samples will be higher than the number of sequences originally in the train data.

### Evaluation and prediction with sliding window

During evaluation and prediction, the model will predict a label for each window or sub-sequence of an example. The final prediction for an example will be the mode of the predictions for all its sub-sequences.

In the case of a tie, the predicted label will be assigned the `tie_value` (default `1`).

**Note:** Sliding window technique is not currently implemented for multi-label classification.
{: .notice--warning}


## Lazy Loading Data

The system memory required to keep a large dataset in memory can be prohibitively large. In such cases, the data can be lazy loaded from disk to minimize memory consumption.

To enable lazy loading, you must set the `lazy_loading` flag to `True` in `ClassificationArgs`.


```python
model_args = ClassificationArgs()
model_args.lazy_loading = True
```

**Note:** This will typically be slower as the feature conversion is done on the fly. However, the tradeoff between speed and memory consumption should be reasonable.
{: .notice--info}

**Tip:** See [Lazy Loading Data Formats](/docs/classification-data-formats/#lazy-loading-data-format) for information on the data formats.
{: .notice--success}

**Tip:** See [Configuring a Classification model](/docs/classification-models/#configuring-a-classification-model) for information on configuring the model to read the lazy loading data file correctly.
{: .notice--success}

**Tip:** You can find minimal example scripts in the `examples/text_classification` directory.
{: .notice--success}


## Custom Labels

By default, `ClassificationModel` expects the labels to be ints from `0` up to `num_labels`.

If your dataset contains labels in another format (e.g. string labels like `positive`, `negative`), you can provide the list of all labels to the model args. Simple Transformers will handle the label mappings internally. Note that this will also automatically set `num_labels` to the length of the labels list.

```python
from simpletransformers.classification import ClassificationModel, ClassificationArgs
import pandas as pd
import logging


logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

# Preparing train data
train_data = [
    ["Aragorn was the heir of Isildur", "true"],
    ["Frodo was the heir of Isildur", "false"],
]
train_df = pd.DataFrame(train_data)
train_df.columns = ["text", "labels"]

# Preparing eval data
eval_data = [
    ["Theoden was the king of Rohan", "true"],
    ["Merry was the king of Rohan", "false"],
]
eval_df = pd.DataFrame(eval_data)
eval_df.columns = ["text", "labels"]

# Optional model configuration
model_args = ClassificationArgs()
model_args.num_train_epochs=1
model_args.labels_list = ["true", "false"]

# Create a ClassificationModel
model = ClassificationModel(
    "roberta", "roberta-base", args=model_args
)

# Train the model
model.train_model(train_df)

# Evaluate the model
result, model_outputs, wrong_predictions = model.eval_model(eval_df)

# Make predictions with the model
predictions, raw_outputs = model.predict(["Sam was a Wizard"])

```

**Note:** Custom labels are not currently supported with multi-label classification.
{: .notice--warning}
