---
title: Retrieval Specifics
permalink: /docs/retrieval-specifics/
excerpt: "Specific notes for Retrieval tasks."
last_modified_at: 2021/10/02 15:56:17
toc: true
---

Retrieval models (`RetrievalModel`) are models used to retrieve relevant documents from a corpus given a query.

Currently, only [DPR](https://arxiv.org/abs/2004.04906) models are supported.


## Usage Steps

Using a retrieval model in Simple Transformers follows the [standard pattern](/docs/usage/#task-specific-models).

1. Initialize a `RetrievalModel`
2. Train the model with `train_model()`
3. Evaluate the model with `eval_model()`
4. Make predictions on (unlabelled) data with `predict()`


**Note:** You must have Faiss (GPU or CPU) installed to use RAG Models.
Faiss installation instructions can be found [here](https://github.com/facebookresearch/faiss/blob/master/INSTALL.md).
{: .notice--warning}

### Initializing a `RetrievalModel`

The `__init__` arguments for a `RetrievalModel` are a little different from the common format found in the other models. Please refer [here](/docs/seq2seq-model/#seq2seq-model) for more information.

## Evaluating Generated Sequences

You can evaluate the models' generated sequences using custom metric functions (including evaluation during training). However, due to the way Seq2Seq outputs are generated, this may be significantly slower than evaluation with other models.

**Note:** You must set `evaluate_generated_text` to `True` to evaluate generated sequences.
{: .notice--warning}

```python
import logging

import pandas as pd
from simpletransformers.seq2seq import (
    RetrievalModel,
    Seq2SeqArgs,
)


logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

train_data = [
    [
        "Perseus “Percy” Jackson is the main protagonist and the narrator of the Percy Jackson and the Olympians series.",
        "Percy is the protagonist of Percy Jackson and the Olympians",
    ],
    [
        "Annabeth Chase is one of the main protagonists in Percy Jackson and the Olympians.",
        "Annabeth is a protagonist in Percy Jackson and the Olympians.",
    ],
]

train_df = pd.DataFrame(
    train_data, columns=["input_text", "target_text"]
)

eval_data = [
    [
        "Grover Underwood is a satyr and the Lord of the Wild. He is the satyr who found the demigods Thalia Grace, Nico and Bianca di Angelo, Percy Jackson, Annabeth Chase, and Luke Castellan.",
        "Grover is a satyr who found many important demigods.",
    ],
    [
        "Thalia Grace is the daughter of Zeus, sister of Jason Grace. After several years as a pine tree on Half-Blood Hill, she got a new job leading the Hunters of Artemis.",
        "Thalia is the daughter of Zeus and leader of the Hunters of Artemis.",
    ],
]

eval_df = pd.DataFrame(
    eval_data, columns=["input_text", "target_text"]
)

model_args = Seq2SeqArgs()
model_args.num_train_epochs = 10
model_args.no_save = True
model_args.evaluate_generated_text = True
model_args.evaluate_during_training = True
model_args.evaluate_during_training_verbose = True

# Initialize model
model = RetrievalModel(
    encoder_decoder_type="bart",
    encoder_decoder_name="facebook/bart-large",
    args=model_args,
    use_cuda=True,
)


def count_matches(labels, preds):
    print(labels)
    print(preds)
    return sum(
        [
            1 if label == pred else 0
            for label, pred in zip(labels, preds)
        ]
    )


# Train the model
model.train_model(
    train_df, eval_data=eval_df, matches=count_matches
)

# # Evaluate the model
results = model.eval_model(eval_df)

# Use the model for prediction
print(
    model.predict(
        [
            "Tyson is a Cyclops, a son of Poseidon, and Percy Jackson’s half brother. He is the current general of the Cyclopes army."
        ]
    )
)

```
