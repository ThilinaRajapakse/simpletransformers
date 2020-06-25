---
title: NER Minimal Start
permalink: /docs/ner-minimal-start/
excerpt: "Minimal start for named entity recognition."
last_modified_at: 2020/06/26 02:39:31
---

```python
import logging

import pandas as pd
from simpletransformers.ner import NERModel, NERArgs


logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

train_data = [
    [0, "Harry", "B-PER"],
    [0, "Potter", "I-PER"],
    [0, "was", "O"],
    [0, "a", "O"],
    [0, "student", "B-MISC"],
    [0, "at", "O"],
    [0, "Hogwarts", "B-LOC"],
    [1, "Albus", "B-PER"],
    [1, "Dumbledore", "I-PER"],
    [1, "founded", "O"],
    [1, "the", "O"],
    [1, "Order", "B-ORG"],
    [1, "of", "I-ORG"],
    [1, "the", "I-ORG"],
    [1, "Phoenix", "I-ORG"],
]
train_data = pd.DataFrame(
    train_data, columns=["sentence_id", "words", "labels"]
)

eval_data = [
    [0, "Sirius", "B-PER"],
    [0, "Black", "I-PER"],
    [0, "was", "O"],
    [0, "a", "O"],
    [0, "prisoner", "B-MISC"],
    [0, "at", "O"],
    [0, "Azkaban", "B-LOC"],
    [1, "Lord", "B-PER"],
    [1, "Voldemort", "I-PER"],
    [1, "founded", "O"],
    [1, "the", "O"],
    [1, "Death", "B-ORG"],
    [1, "Eaters", "I-ORG"],
]
eval_data = pd.DataFrame(
    eval_data, columns=["sentence_id", "words", "labels"]
)

# Configure the model
model_args = NERArgs()
model_args.train_batch_size = 16
model_args.evaluate_during_training = True

model = NERModel(
    "roberta", "roberta-base", args=model_args
)

# Train the model
model.train_model(train_data, eval_data=eval_data)

# Evaluate the model
result, model_outputs, preds_list = model.eval_model(eval_data)

# Make predictions with the model
predictions, raw_outputs = model.predict(["Hermione was the best in her class"])

```

## Guides

- [CoNLL Dataset Example](https://towardsdatascience.com/simple-transformers-named-entity-recognition-with-transformer-models-c04b9242a2a0?source=friends_link&sk=e8b98c994173cd5219f01e075727b096)