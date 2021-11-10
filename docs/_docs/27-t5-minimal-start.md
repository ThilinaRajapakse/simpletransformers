---
title: T5 Minimal Start
permalink: /docs/t5-minimal-start/
excerpt: "Minimal start for T5."
last_modified_at: 2021/11/10 16:36:19
---

```python
import logging

import pandas as pd
from simpletransformers.t5 import T5Model, T5Args

logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)


train_data = [
    ["binary classification", "Anakin was Luke's father" , 1],
    ["binary classification", "Luke was a Sith Lord" , 0],
    ["generate question", "Star Wars is an American epic space-opera media franchise created by George Lucas, which began with the eponymous 1977 film and quickly became a worldwide pop-culture phenomenon", "Who created the Star Wars franchise?"],
    ["generate question", "Anakin was Luke's father" , "Who was Luke's father?"],
]
train_df = pd.DataFrame(train_data)
train_df.columns = ["prefix", "input_text", "target_text"]

eval_data = [
    ["binary classification", "Leia was Luke's sister" , 1],
    ["binary classification", "Han was a Sith Lord" , 0],
    ["generate question", "In 2020, the Star Wars franchise's total value was estimated at US$70 billion, and it is currently the fifth-highest-grossing media franchise of all time.", "What is the total value of the Star Wars franchise?"],
    ["generate question", "Leia was Luke's sister" , "Who was Luke's sister?"],
]
eval_df = pd.DataFrame(eval_data)
eval_df.columns = ["prefix", "input_text", "target_text"]

# Configure the model
model_args = T5Args()
model_args.num_train_epochs = 200
model_args.no_save = True
model_args.evaluate_generated_text = True
model_args.evaluate_during_training = True
model_args.evaluate_during_training_verbose = True

model = T5Model("t5", "t5-base", args=model_args)

# Train the model
model.train_model(train_df, eval_data=eval_df)

# Evaluate the model
result = model.eval_model(eval_df)

# Make predictions with the model
to_predict = [
    "binary classification: Luke blew up the first Death Star",
    "generate question: In 1971, George Lucas wanted to film an adaptation of the Flash Gordon serial, but could not obtain the rights, so he began developing his own space opera.",
]

preds = model.predict(to_predict)

```


## Guides

- [Asking the Right Questions: Training a T5 Transformer Model on a New task](https://towardsdatascience.com/asking-the-right-questions-training-a-t5-transformer-model-on-a-new-task-691ebba2d72c?source=friends_link&sk=9f88c539546eca32b702cc0243abd0dd)
- [The Guide to Multi-Tasking with the T5 Transformer](https://towardsdatascience.com/the-guide-to-multi-tasking-with-the-t5-transformer-90c70a08837b?source=friends_link&sk=ffe37deefa8dd4158f3f76e3dd46cf11)
