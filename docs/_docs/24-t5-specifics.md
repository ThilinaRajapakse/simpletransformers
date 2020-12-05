---
title: T5 Specifics
permalink: /docs/t5-specifics/
excerpt: "Specific notes for T5 tasks."
last_modified_at: 2020/12/04 22:18:32
toc: true
---

The T5 Transformer is an Encoder-Decoder architecture where both the input and targets are text sequences. This gives it the flexibility to perform any Natural Language Processing task without having to modify the model architecture in any way. It also means that the same T5 model can be trained to perform multiple tasks *simultaneously*.

Please refer to the *[Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer](https://arxiv.org/abs/1910.10683)* paper for more details.

**Tip:** This [Medium article](https://towardsdatascience.com/asking-the-right-questions-training-a-t5-transformer-model-on-a-new-task-691ebba2d72c?source=friends_link&sk=9f88c539546eca32b702cc0243abd0dd) explains how to train a T5 Model to perform a new task.
{: .notice--success}

**Tip:** This [Medium article](https://towardsdatascience.com/the-guide-to-multi-tasking-with-the-t5-transformer-90c70a08837b?source=friends_link&sk=ffe37deefa8dd4158f3f76e3dd46cf11) explains how to train a single T5 Model to perform multiple tasks.
{: .notice--success}

## Specifying a Task

The T5 model is instructed to perform a particular task by adding a *prefix* to the start of an input sequence. The *prefix* for a specific task may be any arbitrary text as long as the same *prefix* is prepended whenever the model is supposed to execute the given task.

Example *prefixes*:

1. `binary classification`
2. `predict sentiment`
3. `answer question`

By using multiple, unique *prefixes* we can train a T5 model to do multiple tasks. During inference, the model will look at the *prefix* and generate the appropriate output.

**Hint:** See the T5 Data Formats page for more details on how the inputs and outputs are structured.
{: .notice--success}

## Usage Steps

Using a T5 Model in Simple Transformers follows the [standard pattern](/docs/usage/#task-specific-models).

1. Initialize a `T5Model`
2. Train the model with `train_model()`
3. Evaluate the model with `eval_model()`
4. Make predictions on (unlabelled) data with `predict()`


## Supported Model Types

| Model | Model code for `NERModel` |
| ----- | ------------------------- |
| T5    | t5                        |
| MT5   | mt5                       |

**Tip:** The model code is used to specify the `model_type` in a Simple Transformers model.
{: .notice--success}


## Evaluating Generated Sequences

You can evaluate the models' generated sequences using custom metric functions (including evaluation during training). However, due to the way T5 outputs are generated, this may be significantly slower than evaluation with other models.

**Note:** You must set `evaluate_generated_text` to `True` to evaluate generated sequences.
{: .notice--warning}

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

train_data = [
    ["binary classification", "Leia was Luke's sister" , 1],
    ["binary classification", "Han was a Sith Lord" , 0],
    ["generate question", "In 2020, the Star Wars franchise's total value was estimated at US$70 billion, and it is currently the fifth-highest-grossing media franchise of all time.", "What is the total value of the Star Wars franchise?"],
    ["generate question", "Leia was Luke's sister" , "Who was Luke's sister?"],
]
train_df = pd.DataFrame(train_data)
train_df.columns = ["prefix", "input_text", "target_text"]

model_args = T5Args()
model_args.num_train_epochs = 200
model_args.no_save = True
model_args.evaluate_generated_text = True
model_args.evaluate_during_training = True
model_args.evaluate_during_training_verbose = True

model = T5Model("t5", "t5-base", args=model_args)


def count_matches(labels, preds):
    print(labels)
    print(preds)
    return sum([1 if label == pred else 0 for label, pred in zip(labels, preds)])


model.train_model(train_df, eval_data=eval_df, matches=count_matches)

print(model.eval_model(eval_df, matches=count_matches))

```
