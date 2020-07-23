---
title: T5 Data Formats
permalink: /docs/t5-data-formats/
excerpt: "Data formats for T5."
last_modified_at: 2020/07/23 23:13:20
toc: true
---

A single input to a T5 model has the following pattern;

```python
"<prefix>: <input_text> </s>"
```

The *label* sequence has the following pattern;

```python
"<target_sequence> </s>"
```

## Train Data Format

*Used with [`train_model()`](/docs/t5-models/#training-a-t5-model)*

The train data should be a Pandas DataFrame containing the 3 columns - `prefix`, `input_text`, `target_text`.

- `prefix`: A string indicating the task to perform. (E.g. `"binary classification"`, `"generate question"`)
- `input_text`: The input text sequence. `prefix` is automatically prepended to form the full input. (`<prefix>: <input_text>`)
- `target_text`: The target sequence


If `preprocess_inputs` is set to `True` in the model `args`, then the `< /s>` tokens (including preceeding space) and the `: ` *(prefix separator including trailing separator)* between `prefix`  and `input_text` are automatically added. Otherwise, the input DataFrames must contain the `< /s>` tokens (including preceeding space) and the `:` *(prefix separator including trailing separator)*.

| prefix                | input_text                                                                                                                                                                        | target_text                          |
| --------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------ |
| binary classification | Anakin was Luke's father                                                                                                                                                          | 1                                    |
| binary classification | Luke was a Sith Lord                                                                                                                                                              | 0                                    |
| generate question     | Star Wars is an American epic space-opera media franchise created by George Lucas, which began with the eponymous 1977 film and quickly became a worldwide pop-culture phenomenon | Who created the Star Wars franchise? |
| generate question     | Anakin was Luke's father                                                                                                                                                          | Who was Luke's father?               |

```python
train_data = [
    ["binary classification", "Anakin was Luke's father" , 1],
    ["binary classification", "Luke was a Sith Lord" , 0],
    ["generate question", "Star Wars is an American epic space-opera media franchise created by George Lucas, which began with the eponymous 1977 film and quickly became a worldwide pop-culture phenomenon", "Who created the Star Wars franchise?"],
    ["generate question", "Anakin was Luke's father" , "Who was Luke's father?"],
]
train_df = pd.DataFrame(train_data)
train_df.columns = ["prefix", "input_text", "target_text"]
```


## Evaluation Data Format

*Used with [`eval_model()`](/docs/classification-models/#evaluating-a-t5-model)*

The evaluation data format is identical to the train data format.

| prefix                | input_text                                                                                                                                                 | target_text                                         |
| --------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------- |
| binary classification | Leia was Luke's sister                                                                                                                                     | 1                                                   |
| binary classification | Han was a Sith Lord                                                                                                                                        | 0                                                   |
| generate question     | In 2020, the Star Wars franchise's total value was estimated at US$70 billion, and it is currently the fifth-highest-grossing media franchise of all time. | What is the total value of the Star Wars franchise? |
| generate question     | Leia was Luke's sister                                                                                                                                     | Who was Luke's sister?                              |

```python
train_data = [
    ["binary classification", "Leia was Luke's sister" , 1],
    ["binary classification", "Han was a Sith Lord" , 0],
    ["generate question", "In 2020, the Star Wars franchise's total value was estimated at US$70 billion, and it is currently the fifth-highest-grossing media franchise of all time.", "What is the total value of the Star Wars franchise?"],
    ["generate question", "Leia was Luke's sister" , "Who was Luke's sister?"],
]
train_df = pd.DataFrame(train_data)
train_df.columns = ["prefix", "input_text", "target_text"]
```


## Prediction Data Format
*Used with [`predict()`](/docs/qa-model/#making-predictions-with-a-t5-model)*

The prediction data should be a list of strings with the *prefix* and the *prefix separator* (`: `) included.

If `preprocess_inputs` is set to `True` in the model `args`, then the ` < /s>` token (including preceeding space) is automatically added to each string in the list. Otherwise, the strings must have the ` < /s>` (including preceeding space) must be included.

**Note:** Unlike with training and evaluation, the *prefix separator* is **NOT** added in prediction even when `preprocess_inputs` is set to `True`.
{: .notice--warning}

```python
to_predict = [
    "binary classification: Luke blew up the first Death Star",
    "generate question: In 1971, George Lucas wanted to film an adaptation of the Flash Gordon serial, but could not obtain the rights, so he began developing his own space opera.",
]
```
