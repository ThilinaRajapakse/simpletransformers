---
title: Classification Data Formats
permalink: /docs/classification-data-formats/
excerpt: "Classification data formats."
last_modified_at: 2020/07/02 01:40:54
toc: true
---

The required input data formats for each classification sub-task is described in this section.


## Train Data Format

*Used with [`train_model()`](/docs/classification-models/#training-a-classification-model)*

### Binary classification

The train data should be contained in a Pandas Dataframe with at least two columns. One column should contain the text and the other should contain the labels. The text column should be of datatype `str`, while the labels column should be of datatype `int` (0 or 1).

If the dataframe has a header row, the text column should have the heading `text` and the labels column should have the heading `labels`.

| text                            | labels |
| ------------------------------- | ------ |
| Aragorn was the heir of Isildur | 1      |
| Frodo was the heir of Isildur   | 0      |

```python
train_data = [
    ["Aragorn was the heir of Isildur", 1],
    ["Frodo was the heir of Isildur", 0],
]
train_df = pd.DataFrame(train_data)
train_df.columns = ["text", "labels"]
```

### Multi-class classification

Identical to binary classification, except the labels start from `0` and go up to `n`, where `n` is the number of labels.

| text                            | labels |
| ------------------------------- | ------ |
| Aragorn was the heir of Isildur | 1      |
| Frodo was the heir of Isildur   | 0      |
| Pippin is stronger than Merry   | 2      |

```python
train_data = [
    ["Aragorn was the heir of Isildur", 1],
    ["Frodo was the heir of Isildur", 0],
    ["Pippin is stronger than Merry", 2],
]
train_df = pd.DataFrame(train_data)
train_df.columns = ["text", "labels"]
```

### Regression

Identical to binary classification, except the labels are continuous values and the labels column is of type `float`.

| text                            | labels |
| ------------------------------- | ------ |
| Aragorn was the heir of Isildur | 1.0    |
| Frodo was the heir of Isildur   | 0.0    |
| Pippin is stronger than Merry   | 0.3    |

```python
train_data = [
    ["Aragorn was the heir of Isildur", 1.0],
    ["Frodo was the heir of Isildur", 0.0],
    ["Pippin is stronger than Merry", 0.3],
]
train_df = pd.DataFrame(train_data)
train_df.columns = ["text", "labels"]
```

### Multi-label classification

Identical to binary classification, except the labels are lists of *ints* and the labels column is of type `list`.

| text                            | labels |
| ------------------------------- | ------ |
| Aragorn was the heir of Isildur | [0, 1] |
| Frodo was the heir of Isildur   | [0, 0] |
| Pippin is stronger than Merry   | [1, 1] |

```python
train_data = [
    ["Aragorn was the heir of Isildur", [0, 1]],
    ["Frodo was the heir of Isildur", [0, 0]],
    ["Pippin is stronger than Merry", [1, 1]],
]
train_df = pd.DataFrame(train_data)
train_df.columns = ["text", "labels"]
```

**Note:** Each distinct label can only take the values `0` or `1`. I.e., *multi-class-multi-label* classification is not currently supported.
{: .notice--info}

**Warning:** Pandas can cause issues when saving and loading lists stored in a column. Check whether your list has been converted to a String!
{: .notice--warning}



## Evaluation Data Format

*Used with [`eval_model()`](/docs/classification-models/#evaluating-a-classification-model)*

The evaluation data format is identical to the train data format.

### Binary classification

| text                            | labels |
| ------------------------------- | ------ |
| Aragorn was the heir of Isildur | 1      |
| Frodo was the heir of Isildur   | 0      |

### Multi-class classification

| text                            | labels |
| ------------------------------- | ------ |
| Aragorn was the heir of Isildur | 1      |
| Frodo was the heir of Isildur   | 0      |
| Pippin is stronger than Merry   | 2      |


### Regression

| text                            | labels |
| ------------------------------- | ------ |
| Aragorn was the heir of Isildur | 1.0    |
| Frodo was the heir of Isildur   | 0.0    |
| Pippin is stronger than Merry   | 0.3    |

### Multi-label classification

| text                            | labels |
| ------------------------------- | ------ |
| Aragorn was the heir of Isildur | [0, 1] |
| Frodo was the heir of Isildur   | [0, 0] |
| Pippin is stronger than Merry   | [1, 1] |


## Prediction Data Format
*Used with `predict()`*

The prediction data must be a list of strings.

```python
to_predict = [
    "Gandalf was a Wizard",
    "Sam was a Wizard",
]
```

Identical for binary classification, multi-class classification, regression, and multi-label classification.


## Sentence-Pair Data Format

When performing sentence-pair tasks (e.g. sentence similarity), both the training and evaluation dataframes must contain a header row. The dataframes must also have at least 3 columns, `text_a`, `text_b`, and `labels`.

| text_a                         | text_b                                    | labels |
| ------------------------------ | ----------------------------------------- | ------ |
| Gimli fought with a battle axe | Gimli's preferred weapon was a battle axe | 1      |
| Legolas was an expert archer   | Legolas was taller than Gimli             | 0      |

```python
train_data = [
    [
        "Gimli fought with a battle axe",
        "Gimli's preferred weapon was a battle axe",
        1,
    ],
    [
        "Legolas was an expert archer",
        "Legolas was taller than Giml",
        0,
    ],
]
train_df = pd.DataFrame(train_data)
train_df.columns = ["text_a", "text_b", "labels"]
```

The input to the `predict()` method in sentence-pair tasks must be a list of lists, where the inner list contains two sentences (`text_a` and `text_b` for a single sample) while the outer list is the list of all samples.

```python
to_predict = [
    ["Gimli fought with a battle axe", "Gimli's preferred weapon was a battle axe"],
    ["Legolas was an expert archer", "Legolas was taller than Gimli"],
]
```

Everything else is identical to the single sentence data formats.


## Lazy Loading Data Format

The data must be input as a path to a file to use Lazy Loading.

**Warning:** Not currently implemented for Multi-label tasks.
{: .notice--warning}

The format is similar to the structure of corresponding dataframes in the normal input formats. (One sample per row, with `\t` as the separator)

### Binary Classification

```text
Aragorn was the heir of Isildur    1
Frodo was the heir of Isildur    0
```

### Multi-class classification

```text
Aragorn was the heir of Isildur    1
Frodo was the heir of Isildur    0
Pippin is stronger than Merry    2
```

### Regression

```text
Aragorn was the heir of Isildur    1.0
Frodo was the heir of Isildur    0.0
Pippin is stronger than Merry    0.3
```

### Sentence-Pair Classification
```text
Gimli fought with a battle axe    Gimli's preferred weapon was a battle axe    1
Legolas was an expert archer    Legolas was taller than Gimli    0
```
