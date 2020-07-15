---
title: Question Answering Data Formats
permalink: /docs/qa-data-formats/
excerpt: "Data Formats for Question Answering tasks."
last_modified_at: 2020/07/16 03:53:42
toc: true
---

For question answering tasks, the input data can be in JSON files or in a Python list of dictionaries in the correct format.
The structure of both formats is identical, i.e. the input may be a string pointing to a JSON file containing a list of dictionaries, or it the input may be a list of dictionaries itself.


## Input Structure

The input data should be a single list of dictionaries (or path to a JSON file containing the same). A dictionary represents a single context and its associated questions.

Each such dictionary contains two attributes, the `"context"` and `"qas"`.
- `context`: The paragraph or text from which the question is asked.
- `qas`: A list of questions and answers (format below).

Questions and answers are represented as dictionaries. Each dictionary in `qas` has the following format.
- `id`: *(string)* A unique ID for the question. Should be unique across the entire dataset.
- `question`: *(string)* A question.
- `is_impossible`: *(bool)* Indicates whether the question can be answered correctly from the context.
- `answers`: *(list)* The list of correct answers to the question.

A single answer is represented by a dictionary with the following attributes.
- `text`: *(string)* The answer to the question. Must be a substring of the context.
- `answer_start`: *(int)* Starting index of the answer in the context.


## Train Data Format

*Used with [`train_model()`](/docs/qa-model/#training-a-questionansweringmodel)*

Train data can be in the form of a path to a JSON file or a list of dictionaries in the structure specified.

**Note:** There cannot be multiple correct answers to a single question during training. Each question must have a single answer (or an empty string as the answer with `is_impossible=True`).
{: .notice--warning}


### List of dictionaries

```python
train_data = [
    {
        "context": "Mistborn is a series of epic fantasy novels written by American author Brandon Sanderson.",
        "qas": [
            {
                "id": "00001",
                "is_impossible": False,
                "question": "Who is the author of the Mistborn series?",
                "answers": [
                    {
                        "text": "Brandon Sanderson",
                        "answer_start": 71,
                    }
                ],
            }
        ],
    },
    {
        "context": "The first series, published between 2006 and 2008, consists of The Final Empire,"
                   "The Well of Ascension, and The Hero of Ages.",
        "qas": [
            {
                "id": "00002",
                "is_impossible": False,
                "question": "When was the series published?",
                "answers": [
                    {
                        "text": "between 2006 and 2008",
                        "answer_start": 28,
                    }
                ],
            },
            {
                "id": "00003",
                "is_impossible": False,
                "question": "What are the three books in the series?",
                "answers": [
                    {
                        "text": "The Final Empire, The Well of Ascension, and The Hero of Ages",
                        "answer_start": 63,
                    }
                ],
            },
            {
                "id": "00004",
                "is_impossible": True,
                "question": "Who is the main character in the series?",
                "answers": [],
            },
        ],
    },
]

```

### JSON file

```python
train_data = "data/train.json"
```


## Evaluation Data Format

*Used with [`eval_model()`](/docs/qa-model/#evaluating-a-questionansweringmodel)*

Evaluation data can be in the form of a path to a JSON file or a list of dictionaries in the structure specified.

```python

eval_data = [
    {
        "context": "The series primarily takes place in a region called the Final Empire "
                   "on a world called Scadrial, where the sun and sky are red, vegetation is brown, "
                   "and the ground is constantly being covered under black volcanic ashfalls.",
        "qas": [
            {
                "id": "00001",
                "is_impossible": False,
                "question": "Where does the series take place?",
                "answers": [
                    {
                        "text": "region called the Final Empire",
                        "answer_start": 38,
                    },
                    {
                        "text": "world called Scadrial",
                        "answer_start": 74,
                    },
                ],
            }
        ],
    },
    {
        "context": "\"Mistings\" have only one of the many Allomantic powers, while \"Mistborns\" have all the powers.",
        "qas": [
            {
                "id": "00002",
                "is_impossible": False,
                "question": "How many powers does a Misting possess?",
                "answers": [
                    {
                        "text": "one",
                        "answer_start": 21,
                    }
                ],
            },
            {
                "id": "00003",
                "is_impossible": True,
                "question": "What are Allomantic powers?",
                "answers": [],
            },
        ],
    },
]

```


## Prediction Data Format

*Used with [`predict()`](/docs/qa-model/#making-predictions-with-a-questionansweringmodel)*

The `predict()` method of a Simple Transformers model is typically used to get a prediction from the model when the true label/answer is not known. Reflecting this, the `predict()` method of the `QuestionAnsweringModel` class expects a list of dictionaries which contains only contexts, questions, and an unique ID for each question.

The prediction data should be in the following format.

```python
to_predict = [
    {
        "context": "Vin is a Mistborn of great power and skill.",
        "qas": [
            {
                "question": "What is Vin's speciality?",
                "id": "0",
            }
        ],
    }
]

```


## Lazy Loading Data Format

The training data (`train_data`) must be input as a path (str) to a [JSONL](http://jsonlines.org/) file to use Lazy Loading.

The structure of the JSON object is identical to the normal Question Answering train data [format](https://simpletransformers.ai/docs/qa-data-formats/#train-data-format).

**Note:** Currently, lazy loading is only supported for training. The full `eval_data` will be loaded to memory.
{: .notice--warning}
