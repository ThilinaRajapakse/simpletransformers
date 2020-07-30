---
title: Question Answering  Minimal Start
permalink: /docs/qa-minimal-start/
excerpt: "Minimal start for Question Answering tasks."
last_modified_at: 2020/07/30 20:41:22
---

```python
import logging

from simpletransformers.question_answering import QuestionAnsweringModel, QuestionAnsweringArgs


logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

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

# Configure the model
model_args = QuestionAnsweringArgs()
model_args.train_batch_size = 16
model_args.evaluate_during_training = True

model = QuestionAnsweringModel(
    "roberta", "roberta-base", args=model_args
)

# Train the model
model.train_model(train_data, eval_data=eval_data)

# Evaluate the model
result, texts = model.eval_model(eval_data)

# Make predictions with the model
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

answers, probabilities = model.predict(to_predict)

print(answers)

```


## Guide

- [SQuAD 2.0 - Question Answering](https://towardsdatascience.com/question-answering-with-bert-xlnet-xlm-and-distilbert-using-simple-transformers-4d8785ee762a?source=friends_link&sk=e8e6f9a39f20b5aaf08bbcf8b0a0e1c2)
