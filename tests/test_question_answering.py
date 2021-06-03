import json
import logging
import os

import pytest

from simpletransformers.question_answering import QuestionAnsweringModel


@pytest.mark.parametrize(
    "model_type, model_name",
    [
        ("bert", "bert-base-uncased"),
        ("longformer", "allenai/longformer-base-4096"),
        # ("reformer", "google/reformer-crime-and-punishment"),
        # ("xlnet", "xlnet-base-cased"),
        # ("xlm", "xlm-mlm-17-1280"),
        # ("roberta", "roberta-base"),
        # ("distilbert", "distilbert-base-uncased"),
        # ("albert", "albert-base-v1"),
        # ("camembert", "camembert-base"),
        # ("xlmroberta", "xlm-roberta-base"),
        # ("flaubert", "flaubert-base-cased"),
    ],
)
def test_question_answering(model_type, model_name):
    # Create dummy data to use for training.
    train_data = [
        {
            "context": "This is the first context",
            "qas": [
                {
                    "id": "00001",
                    "is_impossible": False,
                    "question": "Which context is this?",
                    "answers": [{"text": "the first", "answer_start": 8}],
                }
            ],
        },
        {
            "context": "Other legislation followed, including the Migratory Bird Conservation Act of 1929, a 1937 treaty prohibiting the hunting of right and gray whales,\
                and the Bald Eagle Protection Act of 1940. These later laws had a low cost to society—the species were relatively rare—and little opposition was raised",
            "qas": [
                {
                    "id": "00002",
                    "is_impossible": False,
                    "question": "What was the cost to society?",
                    "answers": [{"text": "low cost", "answer_start": 225}],
                },
                {
                    "id": "00003",
                    "is_impossible": False,
                    "question": "What was the name of the 1937 treaty?",
                    "answers": [
                        {"text": "Bald Eagle Protection Act", "answer_start": 167}
                    ],
                },
                {
                    "id": "00004",
                    "is_impossible": True,
                    "question": "How did Alexandar Hamilton die?",
                    "answers": [],
                },
            ],
        },
    ]  # noqa

    for i in range(4):
        train_data.extend(train_data)

    # Save as a JSON file
    os.makedirs("data", exist_ok=True)
    with open("data/train.json", "w") as f:
        json.dump(train_data, f)

    logging.basicConfig(level=logging.WARNING)
    transformers_logger = logging.getLogger("transformers")
    transformers_logger.setLevel(logging.ERROR)

    # Create the QuestionAnsweringModel
    model = QuestionAnsweringModel(
        model_type,
        model_name,
        args={
            "no_save": True,
            "reprocess_input_data": True,
            "overwrite_output_dir": True,
        },
        use_cuda=False,
    )

    # Train the model
    model.train_model("data/train.json")

    # Evaluate the model. (Being lazy and evaluating on the train data itself)
    result, text = model.eval_model("data/train.json")

    # Making predictions using the model.
    to_predict = [
        {
            "context": "This is the context used for demonstrating predictions.",
            "qas": [{"question": "What is this context?", "id": "0"}],
        }
    ]

    model.predict(to_predict)
