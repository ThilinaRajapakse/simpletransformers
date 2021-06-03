import json
import logging
import os

from tqdm.auto import tqdm

from simpletransformers.question_answering import QuestionAnsweringModel

logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

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
                "answers": [{"text": "Bald Eagle Protection Act", "answer_start": 167}],
            },
            {
                "id": "00004",
                "is_impossible": True,
                "question": "How did Alexandar Hamilton die?",
                "answers": [],
            },
        ],
    },
]  # noqa: ignore flake8"

for i in range(20):
    train_data.extend(train_data)

# Save as a JSON file
os.makedirs("data", exist_ok=True)
with open("data/train.json", "w") as f:
    json.dump(train_data, f)

# Save as a JSONL file
with open("data/train.jsonl", "w") as outfile:
    for entry in tqdm(train_data):
        json.dump(entry, outfile)
        outfile.write("\n")

train_args = {
    "reprocess_input_data": True,
    "overwrite_output_dir": True,
    "evaluate_during_training": True,
    "evaluate_during_training_steps": 10000,
    "train_batch_size": 8,
    "num_train_epochs": 1,
    # 'wandb_project': 'test-new-project',
    # "use_early_stopping": True,
    "n_best_size": 3,
    "fp16": False,
    "no_save": True,
    "manual_seed": 4,
    "max_seq_length": 512,
    "no_save": True,
    "n_best_size": 10,
    "lazy_loading": True,
    # "use_multiprocessing": False,
}

# Create the QuestionAnsweringModel
model = QuestionAnsweringModel(
    "bert", "bert-base-cased", args=train_args, use_cuda=True, cuda_device=0
)

# Train the model with JSON file
model.train_model("data/train.jsonl", eval_data="data/train.json")

# Making predictions using the model.
to_predict = [
    {
        "context": "Other legislation followed, including the Migratory Bird Conservation Act of 1929, a 1937 treaty prohibiting the hunting of right and gray whales,\
            and the Bald Eagle Protection Act of 1940. These later laws had a low cost to society—the species were relatively rare—and little opposition was raised",
        "qas": [{"question": "What was the name of the 1937 treaty?", "id": "0"}],
    }
]

print(model.predict(to_predict, n_best_size=2))

# flake8: noqa
