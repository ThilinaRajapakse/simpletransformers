import sys

import pandas as pd

from simpletransformers.classification import ClassificationModel

prefix = "data/"

train_df = pd.read_csv(prefix + "train.csv", header=None)
train_df.head()

eval_df = pd.read_csv(prefix + "test.csv", header=None)
eval_df.head()

train_df[0] = (train_df[0] == 2).astype(int)
eval_df[0] = (eval_df[0] == 2).astype(int)

train_df = pd.DataFrame(
    {"text": train_df[1].replace(r"\n", " ", regex=True), "labels": train_df[0]}
)

print(train_df.head())

eval_df = pd.DataFrame(
    {"text": eval_df[1].replace(r"\n", " ", regex=True), "labels": eval_df[0]}
)

print(eval_df.head())


model_type = sys.argv[1]

if model_type == "bert":
    model_name = "bert-base-cased"

elif model_type == "roberta":
    model_name = "roberta-base"

elif model_type == "distilbert":
    model_name = "distilbert-base-cased"

elif model_type == "distilroberta":
    model_type = "roberta"
    model_name = "distilroberta-base"

elif model_type == "electra-base":
    model_type = "electra"
    model_name = "google/electra-base-discriminator"

elif model_type == "electra-small":
    model_type = "electra"
    model_name = "google/electra-small-discriminator"

elif model_type == "xlnet":
    model_name = "xlnet-base-cased"

train_args = {
    "reprocess_input_data": True,
    "overwrite_output_dir": True,
    "use_cached_eval_features": True,
    "output_dir": f"outputs/{model_type}",
    "best_model_dir": f"outputs/{model_type}/best_model",
    "evaluate_during_training": True,
    "max_seq_length": 128,
    "num_train_epochs": 3,
    "evaluate_during_training_steps": 1000,
    "wandb_project": "Classification Model Comparison",
    "wandb_kwargs": {"name": model_name},
    "save_model_every_epoch": False,
    "save_eval_checkpoints": False,
    # "use_early_stopping": True,
    # "early_stopping_metric": "mcc",
    # "n_gpu": 2,
    # "manual_seed": 4,
    # "use_multiprocessing": False,
    "train_batch_size": 128,
    "eval_batch_size": 64,
    # "config": {
    #     "output_hidden_states": True
    # }
}

if model_type == "xlnet":
    train_args["train_batch_size"] = 64
    train_args["gradient_accumulation_steps"] = 2


# Create a ClassificationModel
model = ClassificationModel(model_type, model_name, args=train_args)

# Train the model
model.train_model(train_df, eval_df=eval_df)

# # # Evaluate the model
# result, model_outputs, wrong_predictions = model.eval_model(eval_df)
