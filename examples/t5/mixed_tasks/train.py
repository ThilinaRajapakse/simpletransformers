import pandas as pd

from simpletransformers.t5 import T5Model

train_df = pd.read_csv("data/train.tsv", sep="\t").astype(str)
eval_df = pd.read_csv("data/eval.tsv", sep="\t").astype(str)

model_args = {
    "max_seq_length": 196,
    "train_batch_size": 16,
    "eval_batch_size": 64,
    "num_train_epochs": 1,
    "evaluate_during_training": True,
    "evaluate_during_training_steps": 15000,
    "evaluate_during_training_verbose": True,
    "use_multiprocessing": False,
    "fp16": False,
    "save_steps": -1,
    "save_eval_checkpoints": False,
    "save_model_every_epoch": False,
    "reprocess_input_data": True,
    "overwrite_output_dir": True,
    "wandb_project": "T5 mixed tasks - Binary, Multi-Label, Regression",
}

model = T5Model("t5", "t5-base", args=model_args)

model.train_model(train_df, eval_data=eval_df)
