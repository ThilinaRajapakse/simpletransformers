import pandas as pd

from simpletransformers.t5 import T5Model

train_df = pd.read_csv("data/train_df.tsv", sep="\t").astype(str)
eval_df = pd.read_csv("data/eval_df.tsv", sep="\t").astype(str)

model_args = {
    "reprocess_input_data": True,
    "overwrite_output_dir": True,
    "max_seq_length": 128,
    "train_batch_size": 8,
    "num_train_epochs": 1,
    "save_eval_checkpoints": True,
    "save_steps": -1,
    "use_multiprocessing": False,
    # "silent": True,
    "evaluate_during_training": True,
    "evaluate_during_training_steps": 15000,
    "evaluate_during_training_verbose": True,
    "fp16": False,
    "wandb_project": "Question Generation with T5",
}

model = T5Model("t5", "t5-large", args=model_args)

model.train_model(train_df, eval_data=eval_df)
