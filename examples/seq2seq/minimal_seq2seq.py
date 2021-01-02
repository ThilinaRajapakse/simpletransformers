import logging

import pandas as pd

from simpletransformers.seq2seq import Seq2SeqModel

logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)


train_data = [
    ["one", "1"],
    ["two", "2"],
]

train_df = pd.DataFrame(train_data, columns=["input_text", "target_text"])

eval_data = [
    ["three", "3"],
    ["four", "4"],
]

eval_df = pd.DataFrame(eval_data, columns=["input_text", "target_text"])

model_args = {
    "reprocess_input_data": True,
    "overwrite_output_dir": True,
    "max_seq_length": 10,
    "train_batch_size": 2,
    "num_train_epochs": 100,
    "save_eval_checkpoints": False,
    "save_model_every_epoch": False,
    # "silent": True,
    "evaluate_generated_text": True,
    "evaluate_during_training": True,
    "evaluate_during_training_verbose": True,
    "use_multiprocessing": False,
    "save_best_model": False,
    "max_length": 15,
}

model = Seq2SeqModel("bert", "bert-base-cased", "bert-base-cased", args=model_args)


def count_matches(labels, preds):
    print(labels)
    print(preds)
    return sum([1 if label == pred else 0 for label, pred in zip(labels, preds)])


model.train_model(train_df, eval_data=eval_df, matches=count_matches)

print(model.eval_model(eval_df, matches=count_matches))

print(model.predict(["four", "five"]))
