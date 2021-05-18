import os
from datetime import datetime
import logging

import pandas as pd
from sklearn.model_selection import train_test_split
from simpletransformers.seq2seq import Seq2SeqModel, Seq2SeqArgs

from utils import load_data, clean_unnecessary_spaces


logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.ERROR)

# Google Data
train_df = pd.read_csv("data/train.tsv", sep="\t").astype(str)
eval_df = pd.read_csv("data/dev.tsv", sep="\t").astype(str)

train_df = train_df.loc[train_df["label"] == "1"]
eval_df = eval_df.loc[eval_df["label"] == "1"]

train_df = train_df.rename(
    columns={"sentence1": "input_text", "sentence2": "target_text"}
)
eval_df = eval_df.rename(
    columns={"sentence1": "input_text", "sentence2": "target_text"}
)

train_df = train_df[["input_text", "target_text"]]
eval_df = eval_df[["input_text", "target_text"]]

train_df["prefix"] = "paraphrase"
eval_df["prefix"] = "paraphrase"

# MSRP Data
train_df = pd.concat(
    [
        train_df,
        load_data("data/msr_paraphrase_train.txt", "#1 String", "#2 String", "Quality"),
    ]
)
eval_df = pd.concat(
    [
        eval_df,
        load_data("data/msr_paraphrase_test.txt", "#1 String", "#2 String", "Quality"),
    ]
)

# Quora Data

# The Quora Dataset is not separated into train/test, so we do it manually the first time.
df = load_data(
    "data/quora_duplicate_questions.tsv", "question1", "question2", "is_duplicate"
)
q_train, q_test = train_test_split(df)

q_train.to_csv("data/quora_train.tsv", sep="\t")
q_test.to_csv("data/quora_test.tsv", sep="\t")

# The code block above only needs to be run once.
# After that, the two lines below are sufficient to load the Quora dataset.

# q_train = pd.read_csv("data/quora_train.tsv", sep="\t")
# q_test = pd.read_csv("data/quora_test.tsv", sep="\t")

train_df = pd.concat([train_df, q_train])
eval_df = pd.concat([eval_df, q_test])

train_df = train_df[["prefix", "input_text", "target_text"]]
eval_df = eval_df[["prefix", "input_text", "target_text"]]

train_df = train_df.dropna()
eval_df = eval_df.dropna()

train_df["input_text"] = train_df["input_text"].apply(clean_unnecessary_spaces)
train_df["target_text"] = train_df["target_text"].apply(clean_unnecessary_spaces)

eval_df["input_text"] = eval_df["input_text"].apply(clean_unnecessary_spaces)
eval_df["target_text"] = eval_df["target_text"].apply(clean_unnecessary_spaces)

print(train_df)

model_args = Seq2SeqArgs()
model_args.eval_batch_size = 64
model_args.evaluate_during_training = True
model_args.evaluate_during_training_steps = 2500
model_args.evaluate_during_training_verbose = True
model_args.fp16 = False
model_args.learning_rate = 5e-5
model_args.max_seq_length = 128
model_args.num_train_epochs = 2
model_args.overwrite_output_dir = True
model_args.reprocess_input_data = True
model_args.save_eval_checkpoints = False
model_args.save_steps = -1
model_args.train_batch_size = 8
model_args.use_multiprocessing = False

model_args.do_sample = True
model_args.num_beams = None
model_args.num_return_sequences = 3
model_args.max_length = 128
model_args.top_k = 50
model_args.top_p = 0.95

model_args.wandb_project = "Paraphrasing with BART"


model = Seq2SeqModel(
    encoder_decoder_type="bart",
    encoder_decoder_name="facebook/bart-large",
    args=model_args,
)

model.train_model(train_df, eval_data=eval_df)

to_predict = [
    prefix + ": " + str(input_text)
    for prefix, input_text in zip(
        eval_df["prefix"].tolist(), eval_df["input_text"].tolist()
    )
]
truth = eval_df["target_text"].tolist()

preds = model.predict(to_predict)

# Saving the predictions if needed
os.makedirs("predictions", exist_ok=True)

with open(f"predictions/predictions_{datetime.now()}.txt", "w") as f:
    for i, text in enumerate(eval_df["input_text"].tolist()):
        f.write(str(text) + "\n\n")

        f.write("Truth:\n")
        f.write(truth[i] + "\n\n")

        f.write("Prediction:\n")
        for pred in preds[i]:
            f.write(str(pred) + "\n")
        f.write(
            "________________________________________________________________________________\n"
        )
