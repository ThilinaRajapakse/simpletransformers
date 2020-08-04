import pandas as pd

from utils import load_rte_data_file

# Preparing train data
train_df = load_rte_data_file("data/train.jsonl")
eval_df = load_rte_data_file("data/val.jsonl")
eval_df, test_df = train_test_split(eval_df, test_size=0.5, random_state=4)

eval_df.to_json("data/eval_df.jsonl", orient="records", lines=True)
test_df.to_json("data/test_df.jsonl", orient="records", lines=True)
