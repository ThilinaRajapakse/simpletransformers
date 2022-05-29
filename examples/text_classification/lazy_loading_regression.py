import os

import pandas as pd

from simpletransformers.classification import ClassificationModel

train_data = [
    ["Example sentence belonging to class 1", "Yep, this is 1", 0.8],
    ["Example sentence belonging to class 0", "Yep, this is 0", 0.2],
    [
        "This is an entirely different phrase altogether and should be treated so.",
        "Is this being picked up?",
        1000.5,
    ],
]

train_df = pd.DataFrame(train_data, columns=["text_a", "text_b", "labels"])

eval_data = [
    ["Example sentence belonging to class 1", "Yep, this is 1", 1.9],
    ["Example sentence belonging to class 0", "Yep, this is 0", 0.1],
    ["Example  2 sentence belonging to class 0", "Yep, this is 0", 5],
]

eval_df = pd.DataFrame(eval_data, columns=["text_a", "text_b", "labels"])

os.makedirs("data", exist_ok=True)

train_df.to_csv("data/regression_train.tsv", sep="\t", index=False)
eval_df.to_csv("data/regression_eval.tsv", sep="\t", index=False)

train_args = {
    "reprocess_input_data": True,
    "overwrite_output_dir": True,
    "lazy_text_a_column": 0,
    "lazy_text_b_column": 1,
    "lazy_labels_column": 2,
    "lazy_header_row": True,
    "regression": True,
    "lazy_loading": True,
}

# Create a TransformerModel
model = ClassificationModel("bert", "bert-base-cased", num_labels=1, args=train_args)
# print(train_df.head())

# Train the model
model.train_model("data/regression_train.tsv")

# # # Evaluate the model
result, model_outputs, wrong_predictions = model.eval_model("data/regression_eval.tsv")

print(result)

preds, out = model.predict([["Test sentence", "Other sentence"]])

print(preds)
