from os.path import dirname, join

import pandas as pd
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import classification_report

from simpletransformers.language_representation import RepresentationModel

project_root = dirname(
    dirname(dirname(dirname(__file__)))
)  # path to root of the project

MODEL_TYPE = "gpt2"  # change this to test other model types: bert, roberta, gpt2


prefix = project_root + "/data/"

train_df = pd.read_csv(prefix + "train.csv", header=None)
train_df.head()

eval_df = pd.read_csv(prefix + "test.csv", header=None)
eval_df.head()

train_df[0] = (train_df[0] == 2).astype(int)
eval_df[0] = (eval_df[0] == 2).astype(int)
# don't use entire dataset, since it's too big and will tale a long time to run, select only a portion of it
train_df = pd.DataFrame(
    {"text": train_df[1].replace(r"\n", " ", regex=True), "labels": train_df[0]}
)[:1000]
print(train_df.head())
eval_df = pd.DataFrame(
    {"text": eval_df[1].replace(r"\n", " ", regex=True), "labels": eval_df[0]}
)[:100]
print(eval_df.head())


if MODEL_TYPE == "bert":
    model_name = "bert-base-uncased"

elif MODEL_TYPE == "roberta":
    model_name = "roberta-base"
elif MODEL_TYPE == "gpt2":
    model_name = "gpt2"


model = RepresentationModel(
    model_type=MODEL_TYPE,
    model_name=model_name,
    use_cuda=False,
    args={"no_save": True, "reprocess_input_data": True, "overwrite_output_dir": True},
)

train_vectors = model.encode_sentences(
    train_df["text"].to_list(), combine_strategy="mean"
)
eval_vectors = model.encode_sentences(
    eval_df["text"].to_list(), combine_strategy="mean"
)


clf_model = RidgeClassifier()
clf_model.fit(train_vectors, train_df["labels"])
predictions = clf_model.predict(eval_vectors)
print(classification_report(eval_df["labels"], predictions))
