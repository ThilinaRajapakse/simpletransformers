import pandas as pd
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import classification_report

from simpletransformers.language_representation import RepresentationModel

train_data = [
    ["Example sentence belonging to class 1", 1],
    ["Example sentence belonging to class 0", 0],
]
train_df = pd.DataFrame(train_data, columns=["text", "target"])

eval_data = [
    ["Example eval sentence belonging to class 1", 1],
    ["Example eval sentence belonging to class 0", 0],
]
eval_df = pd.DataFrame(eval_data, columns=["text", "target"])

model = RepresentationModel(
    model_type="bert",
    model_name="bert-base-uncased",
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
clf_model.fit(train_vectors, train_df["target"])
predictions = clf_model.predict(eval_vectors)
print(classification_report(eval_df["target"], predictions))
