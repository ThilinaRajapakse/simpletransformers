import pandas as pd

from simpletransformers.classification import MultiLabelClassificationModel

# Train and Evaluation data needs to be in a Pandas Dataframe containing at least two columns, a 'text' and a 'labels' column. The `labels` column should contain multi-hot encoded lists.
train_data = [
    ["Example sentence 1 for multilabel classification.", [1, 1, 1, 1, 0, 1]]
] + [["This is another example sentence. ", [0, 1, 1, 0, 0, 0]]]
train_df = pd.DataFrame(train_data, columns=["text", "labels"])

eval_data = [
    ["Example eval sentence for multilabel classification.", [1, 1, 1, 1, 0, 1]],
    ["Example eval senntence belonging to class 2", [0, 1, 1, 0, 0, 0]],
]
eval_df = pd.DataFrame(eval_data)

# Create a MultiLabelClassificationModel
model = MultiLabelClassificationModel(
    "roberta",
    "roberta-base",
    num_labels=6,
    args={
        "reprocess_input_data": True,
        "overwrite_output_dir": True,
        "num_train_epochs": 5,
    },
)

# You can set class weights by using the optional weight argument
print(train_df.head())

# Train the model
model.train_model(train_df)

# Evaluate the model
result, model_outputs, wrong_predictions = model.eval_model(eval_df)
print(result)
print(model_outputs)

predictions, raw_outputs = model.predict(
    ["This thing is entirely different from the other thing. "]
)
print(predictions)
print(raw_outputs)
