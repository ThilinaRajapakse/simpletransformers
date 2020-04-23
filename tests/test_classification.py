import pandas as pd
import pytest
from simpletransformers.classification import ClassificationModel, MultiLabelClassificationModel


@pytest.mark.parametrize(
    "model_type, model_name",
    [
        ("bert", "bert-base-uncased"),
        ("xlnet", "xlnet-base-cased"),
        ("xlm", "xlm-mlm-17-1280"),
        ("roberta", "roberta-base"),
        ("distilbert", "distilbert-base-uncased"),
        ("albert", "albert-base-v1"),
        ("camembert", "camembert-base"),
        ("xlmroberta", "xlm-roberta-base"),
        ("flaubert", "flaubert-base-cased"),
    ],
)
def test_binary_classification(model_type, model_name):
    # Train and Evaluation data needs to be in a Pandas Dataframe of two columns.
    # The first column is the text with type str, and the second column is the
    # label with type int.
    train_data = [
        ["Example sentence belonging to class 1", 1],
        ["Example sentence belonging to class 0", 0],
    ]
    train_df = pd.DataFrame(train_data)

    eval_data = [
        ["Example eval sentence belonging to class 1", 1],
        ["Example eval sentence belonging to class 0", 0],
    ]
    eval_df = pd.DataFrame(eval_data)

    # Create a ClassificationModel
    model = ClassificationModel(
        model_type, model_name, use_cuda=False, args={"reprocess_input_data": True, "overwrite_output_dir": True},
    )

    # Train the model
    model.train_model(train_df)

    # Evaluate the model
    result, model_outputs, wrong_predictions = model.eval_model(eval_df)


@pytest.mark.parametrize(
    "model_type, model_name",
    [
        ("bert", "bert-base-uncased"),
        ("xlnet", "xlnet-base-cased"),
        ("xlm", "xlm-mlm-17-1280"),
        ("roberta", "roberta-base"),
        ("distilbert", "distilbert-base-uncased"),
        ("albert", "albert-base-v1"),
        ("camembert", "camembert-base"),
        ("xlmroberta", "xlm-roberta-base"),
        ("flaubert", "flaubert-base-cased"),
    ],
)
def test_multiclass_classification(model_type, model_name):
    # Train and Evaluation data needs to be in a Pandas Dataframe containing at
    # least two columns. If the Dataframe has a header, it should contain a 'text'
    # and a 'labels' column. If no header is present, the Dataframe should
    # contain at least two columns, with the first column is the text with
    # type str, and the second column in the label with type int.
    train_data = [
        ["Example sentence belonging to class 1", 1],
        ["Example sentence belonging to class 0", 0],
        ["Example eval senntence belonging to class 2", 2],
    ]
    train_df = pd.DataFrame(train_data)

    eval_data = [
        ["Example eval sentence belonging to class 1", 1],
        ["Example eval sentence belonging to class 0", 0],
        ["Example eval senntence belonging to class 2", 2],
    ]
    eval_df = pd.DataFrame(eval_data)

    # Create a ClassificationModel
    model = ClassificationModel(
        model_type,
        model_name,
        num_labels=3,
        args={"reprocess_input_data": True, "overwrite_output_dir": True},
        use_cuda=False,
    )

    # Train the model
    model.train_model(train_df)

    # Evaluate the model
    result, model_outputs, wrong_predictions = model.eval_model(eval_df)

    predictions, raw_outputs = model.predict(["Some arbitary sentence"])


@pytest.mark.parametrize(
    "model_type, model_name",
    [
        ("bert", "bert-base-uncased"),
        ("xlnet", "xlnet-base-cased"),
        ("xlm", "xlm-mlm-17-1280"),
        ("roberta", "roberta-base"),
        ("distilbert", "distilbert-base-uncased"),
        ("albert", "albert-base-v1"),
    ],
)
def test_multilabel_classification(model_type, model_name):
    # Train and Evaluation data needs to be in a Pandas Dataframe containing at
    # least two columns, a 'text' and a 'labels' column. The `labels` column
    # should contain multi-hot encoded lists.
    train_data = [["Example sentence 1 for multilabel classification.", [1, 1, 1, 1, 0, 1]]] + [
        ["This is another example sentence. ", [0, 1, 1, 0, 0, 0]]
    ]
    train_df = pd.DataFrame(train_data, columns=["text", "labels"])

    eval_data = [
        ["Example eval sentence for multilabel classification.", [1, 1, 1, 1, 0, 1]],
        ["Example eval senntence belonging to class 2", [0, 1, 1, 0, 0, 0]],
    ]
    eval_df = pd.DataFrame(eval_data)

    # Create a MultiLabelClassificationModel
    model = MultiLabelClassificationModel(
        model_type,
        model_name,
        num_labels=6,
        args={"reprocess_input_data": True, "overwrite_output_dir": True, "num_train_epochs": 1},
        use_cuda=False,
    )

    # Train the model
    model.train_model(train_df)

    # Evaluate the model
    result, model_outputs, wrong_predictions = model.eval_model(eval_df)

    predictions, raw_outputs = model.predict(["This thing is entirely different from the other thing. "])
