import os

import pandas as pd
import pytest

from simpletransformers.language_modeling import LanguageModelingModel


@pytest.mark.parametrize(
    "model_type, model_name",
    [
        ("bert", "bert-base-uncased"),
        ("longformer", "allenai/longformer-base-4096"),
        ("bert", None),
        ("electra", None),
        ("longformer", None),
        # ("xlnet", "xlnet-base-cased"),
        # ("xlm", "xlm-mlm-17-1280"),
        ("roberta", "roberta-base"),
        # ("distilbert", "distilbert-base-uncased"),
        # ("albert", "albert-base-v1"),
        # ("camembert", "camembert-base"),
        # ("xlmroberta", "xlm-roberta-base"),
        # ("flaubert", "flaubert-base-cased"),
    ],
)
def test_language_modeling(model_type, model_name):
    with open("train.txt", "w") as f:
        for i in range(100):
            f.writelines("Hello world with Simple Transformers! \n")

    if model_type == "electra":
        model_args = {
            "reprocess_input_data": True,
            "overwrite_output_dir": True,
            "num_train_epochs": 1,
            "no_save": True,
            "vocab_size": 100,
            "generator_config": {
                "embedding_size": 512,
                "hidden_size": 256,
                "num_hidden_layers": 1,
            },
            "discriminator_config": {
                "embedding_size": 512,
                "hidden_size": 256,
                "num_hidden_layers": 2,
            },
        }
    else:
        model_args = {
            "reprocess_input_data": True,
            "overwrite_output_dir": True,
            "num_train_epochs": 1,
            "no_save": True,
        }
        if model_name is None:
            model_args["vocab_size"] = 100

    model = LanguageModelingModel(
        model_type,
        model_name,
        args=model_args,
        train_files="train.txt",
        use_cuda=False,
    )

    # Train the model
    model.train_model("train.txt")
