import pandas as pd
import pytest
from simpletransformers.language_modeling import LanguageModelingModel
import os


def test_language_modeling():
    with open("train.txt", "w") as f:
        for i in range(100):
            f.writelines("Hello world with Simple Transformers! \n")

    model_args = {
        "reprocess_input_data": True,
        "overwrite_output_dir": True,
        "num_train_epochs": 1,
        "no_save": True,
        "vocab_size": 100,
        "generator_config": {"embedding_size": 512, "hidden_size": 256, "num_hidden_layers": 1,},
        "discriminator_config": {"embedding_size": 512, "hidden_size": 256, "num_hidden_layers": 2,},
    }

    model = LanguageModelingModel("electra", None, args=model_args, train_files="train.txt", use_cuda=False,)

    # Train the model
    model.train_model("train.txt")
