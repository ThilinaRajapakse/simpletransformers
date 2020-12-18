import pandas as pd
import pytest

from simpletransformers.t5 import T5Model


def test_t5():
    train_data = [
        ["convert", "one", "1"],
        ["convert", "two", "2"],
    ]

    train_df = pd.DataFrame(train_data, columns=["prefix", "input_text", "target_text"])

    eval_data = [
        ["convert", "three", "3"],
        ["convert", "four", "4"],
    ]

    eval_df = pd.DataFrame(eval_data, columns=["prefix", "input_text", "target_text"])

    eval_df = train_df.copy()

    model_args = {
        "reprocess_input_data": True,
        "overwrite_output_dir": True,
        "max_seq_length": 10,
        "train_batch_size": 2,
        "num_train_epochs": 2,
        "save_model_every_epoch": False,
        "max_length": 20,
        "num_beams": 1,
    }

    # Create T5 Model
    model = T5Model("t5", "t5-base", args=model_args, use_cuda=False)

    # Train T5 Model on new task
    model.train_model(train_df)

    # Evaluate T5 Model on new task
    model.eval_model(eval_df)

    # Predict with trained T5 model
    model.predict(["convert: four", "convert: five"])

    # Load test
    model = T5Model("t5", "outputs", args=model_args, use_cuda=False)

    # Evaluate T5 Model on new task
    model.eval_model(eval_df)

    # Predict with trained T5 model
    model.predict(["convert: four", "convert: five"])
