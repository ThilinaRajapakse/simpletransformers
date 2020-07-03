import pandas as pd
import pytest
from simpletransformers.seq2seq import Seq2SeqArgs, Seq2SeqModel
import os


@pytest.mark.parametrize(
    "encoder_decoder_type, encoder_decoder_name",
    [
        ("bart", "facebook/bart-large"),
    ],
)
def test_seq2seq(encoder_decoder_type, encoder_decoder_name):
    train_data = [
        ["one", "1"],
        ["two", "2"],
    ]

    train_df = pd.DataFrame(train_data, columns=["input_text", "target_text"])

    eval_data = [
        ["three", "3"],
        ["four", "4"],
    ]

    eval_df = pd.DataFrame(eval_data, columns=["input_text", "target_text"])

    model_args = {
        "reprocess_input_data": True,
        "overwrite_output_dir": True,
        "max_seq_length": 128,
        "train_batch_size": 2,
        "num_train_epochs": 2,
        "use_multiprocessing": False,
        "max_length": 15,
        "manual_seed": 4,
        "do_sample": False,
        "num_return_sequences": 1,
    }

    model = Seq2SeqModel(
        encoder_decoder_type=encoder_decoder_type,
        encoder_decoder_name=encoder_decoder_name,
        args=model_args,
        use_cuda=False,
    )

    model.train_model(train_df)

    model.eval_model(eval_df)

    a = model.predict(["five"])[0]

    model = Seq2SeqModel(
        encoder_decoder_type=encoder_decoder_type,
        encoder_decoder_name="outputs",
        args=model_args,
        use_cuda=False,
    )

    b = model.predict(["five"])[0]

    assert a == b


