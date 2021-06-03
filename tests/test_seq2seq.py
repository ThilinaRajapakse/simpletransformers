import os

import pandas as pd
import pytest

from simpletransformers.seq2seq import Seq2SeqArgs, Seq2SeqModel


@pytest.mark.parametrize(
    "encoder_decoder_type, encoder_decoder_name, encoder_type, use_hf_datasets",
    [
        ("bart", "facebook/bart-large", "bart", True),
        ("bart", "facebook/bart-large", "bart", False),
        ("roberta-base", "bert-base-cased", "roberta", True),
        ("roberta-base", "bert-base-cased", "roberta", False),
    ],
)
def test_seq2seq(
    encoder_decoder_type, encoder_decoder_name, encoder_type, use_hf_datasets
):
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
        "use_hf_datasets": use_hf_datasets,
    }

    if encoder_type == "bart":
        model = Seq2SeqModel(
            encoder_decoder_type=encoder_decoder_type,
            encoder_decoder_name=encoder_decoder_name,
            args=model_args,
            use_cuda=False,
        )
    else:
        model = Seq2SeqModel(
            encoder_type=encoder_type,
            encoder_name=encoder_decoder_type,
            decoder_name=encoder_decoder_name,
            args=model_args,
            use_cuda=False,
        )

    model.train_model(train_df)

    model.eval_model(eval_df)

    a = model.predict(["five"])[0]
