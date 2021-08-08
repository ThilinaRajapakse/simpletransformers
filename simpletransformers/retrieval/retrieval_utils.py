import logging
import os
import pickle
from multiprocessing import Pool
from functools import partial

import torch
import transformers

from torch.utils.data import Dataset
from tqdm.auto import tqdm

from datasets import Features, Sequence, Value, load_dataset
from datasets import Dataset as HFDataset


def load_hf_dataset(data, context_tokenizer, question_tokenizer, args):
    if isinstance(data, str):
        dataset = load_dataset(
            "csv",
            data_files=data,
            delimiter="\t",
            download_mode="force_redownload"
            if args.reprocess_input_data
            else "reuse_dataset_if_exists",
            cache_dir=args.dataset_cache_dir,
        )
    else:
        dataset = HFDataset.from_pandas(data)

    dataset = dataset.map(
        lambda x: preprocess_batch_for_hf_dataset(
            x,
            context_tokenizer=context_tokenizer,
            question_tokenizer=question_tokenizer,
            args=args,
        ),
        batched=True,
    )

    column_names = [
        "context_ids",
        "question_ids",
        "context_mask",
        "question_mask",
    ]

    dataset.set_format(type="pt", columns=column_names)

    if isinstance(data, str):
        # This is not necessarily a train dataset. The datasets library insists on calling it train.
        return dataset["train"]
    else:
        return dataset


def preprocess_batch_for_hf_dataset(
    dataset, context_tokenizer, question_tokenizer, args
):
    context_inputs = context_tokenizer(
        dataset["gold_passage"],
        max_length=args.max_seq_length,
        padding="max_length",
        return_tensors="np",
        truncation=True,
    )

    question_inputs = question_tokenizer(
        dataset["question_text"],
        max_length=args.max_seq_length,
        padding="max_length",
        return_tensors="np",
        truncation=True,
    )

    context_ids = context_inputs["input_ids"].squeeze()
    question_ids = question_inputs["input_ids"].squeeze()
    context_mask = context_inputs["attention_mask"].squeeze()
    question_mask = question_inputs["attention_mask"].squeeze()

    return {
        "context_ids": context_ids,
        "question_ids": question_ids,
        "context_mask": context_mask,
        "question_mask": question_mask,
    }
