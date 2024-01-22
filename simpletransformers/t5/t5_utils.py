import logging
import os
import pickle
from multiprocessing import Pool
from os import truncate
from typing import Tuple

import pandas as pd
import torch
from tokenizers.implementations import ByteLevelBPETokenizer
from tokenizers.processors import BertProcessing
from torch.utils.data import Dataset
from tqdm.auto import tqdm
from transformers import PreTrainedTokenizer
from datasets import load_dataset, load_from_disk
from datasets import Dataset as HFDataset
import datasets


logger = logging.getLogger(__name__)


def preprocess_batch_for_hf_dataset(dataset, tokenizer, args, tokenize_targets=True):
    if args.preprocess_inputs:
        if args.add_prefix:
            text = [
                prefix + ": " + input_text
                for prefix, input_text in zip(dataset["prefix"], dataset["input_text"])
            ]
        else:
            text = dataset["input_text"]
        model_inputs = tokenizer(
            text=text,
            max_length=args.max_seq_length,
            padding="max_length",
            return_tensors="np",
            truncation=True,
        )
        if tokenize_targets:
            with tokenizer.as_target_tokenizer():
                labels = tokenizer(
                    text=dataset["target_text"],
                    max_length=args.max_seq_length,
                    padding="max_length",
                    return_tensors="np",
                    truncation=True,
                )
            model_inputs["labels"] = labels["input_ids"]

        return model_inputs
    else:
        model_inputs = tokenizer(
            text=[
                prefix + input_text
                for prefix, input_text in zip(dataset["prefix"], dataset["input_text"])
            ]
            if args.add_prefix
            else dataset["input_text"],
            max_length=args.max_seq_length,
            padding="max_length",
            return_tensors="np",
            truncation=True,
        )
        if tokenize_targets:
            with tokenizer.as_target_tokenizer():
                labels = tokenizer(
                    text=dataset["target_text"],
                    max_length=args.max_seq_length,
                    padding="max_length",
                    return_tensors="np",
                    truncation=True,
                )
            model_inputs["labels"] = labels["input_ids"]

        return model_inputs


def load_hf_dataset(data, tokenizer, args, tokenize_targets=True, reranking=False):
    if args.model_type == "eet5" or reranking:
        dataset = load_from_disk(data)
    elif isinstance(data, str):
        dataset = load_dataset(
            "csv",
            data_files=data,
            delimiter="\t",
            download_mode="force_redownload"
            if args.reprocess_input_data
            else "reuse_dataset_if_exists",
            features=datasets.Features(
                {
                    "prefix": datasets.Value("string"),
                    "input_text": datasets.Value("string"),
                    "target_text": datasets.Value("string"),
                }
            ) if args.add_prefix else datasets.Features(
                {
                    "input_text": datasets.Value("string"),
                    "target_text": datasets.Value("string"),
                }
            ),
        )
    else:
        dataset = HFDataset.from_pandas(data)

    # tokenize_targets = not (evaluate and args.model_type == "eet5")

    dataset = dataset.map(
        lambda x: preprocess_batch_for_hf_dataset(x, tokenizer=tokenizer, args=args, tokenize_targets=tokenize_targets),
        batched=True,
    )

    if args.model_type == "eet5" or reranking:
        # If embeddings in dataset and encoder_ouputs not in dataset, rename embeddings to encoder_outputs
        if "embeddings" in dataset.features:
            dataset = dataset.rename_column("embeddings", "encoder_outputs")

        if args.model_type == "eet5":
            if tokenize_targets:
                dataset.set_format(
                    type="pt",
                    columns=["input_ids", "attention_mask", "labels", "encoder_outputs"],
                )
            else:
                dataset.set_format(
                    type="pt", columns=["input_ids", "attention_mask", "encoder_outputs"]
                )
        else:
            if tokenize_targets:
                dataset.set_format(
                    type="pt",
                    columns=["input_ids", "attention_mask", "labels"],
                )
            else:
                dataset.set_format(
                    type="pt", columns=["input_ids", "attention_mask"]
                )
    else:
        dataset.set_format(type="pt", columns=["input_ids", "attention_mask", "labels"])

    if isinstance(data, str) and not (args.model_type == "eet5" or reranking):
        # This is not necessarily a train dataset. The datasets library insists on calling it train.
        return dataset["train"]
    else:
        return dataset


def preprocess_data(data):
    prefix, input_text, target_text, tokenizer, args = data

    # Add EOS again if truncated?
    if args.preprocess_inputs:
        batch = tokenizer(
            text=[prefix + ": " + input_text],
            max_length=args.max_seq_length,
            padding="max_length",
            return_tensors="np",
            truncation=True,
        )
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                text=[target_text],
                max_length=args.max_seq_length,
                padding="max_length",
                return_tensors="np",
                truncation=True,
            )
        batch["labels"] = labels["input_ids"]
    else:
        batch = tokenizer(
            text=[prefix + input_text],
            max_length=args.max_seq_length,
            padding="max_length",
            return_tensors="np",
            truncation=True,
        )
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                text=[target_text],
                max_length=args.max_seq_length,
                padding="max_length",
                return_tensors="np",
                truncation=True,
            )
        batch["labels"] = labels["input_ids"]
    input_ids = batch["input_ids"][0]
    attention_mask = batch["attention_mask"][0]
    labels = batch["labels"][0]
    return (input_ids, attention_mask, labels)


class T5Dataset(Dataset):
    def __init__(self, tokenizer, args, data, mode):
        cached_features_file = os.path.join(
            args.cache_dir,
            args.model_name.replace("/", "_")
            + "_cached_"
            + str(args.max_seq_length)
            + str(len(data)),
        )

        if os.path.exists(cached_features_file) and (
            (not args.reprocess_input_data and not args.no_cache)
            or (mode == "dev" and args.use_cached_eval_features and not args.no_cache)
        ):
            logger.info(" Loading features from cached file %s", cached_features_file)
            with open(cached_features_file, "rb") as handle:
                self.examples = pickle.load(handle)
        else:
            logger.info(" Creating features from dataset file at %s", args.cache_dir)

            data = [
                (prefix, input_text, target_text, tokenizer, args)
                for prefix, input_text, target_text in zip(
                    data["prefix"], data["input_text"], data["target_text"]
                )
            ]

            if (mode == "train" and args.use_multiprocessing) or (
                mode == "dev" and args.use_multiprocessing_for_evaluation
            ):
                if args.multiprocessing_chunksize == -1:
                    chunksize = max(len(data) // (args.process_count * 2), 500)
                else:
                    chunksize = args.multiprocessing_chunksize

                with Pool(args.process_count) as p:
                    self.examples = list(
                        tqdm(
                            p.imap(preprocess_data, data, chunksize=chunksize),
                            total=len(data),
                            disable=args.silent,
                        )
                    )
            else:
                self.examples = [
                    preprocess_data(d) for d in tqdm(data, disable=args.silent)
                ]

            if not args.no_cache:
                logger.info(
                    " Saving features into cached file %s", cached_features_file
                )
                with open(cached_features_file, "wb") as handle:
                    pickle.dump(self.examples, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index]
