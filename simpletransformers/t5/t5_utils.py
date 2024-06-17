import json
import logging
import os
import pickle
from multiprocessing import Pool
from os import truncate
from typing import Tuple
import warnings

import pandas as pd
import torch
from tokenizers.implementations import ByteLevelBPETokenizer
from tokenizers.processors import BertProcessing
from torch.utils.data import Dataset, Sampler
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


def load_hf_dataset(
    data, tokenizer, args, tokenize_targets=True, reranking=False, evaluate=False
):
    if isinstance(data, str):
        if (args.model_type == "eet5" or reranking) and os.path.isdir(data):
            dataset = load_from_disk(data)
        elif isinstance(data, str):
            if reranking:
                if args.add_prefix:
                    features = datasets.Features(
                        {
                            "prefix": datasets.Value("string"),
                            "input_text": datasets.Value("string"),
                        }
                    )
                else:
                    if evaluate:
                        features = datasets.Features(
                            {
                                "query_id": datasets.Value("string"),
                                "passage_id": datasets.Value("string"),
                                "input_text": datasets.Value("string"),
                            }
                        )
                    else:
                        features = datasets.Features(
                            {
                                "input_text": datasets.Value("string"),
                            }
                        )
            else:
                if args.add_prefix:
                    features = datasets.Features(
                        {
                            "prefix": datasets.Value("string"),
                            "input_text": datasets.Value("string"),
                            "target_text": datasets.Value("string"),
                        }
                    )
                else:
                    features = datasets.Features(
                        {
                            "input_text": datasets.Value("string"),
                            "target_text": datasets.Value("string"),
                        }
                    )

            dataset = load_dataset(
                "csv",
                data_files=data,
                delimiter="\t",
                download_mode="force_redownload"
                if args.reprocess_input_data
                else "reuse_dataset_if_exists",
                features=features,
                cache_dir=args.dataset_cache_dir,
            )
    else:
        dataset = HFDataset.from_pandas(data)

    # tokenize_targets = not (evaluate and args.model_type == "eet5")

    dataset = dataset.map(
        lambda x: preprocess_batch_for_hf_dataset(
            x, tokenizer=tokenizer, args=args, tokenize_targets=tokenize_targets
        ),
        batched=True,
    )

    if args.model_type == "eet5" or reranking:
        try:
            dataset = dataset["train"]
        except:
            pass
        # If embeddings in dataset and encoder_ouputs not in dataset, rename embeddings to encoder_outputs
        if "embeddings" in dataset.features:
            dataset = dataset.rename_column("embeddings", "encoder_outputs")

        if args.model_type == "eet5":
            if tokenize_targets:
                dataset.set_format(
                    type="pt",
                    columns=[
                        "input_ids",
                        "attention_mask",
                        "labels",
                        "encoder_outputs",
                    ],
                )
            else:
                dataset.set_format(
                    type="pt",
                    columns=["input_ids", "attention_mask", "encoder_outputs"],
                )
        else:
            if tokenize_targets:
                dataset.set_format(
                    type="pt",
                    columns=["input_ids", "attention_mask", "labels"],
                )
            else:
                dataset.set_format(type="pt", columns=["input_ids", "attention_mask"])
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


def convert_beir_to_monot5_format(
    data, run_dict=None, top_k=None, include_title=False, save_path=None
):
    """
    Utility function to convert BEIR format to MonoT5 format

    Args:
        data: A directory containing a dataset in the BEIR format
        run_dict: Path to a run file to build a reranking dataset. If not provided, all documents are considered.
                  run_dict should be a json file with the following format:
                    {
                        "query_id1": ["doc_id1": score1, "doc_id2": score2, ...],
                        "query_id2": ["doc_id1": score1, "doc_id2": score2, ...],
                        ...
                    }
        top_k: Number of documents to consider for reranking. Only used if run_dict is provided.
        include_title: Whether to include the title of the document in the MonoT5 format.
        save_path: Path to save the converted dataset. If not provided, the dataset is returned as a DataFrame.
    """

    if run_dict:
        with open(run_dict, "r") as f:
            run_dict = json.load(f)
        if top_k:
            for query_id in run_dict:
                run_dict[query_id] = dict(
                    sorted(
                        run_dict[query_id].items(), key=lambda x: x[1], reverse=True
                    )[:top_k]
                )

        # Make sure both query_id and doc_id are strings
        updated_dict = {}
        for query_id in run_dict:
            updated_dict[str(query_id)] = {
                str(k): v for k, v in run_dict[query_id].items()
            }

        run_dict = updated_dict
    else:
        if top_k:
            warnings.warn(
                "top_k is only used when run_dict is provided. Ignoring top_k."
            )

    queries_df = pd.read_json(os.path.join(data, "queries.jsonl"), lines=True)
    corpus_df = pd.read_json(os.path.join(data, "corpus.jsonl"), lines=True)

    queries_df["_id"] = queries_df["_id"].astype(str)
    corpus_df["_id"] = corpus_df["_id"].astype(str)

    if include_title:
        corpus_df["text"] = corpus_df["title"] + " " + corpus_df["text"]

    queries_df = queries_df.set_index("_id")
    corpus_df = corpus_df.set_index("_id")

    if run_dict:
        reranking_data = []
        for query_id in tqdm(run_dict, total=len(run_dict)):
            for passage_id in run_dict[query_id]:
                reranking_data.append(
                    {
                        "query_id": query_id,
                        "query": queries_df.loc[query_id]["text"],
                        "passage_id": passage_id,
                        "passage": corpus_df.loc[passage_id]["text"],
                    }
                )
    else:
        reranking_data = []
        for query_id, query in tqdm(queries_df.iterrows(), total=len(queries_df)):
            for passage_id, passage in corpus_df.iterrows():
                reranking_data.append(
                    {
                        "query_id": query_id,
                        "query": query["text"],
                        "passage_id": passage_id,
                        "passage": passage["text"],
                    }
                )

    # MonoT5 format DF should have the columns: query_id, passage_id, input_text
    # input_text should be in the format: "Query: <query> Document: <document> Relevant:"
    reranking_df = pd.DataFrame(reranking_data)
    reranking_df["input_text"] = reranking_df.apply(
        lambda x: f"Query: {x['query']} Document: {x['passage']} Relevant:", axis=1
    )

    reranking_df = reranking_df[["query_id", "passage_id", "input_text"]]

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        reranking_df.to_csv(save_path, sep="\t", index=False)

    return reranking_df


class ChunkSampler(Sampler):
    def __init__(self, data_source, chunk_size, batch_size):
        assert (
            batch_size % chunk_size == 0
        ), "Batch size must be divisible by chunk size"
        self.data_source = data_source
        self.chunk_size = chunk_size
        self.num_chunks = len(data_source) // chunk_size

    def __iter__(self):
        # Create a list of chunk indices
        chunk_indices = torch.randperm(self.num_chunks).tolist()

        # Create indices by chunk
        indices = []
        for chunk_idx in chunk_indices:
            start_idx = chunk_idx * self.chunk_size
            chunk_indices = list(range(start_idx, start_idx + self.chunk_size))
            indices.extend(chunk_indices)

        return iter(indices)

    def __len__(self):
        return len(self.data_source)
