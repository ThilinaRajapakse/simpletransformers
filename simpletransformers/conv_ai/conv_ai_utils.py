# Copyright (c) 2019-present, HuggingFace Inc.
# All rights reserved. This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import json
import logging
import os
import socket
import tarfile
import tempfile
from datetime import datetime
from multiprocessing import Pool

from tqdm.auto import tqdm

import torch
from transformers import cached_path

PERSONACHAT_URL = "https://s3.amazonaws.com/datasets.huggingface.co/personachat/personachat_self_original.json"
HF_FINETUNED_MODEL = (
    "https://s3.amazonaws.com/models.huggingface.co/transfer-learning-chatbot/gpt_personachat_cache.tar.gz"  # noqa
)

logger = logging.getLogger(__file__)


def download_pretrained_model():
    """ Download and extract finetuned model from S3 """
    resolved_archive_file = cached_path(HF_FINETUNED_MODEL)
    tempdir = tempfile.mkdtemp()
    logger.info("extracting archive file {} to temp dir {}".format(resolved_archive_file, tempdir))
    with tarfile.open(resolved_archive_file, "r:gz") as archive:
        archive.extractall(tempdir)
    return tempdir


def tokenize_multi(data):
    obj, tokenizer = data
    if isinstance(obj, str):
        return tokenizer.convert_tokens_to_ids(tokenizer.tokenize(obj))
    if isinstance(obj, dict):
        return dict((n, tokenize_multi((o, tokenizer))) for n, o in obj.items())
    return list(tokenize_multi((o, tokenizer)) for o in obj)


def get_dataset(
    tokenizer,
    dataset_path,
    dataset_cache,
    process_count,
    proxies,
    evaluate=False,
    interact=False,
    no_cache=False,
    args=None,
):
    """ Get tokenized PERSONACHAT dataset from S3 or cache."""
    dataset_path = dataset_path or PERSONACHAT_URL

    mode = "eval" if evaluate else "train"
    if interact:
        mode = "interact"

    dataset_cache = (
        dataset_cache + "_" + type(tokenizer).__name__ + "_" + mode
    )  # To avoid using GPT cache for GPT-2 and vice-versa
    if dataset_cache and os.path.isfile(dataset_cache) and not no_cache:
        logger.info("Load tokenized dataset from cache at %s", dataset_cache)
        dataset = torch.load(dataset_cache)
    else:
        logger.info("Download dataset from %s", dataset_path)
        personachat_file = cached_path(dataset_path, proxies=proxies)
        with open(personachat_file, "r", encoding="utf-8") as f:
            dataset = json.loads(f.read())

        logger.info("Tokenize and encode the dataset")

        def tokenize(obj):
            if isinstance(obj, str):
                return tokenizer.convert_tokens_to_ids(tokenizer.tokenize(obj))
            if isinstance(obj, dict):
                return dict((n, tokenize(o)) for n, o in obj.items())

            data = [(d, tokenizer) for d in obj]
            with Pool(process_count) as p:
                tokenized_data = list(
                    tqdm(p.imap(tokenize_multi, data, chunksize=args["multiprocessing_chunksize"]), total=len(data))
                )
            return tokenized_data

        if not interact and dataset_path == PERSONACHAT_URL:
            if not evaluate:
                dataset = dataset["train"]
            else:
                dataset = dataset["valid"]

        dataset = tokenize(dataset)
        torch.save(dataset, dataset_cache)
    return dataset


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self
