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

logger = logging.getLogger(__name__)


def preprocess_data(data):
    prefix, input_text, target_text, tokenizer, args = data

    # Add EOS again if truncated?
    if args.preprocess_inputs:
        batch = tokenizer.prepare_seq2seq_batch(
            src_texts=[prefix + ": " + input_text],
            tgt_texts=[target_text],
            max_length=args.max_seq_length,
            padding="max_length",
            return_tensors="pt",
            truncation=True,
        )
        # input_text = tokenizer.encode(
        #     prefix + ": " + input_text,
        #     max_length=args.max_seq_length,
        #     padding="max_length",
        #     return_tensors="pt",
        #     truncation=True,
        # )

        # target_text = tokenizer.encode(
        #     target_text,
        #     max_length=args.max_seq_length,
        #     padding="max_length",
        #     return_tensors="pt",
        #     truncation=True,
        # )
    else:
        batch = tokenizer.prepare_seq2seq_batch(
            src_texts=[prefix + input_text],
            tgt_texts=[target_text],
            max_length=args.max_seq_length,
            padding="max_length",
            return_tensors="pt",
            truncation=True,
        )
        # input_text = tokenizer.encode(
        #     prefix + input_text,
        #     max_length=args.max_seq_length,
        #     padding="max_length",
        #     return_tensors="pt",
        #     truncation=True,
        # )

        # target_text = tokenizer.encode(
        #     target_text, max_length=args.max_seq_length, padding="max_length", return_tensors="pt", truncation=True
        # )
    input_ids = batch["input_ids"][0]
    attention_mask = batch["attention_mask"][0]
    labels = batch["labels"][0]
    return (input_ids, attention_mask, labels)


class T5Dataset(Dataset):
    def __init__(self, tokenizer, args, data, mode):
        cached_features_file = os.path.join(
            args.cache_dir, args.model_name.replace("/", "_") + "_cached_" + str(args.max_seq_length) + str(len(data)),
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
                for prefix, input_text, target_text in zip(data["prefix"], data["input_text"], data["target_text"])
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
                        tqdm(p.imap(preprocess_data, data, chunksize=chunksize), total=len(data), disable=args.silent,)
                    )
            else:
                self.examples = [preprocess_data(d) for d in tqdm(data, disable=args.silent)]

            if not args.no_cache:
                logger.info(" Saving features into cached file %s", cached_features_file)
                with open(cached_features_file, "wb") as handle:
                    pickle.dump(self.examples, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index]
