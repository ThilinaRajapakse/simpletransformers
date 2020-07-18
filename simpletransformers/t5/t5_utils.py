import logging
import os
from os import truncate
import pickle
from multiprocessing import Pool
from typing import Tuple

from tqdm.auto import tqdm

import pandas as pd
import torch
from tokenizers.implementations import ByteLevelBPETokenizer
from tokenizers.processors import BertProcessing
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

logger = logging.getLogger(__name__)


def preprocess_data(data):
    prefix, input_text, target_text, tokenizer, args = data

    # Add EOS again if truncated?
    if args.preprocess_inputs:
        input_text = tokenizer.encode(
            prefix + ": " + input_text + " </s>",
            max_length=args.max_seq_length,
            pad_to_max_length=True,
            return_tensors="pt",
        )

        target_text = tokenizer.encode(
            target_text + " </s>", max_length=args.max_seq_length, pad_to_max_length=True, return_tensors="pt"
        )
    else:
        input_text = tokenizer.encode(
            prefix + input_text, max_length=args.max_seq_length, pad_to_max_length=True, return_tensors="pt"
        )

        target_text = tokenizer.encode(
            target_text, max_length=args.max_seq_length, pad_to_max_length=True, return_tensors="pt"
        )
    return (torch.flatten(input_text), torch.flatten(target_text))


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

            if args.use_multiprocessing:
                with Pool(args.process_count) as p:
                    self.examples = list(
                        tqdm(
                            p.imap(preprocess_data, data, chunksize=args.multiprocessing_chunksize),
                            total=len(data),
                            disable=args.silent,
                        )
                    )
            else:
                self.examples = [preprocess_data(d) for d in tqdm(data, disable=args.silent)]

            logger.info(" Saving features into cached file %s", cached_features_file)
            with open(cached_features_file, "wb") as handle:
                pickle.dump(self.examples, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index]
