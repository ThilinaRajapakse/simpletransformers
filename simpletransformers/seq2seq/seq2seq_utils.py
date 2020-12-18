import logging
import os
import pickle
from multiprocessing import Pool
from typing import Tuple

import pandas as pd
import torch
from tokenizers.implementations import ByteLevelBPETokenizer
from tokenizers.processors import BertProcessing
from torch.utils.data import Dataset
from tqdm.auto import tqdm
from transformers import PreTrainedTokenizer
from transformers.models.bart.modeling_bart import shift_tokens_right

logger = logging.getLogger(__name__)


def preprocess_data(data):
    input_text, target_text, encoder_tokenizer, decoder_tokenizer, args = data

    input_text = encoder_tokenizer.encode(
        input_text, max_length=args.max_seq_length, padding="max_length", return_tensors="pt", truncation=True
    )

    target_text = decoder_tokenizer.encode(
        target_text, max_length=args.max_seq_length, padding="max_length", return_tensors="pt", truncation=True
    )
    return (torch.flatten(input_text), torch.flatten(target_text))


class Seq2SeqDataset(Dataset):
    def __init__(self, encoder_tokenizer, decoder_tokenizer, args, data, mode):
        cached_features_file = os.path.join(
            args.cache_dir, args.model_name.replace("/", "_") + "_cached_" + str(args.max_seq_length) + str(len(data))
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
                (input_text, target_text, encoder_tokenizer, decoder_tokenizer, args)
                for input_text, target_text in zip(data["input_text"], data["target_text"])
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


def preprocess_data_bart(data):
    input_text, target_text, tokenizer, args = data

    input_ids = tokenizer.batch_encode_plus(
        [input_text], max_length=args.max_seq_length, padding="max_length", return_tensors="pt", truncation=True
    )

    target_ids = tokenizer.batch_encode_plus(
        [target_text], max_length=args.max_seq_length, padding="max_length", return_tensors="pt", truncation=True
    )

    return {
        "source_ids": input_ids["input_ids"].squeeze(),
        "source_mask": input_ids["attention_mask"].squeeze(),
        "target_ids": target_ids["input_ids"].squeeze(),
    }


def preprocess_data_mbart(data):
    input_text, target_text, tokenizer, args = data

    tokenized_example = tokenizer.prepare_seq2seq_batch(
        src_texts=[input_text],
        tgt_texts=[target_text],
        src_lang=args.src_lang,
        tgt_lang=args.tgt_lang,
        max_length=args.max_seq_length,
        padding="max_length",  # pad_to_max_length=True won't work in this case
        return_tensors="pt",
        truncation=True,
    )

    decoder_input_ids = tokenized_example["labels"].clone()
    decoder_input_ids = shift_tokens_right(decoder_input_ids, tokenizer.pad_token_id)

    labels = tokenized_example["labels"]
    labels[labels == tokenizer.pad_token_id] = -100

    return {
        "input_ids": tokenized_example["input_ids"].squeeze(),
        "attention_mask": tokenized_example["attention_mask"].squeeze(),
        "decoder_input_ids": decoder_input_ids.squeeze(),
        "labels": labels.squeeze(),
    }


class SimpleSummarizationDataset(Dataset):
    def __init__(self, tokenizer, args, data, mode):
        self.tokenizer = tokenizer

        cached_features_file = os.path.join(
            args.cache_dir, args.model_name + "_cached_" + str(args.max_seq_length) + str(len(data))
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
                (input_text, target_text, tokenizer, args)
                for input_text, target_text in zip(data["input_text"], data["target_text"])
            ]

            preprocess_fn = preprocess_data_mbart if args.model_type == "mbart" else preprocess_data_bart

            if args.use_multiprocessing:
                with Pool(args.process_count) as p:
                    self.examples = list(
                        tqdm(
                            p.imap(preprocess_fn, data, chunksize=args.multiprocessing_chunksize),
                            total=len(data),
                            disable=args.silent,
                        )
                    )
            else:
                self.examples = [preprocess_fn(d) for d in tqdm(data, disable=args.silent)]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index]
