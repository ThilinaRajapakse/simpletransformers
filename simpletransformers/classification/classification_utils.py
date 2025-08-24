# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import, division, print_function

import csv
import json
import logging
import linecache
import os
import shutil
import sys
from collections import Counter
from io import open
from multiprocessing import Pool, cpu_count
import warnings

try:
    from collections import Iterable, Mapping
except ImportError:
    from collections.abc import Iterable, Mapping

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, Sampler
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import f1_score, matthews_corrcoef

from torch.utils.data import Dataset
from datasets import load_dataset
from datasets import Dataset as HFDataset
from tqdm.auto import tqdm

try:
    import torchvision
    import torchvision.transforms as transforms

    torchvision_available = True
    from PIL import Image
except ImportError:
    torchvision_available = False

from copy import deepcopy

csv.field_size_limit(2147483647)

logger = logging.getLogger(__name__)


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(
        self, guid, text_a, text_b=None, label=None, x0=None, y0=None, x1=None, y1=None
    ):
        """
        Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """

        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label
        if x0 is None:
            self.bboxes = None
        else:
            self.bboxes = [[a, b, c, d] for a, b, c, d in zip(x0, y0, x1, y1)]

    def __repr__(self):
        if self.bboxes:
            return str(
                {
                    "guid": self.guid,
                    "text_a": self.text_a,
                    "text_b": self.text_b,
                    "label": self.label,
                    "bboxes": self.bboxes,
                }
            )
        else:
            return str(
                {
                    "guid": self.guid,
                    "text_a": self.text_a,
                    "text_b": self.text_b,
                    "label": self.label,
                }
            )


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id, bboxes=None):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        if bboxes:
            self.bboxes = bboxes


def preprocess_data_multiprocessing(data):
    text_a, text_b, tokenizer, max_seq_length = data

    examples = tokenizer(
        text=text_a,
        text_pair=text_b,
        max_length=max_seq_length,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )

    return examples


def preprocess_batch_for_hf_dataset(
    dataset, tokenizer, max_seq_length, return_tensors=None, global_attention_fn=None
):
    if "text_b" in dataset:
        tokenized_dict = tokenizer(
            text=dataset["text_a"],
            text_pair=dataset["text_b"],
            truncation=True,
            padding="max_length",
            max_length=max_seq_length,
            return_tensors=return_tensors,
        )
    else:
        tokenized_dict = tokenizer(
            text=dataset["text"],
            truncation=True,
            padding="max_length",
            max_length=max_seq_length,
            return_tensors=return_tensors,
        )

    if global_attention_fn:
        if "text_b" in dataset:
            tokenized_dict["global_attention_mask"] = global_attention_fn(
                tokenized_dict["input_ids"],
                dataset["text_a"],
                dataset["text_b"],
            )
        else:
            tokenized_dict["global_attention_mask"] = global_attention_fn(
                tokenized_dict["input_ids"], dataset["text"]
            )

    return tokenized_dict


def preprocess_data(text_a, text_b, labels, tokenizer, max_seq_length):
    return tokenizer(
        text=text_a,
        text_pair=text_b,
        truncation="only_second" if text_b else "only_first",
        padding="max_length",
        # padding="longest",
        max_length=max_seq_length,
        return_tensors="pt",
    )

    # if examples[0].text_b:
    #     tokenized_example = tokenizer.batch_encode_plus(
    #         text=example.text_a,
    #         text_pair=example.text_b,
    #         max_length=max_seq_length,
    #         truncation=True,
    #         padding="max_length",
    #         return_tensors="pt",
    #     )
    # else:
    #     tokenized_example = tokenizer.batch_encode_plus(
    #         text=example.text_a,
    #         max_length=max_seq_length,
    #         truncation=True,
    #         padding="max_length",
    #         return_tensors="pt",
    #     )

    # return [tokenized_example, [example.label for example in data]]


def build_classification_dataset(
    data,
    tokenizer,
    args,
    mode,
    multi_label,
    output_mode,
    no_cache,
    global_attention_fn=None,
):
    cached_features_file = os.path.join(
        args.cache_dir,
        "cached_{}_{}_{}_{}_{}".format(
            mode,
            args.model_type,
            args.max_seq_length,
            len(args.labels_list),
            len(data),
        ),
    )

    if os.path.exists(cached_features_file) and (
        (not args.reprocess_input_data and not args.no_cache)
        or (mode == "dev" and args.use_cached_eval_features and not args.no_cache)
    ):
        data = torch.load(cached_features_file, weights_only=False)
        logger.info(f" Features loaded from cache at {cached_features_file}")
        examples, labels = data
    else:
        logger.info(" Converting to features started. Cache is not used.")

        if len(data) == 3:
            # Sentence pair task
            text_a, text_b, labels = data
        else:
            text_a, labels = data
            text_b = None

        # If labels_map is defined, then labels need to be replaced with ints
        if args.labels_map and not args.regression:
            if multi_label:
                labels = [[args.labels_map[l] for l in label] for label in labels]
            else:
                labels = [args.labels_map[label] for label in labels]

        if (mode == "train" and args.use_multiprocessing) or (
            mode == "dev" and args.use_multiprocessing_for_evaluation
        ):
            if args.multiprocessing_chunksize == -1:
                chunksize = max(len(data) // (args.process_count * 2), 500)
            else:
                chunksize = args.multiprocessing_chunksize

            if text_b is not None:
                data = [
                    (
                        text_a[i : i + chunksize],
                        text_b[i : i + chunksize],
                        tokenizer,
                        args.max_seq_length,
                    )
                    for i in range(0, len(text_a), chunksize)
                ]
            else:
                data = [
                    (text_a[i : i + chunksize], None, tokenizer, args.max_seq_length)
                    for i in range(0, len(text_a), chunksize)
                ]

            with Pool(args.process_count) as p:
                examples = list(
                    tqdm(
                        p.imap(preprocess_data_multiprocessing, data),
                        total=len(text_a) // chunksize,
                        disable=args.silent,
                    )
                )

            examples = {
                key: torch.cat([example[key] for example in examples])
                for key in examples[0]
            }

            if global_attention_fn is not None:
                warnings.warn(
                    "Global attention masks are not supported with multiprocessing. "
                    "Please disable multiprocessing to use global attention masks."
                )
        else:
            dataset = HFDataset.from_dict(
                {
                    "text_a": text_a,
                    "text_b": text_b,
                    "labels": labels,
                }
            )
            dataset = dataset.map(
                lambda x: preprocess_batch_for_hf_dataset(
                    x,
                    tokenizer=tokenizer,
                    max_seq_length=args.max_seq_length,
                    global_attention_fn=global_attention_fn,
                ),
                batched=True,
            )

            # examples = preprocess_data(
            #     text_a, text_b, labels, tokenizer, args.max_seq_length
            # )

            dataset.set_format(type="torch")

            labels = dataset["labels"]

            if global_attention_fn is None:
                examples = {
                    "input_ids": dataset["input_ids"],
                    "attention_mask": dataset["attention_mask"],
                }
            else:
                examples = {
                    "input_ids": dataset["input_ids"],
                    "attention_mask": dataset["attention_mask"],
                    "global_attention_mask": dataset["global_attention_mask"],
                }

        if output_mode == "classification":
            labels = torch.tensor(labels, dtype=torch.long)
        elif output_mode == "regression":
            labels = torch.tensor(labels, dtype=torch.float)

        data = (examples, labels)

        if not args.no_cache and not no_cache:
            logger.info(" Saving features into cached file %s", cached_features_file)
            torch.save(data, cached_features_file)

    return (examples, labels)


class ClassificationDataset(Dataset):
    def __init__(
        self,
        data,
        tokenizer,
        args,
        mode,
        multi_label,
        output_mode,
        no_cache,
        global_attention_fn=None,
    ):
        self.examples, self.labels = build_classification_dataset(
            data,
            tokenizer,
            args,
            mode,
            multi_label,
            output_mode,
            no_cache,
            global_attention_fn,
        )

    def __len__(self):
        return len(self.examples["input_ids"])

    def __getitem__(self, index):
        return (
            {key: self.examples[key][index] for key in self.examples},
            self.labels[index],
        )


def map_labels_to_numeric(example, multi_label, args):
    if multi_label:
        example["labels"] = [args.labels_map[label] for label in example["labels"]]
    else:
        example["labels"] = args.labels_map[example["labels"]]

    return example


def load_hf_dataset(
    data, tokenizer, args, multi_label, reranking=False, global_attention_fn=None
):
    if isinstance(data, str):
        dataset = load_dataset(
            "csv",
            data_files=data,
            delimiter="\t",
            download_mode=(
                "force_redownload"
                if args.reprocess_input_data
                else "reuse_dataset_if_exists"
            ),
        )
    else:
        dataset = HFDataset.from_pandas(data)

    if args.labels_map and not args.regression:
        dataset = dataset.map(lambda x: map_labels_to_numeric(x, multi_label, args))

    dataset = dataset.map(
        lambda x: preprocess_batch_for_hf_dataset(
            x,
            tokenizer=tokenizer,
            max_seq_length=args.max_seq_length,
            global_attention_fn=global_attention_fn,
        ),
        batched=True,
    )

    if reranking:
        if args.model_type in ["bert", "xlnet", "albert", "layoutlm", "layoutlmv2"]:
            columns = ["input_ids", "token_type_ids", "attention_mask"]
        else:
            columns = ["input_ids", "attention_mask"]
    else:
        if args.model_type in ["bert", "xlnet", "albert", "layoutlm", "layoutlmv2"]:
            columns = ["input_ids", "token_type_ids", "attention_mask", "labels"]
        else:
            columns = ["input_ids", "attention_mask", "labels"]

    if global_attention_fn is not None:
        columns.append("global_attention_mask")

    dataset.set_format(type="pt", columns=columns)

    if isinstance(data, str):
        # This is not necessarily a train dataset. The datasets library insists on calling it train.
        return dataset["train"]
    else:
        return dataset


def convert_example_to_feature(
    example_row,
    pad_token=0,
    sequence_a_segment_id=0,
    sequence_b_segment_id=1,
    cls_token_segment_id=1,
    pad_token_segment_id=0,
    mask_padding_with_zero=True,
    sep_token_extra=False,
):
    (
        example,
        max_seq_length,
        tokenizer,
        output_mode,
        cls_token_at_end,
        cls_token,
        sep_token,
        cls_token_segment_id,
        pad_on_left,
        pad_token_segment_id,
        sep_token_extra,
        multi_label,
        stride,
        pad_token,
        add_prefix_space,
        pad_to_max_length,
    ) = example_row

    bboxes = []
    if example.bboxes:
        tokens_a = []
        for word, bbox in zip(example.text_a.split(), example.bboxes):
            word_tokens = tokenizer.tokenize(word)
            tokens_a.extend(word_tokens)
            bboxes.extend([bbox] * len(word_tokens))

        cls_token_box = [0, 0, 0, 0]
        sep_token_box = [1000, 1000, 1000, 1000]
        pad_token_box = [0, 0, 0, 0]

    else:
        if add_prefix_space and not example.text_a.startswith(" "):
            tokens_a = tokenizer.tokenize(" " + example.text_a)
        else:
            tokens_a = tokenizer.tokenize(example.text_a)

    tokens_b = None
    if example.text_b:
        if add_prefix_space and not example.text_b.startswith(" "):
            tokens_b = tokenizer.tokenize(" " + example.text_b)
        else:
            tokens_b = tokenizer.tokenize(example.text_b)
        # Modifies `tokens_a` and `tokens_b` in place so that the total
        # length is less than the specified length.
        # Account for [CLS], [SEP], [SEP] with "- 3". " -4" for RoBERTa.
        special_tokens_count = 4 if sep_token_extra else 3
        _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - special_tokens_count)
    else:
        # Account for [CLS] and [SEP] with "- 2" and with "- 3" for RoBERTa.
        special_tokens_count = 3 if sep_token_extra else 2
        if len(tokens_a) > max_seq_length - special_tokens_count:
            tokens_a = tokens_a[: (max_seq_length - special_tokens_count)]
            if example.bboxes:
                bboxes = bboxes[: (max_seq_length - special_tokens_count)]

    # The convention in BERT is:
    # (a) For sequence pairs:
    #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
    #  type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1
    # (b) For single sequences:
    #  tokens:   [CLS] the dog is hairy . [SEP]
    #  type_ids:   0   0   0   0  0     0   0
    #
    # Where "type_ids" are used to indicate whether this is the first
    # sequence or the second sequence. The embedding vectors for `type=0` and
    # `type=1` were learned during pre-training and are added to the wordpiece
    # embedding vector (and position vector). This is not *strictly* necessary
    # since the [SEP] token unambiguously separates the sequences, but it makes
    # it easier for the model to learn the concept of sequences.
    #
    # For classification tasks, the first vector (corresponding to [CLS]) is
    # used as as the "sentence vector". Note that this only makes sense because
    # the entire model is fine-tuned.
    tokens = tokens_a + [sep_token]
    segment_ids = [sequence_a_segment_id] * len(tokens)

    if bboxes:
        bboxes += [sep_token_box]

    if tokens_b:
        if sep_token_extra:
            tokens += [sep_token]
            segment_ids += [sequence_b_segment_id]

        tokens += tokens_b + [sep_token]

        segment_ids += [sequence_b_segment_id] * (len(tokens_b) + 1)

    if cls_token_at_end:
        tokens = tokens + [cls_token]
        segment_ids = segment_ids + [cls_token_segment_id]
    else:
        tokens = [cls_token] + tokens
        segment_ids = [cls_token_segment_id] + segment_ids
        if bboxes:
            bboxes = [cls_token_box] + bboxes

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

    # Zero-pad up to the sequence length.
    if pad_to_max_length:
        padding_length = max_seq_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            input_mask = (
                [0 if mask_padding_with_zero else 1] * padding_length
            ) + input_mask
            segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
        else:
            input_ids = input_ids + ([pad_token] * padding_length)
            input_mask = input_mask + (
                [0 if mask_padding_with_zero else 1] * padding_length
            )
            segment_ids = segment_ids + ([pad_token_segment_id] * padding_length)
            if bboxes:
                bboxes += [pad_token_box] * padding_length

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        if bboxes:
            assert len(bboxes) == max_seq_length
    # if output_mode == "classification":
    #     label_id = label_map[example.label]
    # elif output_mode == "regression":
    #     label_id = float(example.label)
    # else:
    #     raise KeyError(output_mode)

    # if output_mode == "regression":
    #     label_id = float(example.label)

    if bboxes:
        return InputFeatures(
            input_ids=input_ids,
            input_mask=input_mask,
            segment_ids=segment_ids,
            label_id=example.label,
            bboxes=bboxes,
        )
    else:
        return InputFeatures(
            input_ids=input_ids,
            input_mask=input_mask,
            segment_ids=segment_ids,
            label_id=example.label,
        )


def convert_example_to_feature_sliding_window(
    example_row,
    pad_token=0,
    sequence_a_segment_id=0,
    sequence_b_segment_id=1,
    cls_token_segment_id=1,
    pad_token_segment_id=0,
    mask_padding_with_zero=True,
    sep_token_extra=False,
):
    (
        example,
        max_seq_length,
        tokenizer,
        output_mode,
        cls_token_at_end,
        cls_token,
        sep_token,
        cls_token_segment_id,
        pad_on_left,
        pad_token_segment_id,
        sep_token_extra,
        multi_label,
        stride,
        pad_token,
        add_prefix_space,
        pad_to_max_length,
    ) = example_row

    if stride < 1:
        stride = int(max_seq_length * stride)

    bucket_size = max_seq_length - (3 if sep_token_extra else 2)
    token_sets = []

    if add_prefix_space and not example.text_a.startswith(" "):
        tokens_a = tokenizer.tokenize(" " + example.text_a)
    else:
        tokens_a = tokenizer.tokenize(example.text_a)

    if len(tokens_a) > bucket_size:
        token_sets = [
            tokens_a[i : i + bucket_size] for i in range(0, len(tokens_a), stride)
        ]
    else:
        token_sets.append(tokens_a)

    if example.text_b:
        raise ValueError(
            "Sequence pair tasks not implemented for sliding window tokenization."
        )

    # The convention in BERT is:
    # (a) For sequence pairs:
    #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
    #  type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1
    # (b) For single sequences:
    #  tokens:   [CLS] the dog is hairy . [SEP]
    #  type_ids:   0   0   0   0  0     0   0
    #
    # Where "type_ids" are used to indicate whether this is the first
    # sequence or the second sequence. The embedding vectors for `type=0` and
    # `type=1` were learned during pre-training and are added to the wordpiece
    # embedding vector (and position vector). This is not *strictly* necessary
    # since the [SEP] token unambiguously separates the sequences, but it makes
    # it easier for the model to learn the concept of sequences.
    #
    # For classification tasks, the first vector (corresponding to [CLS]) is
    # used as as the "sentence vector". Note that this only makes sense because
    # the entire model is fine-tuned.

    input_features = []
    for tokens_a in token_sets:
        tokens = tokens_a + [sep_token]
        segment_ids = [sequence_a_segment_id] * len(tokens)

        if cls_token_at_end:
            tokens = tokens + [cls_token]
            segment_ids = segment_ids + [cls_token_segment_id]
        else:
            tokens = [cls_token] + tokens
            segment_ids = [cls_token_segment_id] + segment_ids

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_seq_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            input_mask = (
                [0 if mask_padding_with_zero else 1] * padding_length
            ) + input_mask
            segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
        else:
            input_ids = input_ids + ([pad_token] * padding_length)
            input_mask = input_mask + (
                [0 if mask_padding_with_zero else 1] * padding_length
            )
            segment_ids = segment_ids + ([pad_token_segment_id] * padding_length)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        # if output_mode == "classification":
        #     label_id = label_map[example.label]
        # elif output_mode == "regression":
        #     label_id = float(example.label)
        # else:
        #     raise KeyError(output_mode)

        input_features.append(
            InputFeatures(
                input_ids=input_ids,
                input_mask=input_mask,
                segment_ids=segment_ids,
                label_id=example.label,
            )
        )

    return input_features


def convert_examples_to_features(
    examples,
    max_seq_length,
    tokenizer,
    output_mode,
    cls_token_at_end=False,
    sep_token_extra=False,
    pad_on_left=False,
    cls_token="[CLS]",
    sep_token="[SEP]",
    pad_token=0,
    sequence_a_segment_id=0,
    sequence_b_segment_id=1,
    cls_token_segment_id=1,
    pad_token_segment_id=0,
    mask_padding_with_zero=True,
    process_count=cpu_count() - 2,
    multi_label=False,
    silent=False,
    use_multiprocessing=True,
    sliding_window=False,
    flatten=False,
    stride=None,
    add_prefix_space=False,
    pad_to_max_length=True,
    args=None,
):
    """Loads a data file into a list of `InputBatch`s
    `cls_token_at_end` define the location of the CLS token:
        - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
        - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
    `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
    """

    examples = [
        (
            example,
            max_seq_length,
            tokenizer,
            output_mode,
            cls_token_at_end,
            cls_token,
            sep_token,
            cls_token_segment_id,
            pad_on_left,
            pad_token_segment_id,
            sep_token_extra,
            multi_label,
            stride,
            pad_token,
            add_prefix_space,
            pad_to_max_length,
        )
        for example in examples
    ]

    if use_multiprocessing:
        if args.multiprocessing_chunksize == -1:
            chunksize = max(len(examples) // (args.process_count * 2), 500)
        else:
            chunksize = args.multiprocessing_chunksize
        if sliding_window:
            with Pool(process_count) as p:
                features = list(
                    tqdm(
                        p.imap(
                            convert_example_to_feature_sliding_window,
                            examples,
                            chunksize=chunksize,
                        ),
                        total=len(examples),
                        disable=silent,
                    )
                )
            if flatten:
                features = [
                    feature for feature_set in features for feature in feature_set
                ]
        else:
            with Pool(process_count) as p:
                features = list(
                    tqdm(
                        p.imap(
                            convert_example_to_feature, examples, chunksize=chunksize
                        ),
                        total=len(examples),
                        disable=silent,
                    )
                )
    else:
        if sliding_window:
            features = [
                convert_example_to_feature_sliding_window(example)
                for example in tqdm(examples, disable=silent)
            ]
            if flatten:
                features = [
                    feature for feature_set in features for feature in feature_set
                ]
        else:
            features = [
                convert_example_to_feature(example)
                for example in tqdm(examples, disable=silent)
            ]

    return features


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.

    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


POOLING_BREAKDOWN = {
    1: (1, 1),
    2: (2, 1),
    3: (3, 1),
    4: (2, 2),
    5: (5, 1),
    6: (3, 2),
    7: (7, 1),
    8: (4, 2),
    9: (3, 3),
}


class ImageEncoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        model = torchvision.models.resnet152(pretrained=True)
        modules = list(model.children())[:-2]
        self.model = nn.Sequential(*modules)
        self.pool = nn.AdaptiveAvgPool2d(POOLING_BREAKDOWN[args.num_image_embeds])

    def forward(self, x):
        # Bx3x224x224 -> Bx2048x7x7 -> Bx2048xN -> BxNx2048
        out = self.pool(self.model(x))
        out = torch.flatten(out, start_dim=2)
        out = out.transpose(1, 2).contiguous()
        return out  # BxNx2048


class JsonlDataset(Dataset):
    def __init__(
        self,
        data_path,
        tokenizer,
        transforms,
        labels,
        max_seq_length,
        files_list=None,
        image_path=None,
        text_label=None,
        labels_label=None,
        images_label=None,
        image_type_extension=None,
        data_type_extension=None,
        multi_label=False,
    ):
        self.text_label = text_label if text_label else "text"
        self.labels_label = labels_label if labels_label else "labels"
        self.images_label = images_label if images_label else "images"
        self.image_type_extension = image_type_extension if image_type_extension else ""
        self.data_type_extension = data_type_extension if data_type_extension else ""
        self.multi_label = multi_label

        if isinstance(files_list, str):
            files_list = json.load(open(files_list))
        if isinstance(data_path, str):
            if not files_list:
                files_list = [
                    f
                    for f in os.listdir(data_path)
                    if f.endswith(self.data_type_extension)
                ]
            self.data = [
                dict(
                    json.load(
                        open(os.path.join(data_path, l + self.data_type_extension))
                    ),
                    **{"images": l + image_type_extension},
                )
                for l in files_list
            ]
            self.data_dir = os.path.dirname(data_path)
        else:
            data_path[self.images_label] = data_path[self.images_label].apply(
                lambda x: x + self.image_type_extension
            )
            self.data = data_path.to_dict("records")
            self.data_dir = image_path
        self.tokenizer = tokenizer
        self.labels = labels
        self.n_classes = len(labels)
        self.max_seq_length = max_seq_length

        self.transforms = transforms

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sentence = torch.LongTensor(
            self.tokenizer.encode(
                self.data[index][self.text_label], add_special_tokens=True
            )
        )
        start_token, sentence, end_token = sentence[0], sentence[1:-1], sentence[-1]
        sentence = sentence[: self.max_seq_length]

        if self.multi_label:
            label = torch.zeros(self.n_classes)
            label[
                [self.labels.index(tgt) for tgt in self.data[index][self.labels_label]]
            ] = 1
        else:
            label = torch.tensor(self.labels.index(self.data[index][self.labels_label]))

        image = Image.open(
            os.path.join(self.data_dir, self.data[index]["images"])
        ).convert("RGB")
        image = self.transforms(image)

        return {
            "image_start_token": start_token,
            "image_end_token": end_token,
            "sentence": sentence,
            "image": image,
            "label": label,
        }

    def get_label_frequencies(self):
        label_freqs = Counter()
        for row in self.data:
            label_freqs.update(row[self.labels_label])
        return label_freqs


def collate_fn(batch):
    lens = [len(row["sentence"]) for row in batch]
    bsz, max_seq_len = len(batch), max(lens)

    mask_tensor = torch.zeros(bsz, max_seq_len, dtype=torch.long)
    text_tensor = torch.zeros(bsz, max_seq_len, dtype=torch.long)

    for i_batch, (input_row, length) in enumerate(zip(batch, lens)):
        text_tensor[i_batch, :length] = input_row["sentence"]
        mask_tensor[i_batch, :length] = 1

    img_tensor = torch.stack([row["image"] for row in batch])
    tgt_tensor = torch.stack([row["label"] for row in batch])
    img_start_token = torch.stack([row["image_start_token"] for row in batch])
    img_end_token = torch.stack([row["image_end_token"] for row in batch])

    return (
        text_tensor,
        mask_tensor,
        img_tensor,
        img_start_token,
        img_end_token,
        tgt_tensor,
    )


def get_image_transforms():
    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.46777044, 0.44531429, 0.40661017],
                std=[0.12221994, 0.12145835, 0.14380469],
            ),
        ]
    )


class LazyClassificationDataset(Dataset):
    def __init__(self, data_file, tokenizer, args):
        self.data_file = data_file
        self.start_row = args.lazy_loading_start_line
        self.num_entries = self._get_n_lines(self.data_file, self.start_row)
        self.tokenizer = tokenizer
        self.args = args
        self.delimiter = args.lazy_delimiter
        if args.lazy_text_a_column is not None and args.lazy_text_b_column is not None:
            self.text_a_column = args.lazy_text_a_column
            self.text_b_column = args.lazy_text_b_column
            self.text_column = None
        else:
            self.text_column = args.lazy_text_column
            self.text_a_column = None
            self.text_b_column = None
        self.labels_column = args.lazy_labels_column

    @staticmethod
    def _get_n_lines(data_file, start_row):
        with open(data_file, encoding="utf-8") as f:
            for line_idx, _ in enumerate(f, 1):
                pass

        return line_idx - start_row

    def __getitem__(self, idx):
        line = (
            linecache.getline(self.data_file, idx + 1 + self.start_row)
            .rstrip("\n")
            .split(self.delimiter)
        )

        if not self.text_a_column and not self.text_b_column:
            text = line[self.text_column]
            label = line[self.labels_column]

            # If labels_map is defined, then labels need to be replaced with ints
            if self.args.labels_map:
                label = self.args.labels_map[label]
            if self.args.regression:
                label = torch.tensor(float(label), dtype=torch.float)
            else:
                label = torch.tensor(int(label), dtype=torch.long)

            return (
                self.tokenizer.encode_plus(
                    text,
                    max_length=self.args.max_seq_length,
                    pad_to_max_length=self.args.max_seq_length,
                    return_tensors="pt",
                ),
                label,
            )
        else:
            text_a = line[self.text_a_column]
            text_b = line[self.text_b_column]
            label = line[self.labels_column]
            if self.args.regression:
                label = torch.tensor(float(label), dtype=torch.float)
            else:
                label = torch.tensor(int(label), dtype=torch.long)

            return (
                self.tokenizer.encode_plus(
                    text_a,
                    text_pair=text_b,
                    max_length=self.args.max_seq_length,
                    pad_to_max_length=self.args.max_seq_length,
                    return_tensors="pt",
                ),
                label,
            )

    def __len__(self):
        return self.num_entries


def flatten_results(results, parent_key="", sep="/"):
    out = []
    if isinstance(results, Mapping):
        for key, value in results.items():
            pkey = parent_key + sep + str(key) if parent_key else str(key)
            out.extend(flatten_results(value, parent_key=pkey).items())
    elif isinstance(results, Iterable):
        for key, value in enumerate(results):
            pkey = parent_key + sep + str(key) if parent_key else str(key)
            out.extend(flatten_results(value, parent_key=pkey).items())
    else:
        out.append((parent_key, results))
    return dict(out)


def convert_beir_to_cross_encoder_format(
    data,
    run_dict=None,
    top_k=None,
    include_titles=False,
    save_path=None,
    bm25_format=False,
):
    """
    Utility function to convert BEIR format to cross-encoder format

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
        include_title: Whether to include the title of the document in the cross-encoder format.
        save_path: Path to save the converted dataset. If not provided, the dataset is returned as a DataFrame.
    """
    if run_dict:
        if bm25_format:
            bm_df = pd.read_csv(run_dict, sep=" ", header=None)
            bm_df.columns = ["qid", "Q0", "pid", "rank", "score", "runstring"]
            bm_df = bm_df[["qid", "pid", "rank", "score"]]

            run_dict = {}
            for qid, group in bm_df.groupby("qid"):
                if top_k:
                    run_dict[str(qid)] = {
                        str(row[2]): row[4] for row in group[:top_k].itertuples()
                    }
                else:
                    run_dict[str(qid)] = {
                        str(row[2]): row[4] for row in group.itertuples()
                    }
        else:
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

    if include_titles:
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
                        "passage_id": passage_id,
                        "text_a": queries_df.loc[query_id]["text"],
                        "text_b": corpus_df.loc[passage_id]["text"],
                    }
                )
    else:
        reranking_data = []
        for query_id, query in tqdm(queries_df.iterrows(), total=len(queries_df)):
            for passage_id, passage in corpus_df.iterrows():
                reranking_data.append(
                    {
                        "query_id": query_id,
                        "passage_id": passage_id,
                        "text_a": query["text"],
                        "text_b": passage["text"],
                    }
                )

    reranking_df = pd.DataFrame(reranking_data)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        reranking_df.to_csv(save_path, sep="\t", index=False)

        save_dir = os.path.dirname(save_path)
        run_dict_path = os.path.join(save_dir, "run_dict.json")
        with open(run_dict_path, "w") as f:
            json.dump(run_dict, f)

        # If the BEIR dir contains a qrels directory, copy it to the save_dir
        qrels_dir = os.path.join(data, "qrels")
        if os.path.exists(qrels_dir):
            shutil.copytree(
                qrels_dir, os.path.join(save_dir, "qrels"), dirs_exist_ok=True
            )

    return reranking_df
