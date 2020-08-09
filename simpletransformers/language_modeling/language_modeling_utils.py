import logging
import os
import pickle
from multiprocessing import Pool
from typing import Tuple

import torch
from tokenizers.processors import BertProcessing
from torch.utils.data import Dataset
from tqdm.auto import tqdm
from transformers import PreTrainedTokenizer

logger = logging.getLogger(__name__)


def encode(data):
    tokenizer, line = data
    return tokenizer.encode(line)


def encode_sliding_window(data):
    tokenizer, line, max_seq_length, special_tokens_count, stride, no_padding = data

    tokens = tokenizer.tokenize(line)
    stride = int(max_seq_length * stride)
    token_sets = []
    if len(tokens) > max_seq_length - special_tokens_count:
        token_sets = [tokens[i : i + max_seq_length - special_tokens_count] for i in range(0, len(tokens), stride)]
    else:
        token_sets.append(tokens)

    features = []
    if not no_padding:
        sep_token = tokenizer.sep_token_id
        cls_token = tokenizer.cls_token_id
        pad_token = tokenizer.pad_token_id

        for tokens in token_sets:
            tokens = [cls_token] + tokens + [sep_token]

            input_ids = tokenizer.convert_tokens_to_ids(tokens)
            padding_length = max_seq_length - len(input_ids)
            input_ids = input_ids + ([pad_token] * padding_length)

            assert len(input_ids) == max_seq_length

            features.append(input_ids)
    else:
        for tokens in token_sets:
            input_ids = tokenizer.convert_tokens_to_ids(tokens)
            features.append(input_ids)

    return features


class SimpleDataset(Dataset):
    def __init__(self, tokenizer, args, file_path, mode, block_size=512, special_tokens_count=2, sliding_window=False):
        assert os.path.isfile(file_path)
        block_size = block_size - special_tokens_count
        directory, filename = os.path.split(file_path)
        cached_features_file = os.path.join(
            args.cache_dir, args.model_type + "_cached_lm_" + str(block_size) + "_" + filename
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

            if sliding_window:
                no_padding = True if args.model_type in ["gpt2", "openai-gpt"] else False
                with open(file_path, encoding="utf-8") as f:
                    lines = [
                        (tokenizer, line, args.max_seq_length, special_tokens_count, args.stride, no_padding)
                        for line in f.read().splitlines()
                        if (len(line) > 0 and not line.isspace())
                    ]

                if args.use_multiprocessing:
                    with Pool(args.process_count) as p:
                        self.examples = list(
                            tqdm(
                                p.imap(encode_sliding_window, lines, chunksize=args.multiprocessing_chunksize),
                                total=len(lines),
                                # disable=silent,
                            )
                        )
                else:
                    self.examples = [encode_sliding_window(line) for line in lines]

                self.examples = [example for example_set in self.examples for example in example_set]
            else:
                with open(file_path, encoding="utf-8") as f:
                    lines = [
                        (tokenizer, line) for line in f.read().splitlines() if (len(line) > 0 and not line.isspace())
                    ]

                if args.use_multiprocessing:
                    with Pool(args.process_count) as p:
                        self.examples = list(
                            tqdm(
                                p.imap(encode, lines, chunksize=args.multiprocessing_chunksize),
                                total=len(lines),
                                # disable=silent,
                            )
                        )
                else:
                    self.examples = [encode(line) for line in lines]

                self.examples = [token for tokens in self.examples for token in tokens]
                if len(self.examples) > block_size:
                    self.examples = [
                        tokenizer.build_inputs_with_special_tokens(self.examples[i : i + block_size])
                        for i in tqdm(range(0, len(self.examples) - block_size + 1, block_size))
                    ]
                else:
                    self.examples = [tokenizer.build_inputs_with_special_tokens(self.examples)]

            logger.info(" Saving features into cached file %s", cached_features_file)
            with open(cached_features_file, "wb") as handle:
                pickle.dump(self.examples, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return torch.tensor(self.examples[item], dtype=torch.long)


def mask_tokens(inputs: torch.Tensor, tokenizer: PreTrainedTokenizer, args) -> Tuple[torch.Tensor, torch.Tensor]:
    """ Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original. """

    if tokenizer.mask_token is None:
        raise ValueError(
            "This tokenizer does not have a mask token which is necessary for masked language modeling."
            "Set 'mlm' to False in args if you want to use this tokenizer."
        )

    labels = inputs.clone()
    # We sample a few tokens in each sequence for masked-LM training
    # (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)
    probability_matrix = torch.full(labels.shape, args.mlm_probability)
    special_tokens_mask = [
        tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
    ]
    probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
    if tokenizer._pad_token is not None:
        padding_mask = labels.eq(tokenizer.pad_token_id)
        probability_matrix.masked_fill_(padding_mask, value=0.0)
    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels[~masked_indices] = -100  # We only compute loss on masked tokens

    if args.model_type == "electra":
        # For ELECTRA, we replace all masked input tokens with tokenizer.mask_token
        inputs[masked_indices] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)
    else:
        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
    return inputs, labels
