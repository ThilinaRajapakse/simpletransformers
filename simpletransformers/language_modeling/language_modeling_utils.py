import os
import pickle
import logging
from typing import Tuple
from multiprocessing import Pool

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer
from tokenizers.implementations import ByteLevelBPETokenizer
from tokenizers.processors import BertProcessing
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)


class TextDataset(Dataset):
    def __init__(self, tokenizer: PreTrainedTokenizer, args, file_path: str, mode, block_size=512):
        assert os.path.isfile(file_path)

        block_size = block_size - (tokenizer.max_len - tokenizer.max_len_single_sentence)

        directory, filename = os.path.split(file_path)
        cached_features_file = os.path.join(
            args["cache_dir"], args["model_type"] + "_cached_lm_" + str(block_size) + "_" + filename
        )

        if os.path.exists(cached_features_file) and (
            (not args["reprocess_input_data"] and not args["no_cache"])
            or (mode == "dev" and args["use_cached_eval_features"] and not args["no_cache"])
        ):
            logger.info(" Loading features from cached file %s", cached_features_file)
            with open(cached_features_file, "rb") as handle:
                self.examples = pickle.load(handle)
        else:
            logger.info(" Creating features from dataset file at %s", args["cache_dir"])

            self.examples = []
            with open(file_path, encoding="utf-8") as f:
                text = f.read()

            # tokenizer = ByteLevelBPETokenizer(
            #     "outputs/vocab.json",
            #     "outputs/merges.txt",
            # )
            # tokenizer._tokenizer.post_processor = BertProcessing(
            #     ("</s>", tokenizer.token_to_id("</s>")),
            #     ("<s>", tokenizer.token_to_id("<s>")),
            # )

            # logger.info(" Encoding")
            # tokenized_text = tokenizer.encode(text).ids
            # logger.info(" Encoded")
            # self.examples = [tokenized_text[i : i + block_size] for i in tqdm(range(0, len(tokenized_text) - block_size + 1, block_size))]

            tokenized_text = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text))
            tokenized_text_split = [tokenized_text[i : i + block_size] for i in tqdm(range(0, len(tokenized_text) - block_size + 1, block_size))]

            with Pool(args["process_count"]) as p:
                self.examples = list(
                    tqdm(
                        p.imap(tokenizer.build_inputs_with_special_tokens, tokenized_text_split, chunksize=500),
                        total=len(tokenized_text_split),
                        # disable=silent,
                    )
                )


            # for i in range(0, len(tokenized_text) - block_size + 1, block_size):  # Truncate in block of block_size
            #     self.examples.append(tokenizer.build_inputs_with_special_tokens(tokenized_text[i : i + block_size]))
            # Note that we are loosing the last truncated example here for the sake of simplicity (no padding)
            # If your dataset is small, first you should loook for a bigger one :-) and second you
            # can change this behavior by adding (model specific) padding.

            logger.info(" Saving features into cached file %s", cached_features_file)
            with open(cached_features_file, "wb") as handle:
                pickle.dump(self.examples, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return torch.tensor(self.examples[item], dtype=torch.long)


class LineByLineTextDataset(Dataset):
    def __init__(self, tokenizer: PreTrainedTokenizer, args, file_path: str, block_size=512):
        assert os.path.isfile(file_path)
        # Here, we do not cache the features, operating under the assumption
        # that we will soon use fast multithreaded tokenizers from the
        # `tokenizers` repo everywhere =)
        logger.info(" Creating features from dataset file at %s", file_path)

        with open(file_path, encoding="utf-8") as f:
            lines = [line for line in f.read().splitlines() if (len(line) > 0 and not line.isspace())]

        tokenizer = ByteLevelBPETokenizer(
            "outputs/vocab.json",
            "outputs/merges.txt",
        )
        tokenizer._tokenizer.post_processor = BertProcessing(
            ("</s>", tokenizer.token_to_id("</s>")),
            ("<s>", tokenizer.token_to_id("<s>")),
        )

        tokenizer.enable_truncation(max_length=block_size)
        self.examples = [t.ids for t in tokenizer.encode_batch(lines)]

        # self.examples = tokenizer.batch_encode_plus(lines, add_special_tokens=True, max_length=block_size)["input_ids"]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return torch.tensor(self.examples[i], dtype=torch.long)


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
    probability_matrix = torch.full(labels.shape, args["mlm_probability"])
    special_tokens_mask = [
        tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
    ]
    probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
    if tokenizer._pad_token is not None:
        padding_mask = labels.eq(tokenizer.pad_token_id)
        probability_matrix.masked_fill_(padding_mask, value=0.0)
    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels[~masked_indices] = -100  # We only compute loss on masked tokens

    # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
    inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

    # 10% of the time, we replace masked input tokens with random word
    indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
    random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long)
    inputs[indices_random] = random_words[indices_random]

    # The rest of the time (10% of the time) we keep the masked input tokens unchanged
    return inputs, labels
