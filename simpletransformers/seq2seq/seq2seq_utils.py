import logging
import os
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


class BertSumOptimizer(object):
    """ Specific optimizer for BertSum.
    As described in [1], the authors fine-tune BertSum for abstractive
    summarization using two Adam Optimizers with different warm-up steps and
    learning rate. They also use a custom learning rate scheduler.
    [1] Liu, Yang, and Mirella Lapata. "Text summarization with pretrained encoders."
        arXiv preprint arXiv:1908.08345 (2019).
    """

    def __init__(self, model, lr, warmup_steps, beta_1=0.99, beta_2=0.999, eps=1e-8):
        self.encoder = model.encoder
        self.decoder = model.decoder
        self.lr = lr
        self.warmup_steps = warmup_steps

        self.optimizers = {
            "encoder": torch.optim.Adam(
                model.encoder.parameters(), lr=lr["encoder"], betas=(beta_1, beta_2), eps=eps,
            ),
            "decoder": torch.optim.Adam(
                model.decoder.parameters(), lr=lr["decoder"], betas=(beta_1, beta_2), eps=eps,
            ),
        }

        self._step = 0
        self.current_learning_rates = {}

    def _update_rate(self, stack):
        return self.lr[stack] * min(self._step ** (-0.5), self._step * self.warmup_steps[stack] ** (-1.5))

    def zero_grad(self):
        self.optimizer_decoder.zero_grad()
        self.optimizer_encoder.zero_grad()

    def step(self):
        self._step += 1
        for stack, optimizer in self.optimizers.items():
            new_rate = self._update_rate(stack)
            for param_group in optimizer.param_groups:
                param_group["lr"] = new_rate
            optimizer.step()
            self.current_learning_rates[stack] = new_rate


def preprocess_data(data):
    input_text, target_text, encoder_tokenizer, decoder_tokenizer, args = data

    # Add EOS again if truncated?
    if args["preprocess_inputs"]:
        input_text = encoder_tokenizer.encode(
            input_text, max_length=args["max_seq_length"], pad_to_max_length=True, return_tensors="pt",
        )

        target_text = decoder_tokenizer.encode(
            target_text, max_length=args["max_seq_length"], pad_to_max_length=True, return_tensors="pt"
        )
    else:
        input_text = encoder_tokenizer.encode(
            input_text, max_length=args["max_seq_length"], pad_to_max_length=True, return_tensors="pt",
        )

        target_text = decoder_tokenizer.encode(
            target_text, max_length=args["max_seq_length"], pad_to_max_length=True, return_tensors="pt"
        )
    return (torch.flatten(input_text), torch.flatten(target_text))


class Seq2SeqDataset(Dataset):
    def __init__(self, encoder_tokenizer, decoder_tokenizer, args, data, mode):
        cached_features_file = os.path.join(
            args["cache_dir"], args["model_name"] + "_cached_" + str(args["max_seq_length"]) + str(len(data))
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

        data = [
            (input_text, target_text, encoder_tokenizer, decoder_tokenizer, args)
            for input_text, target_text in zip(data["input_text"], data["target_text"])
        ]

        if args["use_multiprocessing"]:
            with Pool(args["process_count"]) as p:
                self.examples = list(
                    tqdm(p.imap(preprocess_data, data, chunksize=500), total=len(data), disable=args["silent"],)
                )
        else:
            self.examples = [preprocess_data(d) for d in tqdm(data, disable=args["silent"])]

        logger.info(" Saving features into cached file %s", cached_features_file)
        with open(cached_features_file, "wb") as handle:
            pickle.dump(self.examples, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index]
