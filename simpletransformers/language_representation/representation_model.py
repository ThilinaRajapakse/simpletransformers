#!/usr/bin/env python
# coding: utf-8


from __future__ import absolute_import, division, print_function

import logging
import random
import warnings
from functools import partial

import numpy as np
import torch
from tqdm.auto import tqdm
from transformers import (
    BertConfig,
    BertTokenizer,
    GPT2Config,
    GPT2Tokenizer,
    RobertaConfig,
    RobertaTokenizer,
)

from simpletransformers.config.model_args import ModelArgs
from simpletransformers.config.utils import sweep_config_to_sweep_values
from simpletransformers.language_representation.transformer_models.bert_model import (
    BertForTextRepresentation,
)
from simpletransformers.language_representation.transformer_models.gpt2_model import (
    GPT2ForTextRepresentation,
)

try:
    import wandb

    wandb_available = True
except ImportError:
    wandb_available = False

logger = logging.getLogger(__name__)


def mean_across_all_tokens(token_vectors):
    return torch.mean(token_vectors, dim=1)


def concat_all_tokens(token_vectors):
    batch_size, max_tokens, emb_dim = token_vectors.shape
    return torch.reshape(token_vectors, (batch_size, max_tokens * emb_dim))


def select_a_token(token_vectors, token_index):
    return token_vectors[:, token_index, :]


def get_all_tokens(token_vectors):
    return token_vectors


def batch_iterable(iterable, batch_size=1):
    l = len(iterable)
    for ndx in range(0, l, batch_size):
        yield iterable[ndx : min(ndx + batch_size, l)]


class RepresentationModel:
    def __init__(
        self,
        model_type,
        model_name,
        args=None,
        use_cuda=True,
        cuda_device=-1,
        **kwargs,
    ):

        """
        Initializes a RepresentationModel model.

        Args:
            model_type: The type of model (bert, roberta, gpt2)
            model_name: The exact architecture and trained weights to use. This may be a Hugging Face Transformers compatible pre-trained model, a community model, or the path to a directory containing model files.
            args (optional): Default args will be used if this parameter is not provided. If provided, it should be a dict containing the args that should be changed in the default args.
            use_cuda (optional): Use GPU if available. Setting to False will force model to use CPU only.
            cuda_device (optional): Specific GPU that should be used. Will use the first available GPU by default.
            **kwargs (optional): For providing proxies, force_download, resume_download, cache_dir and other options specific to the 'from_pretrained' implementation where this will be supplied.
        """  # noqa: ignore flake8"

        MODEL_CLASSES = {
            "bert": (BertConfig, BertForTextRepresentation, BertTokenizer),
            "roberta": (RobertaConfig, BertForTextRepresentation, RobertaTokenizer),
            "gpt2": (GPT2Config, GPT2ForTextRepresentation, GPT2Tokenizer),
        }

        self.args = self._load_model_args(model_name)

        if isinstance(args, dict):
            self.args.update_from_dict(args)
        elif isinstance(args, ModelArgs):
            self.args = args

        if "sweep_config" in kwargs:
            self.is_sweeping = True
            sweep_config = kwargs.pop("sweep_config")
            sweep_values = sweep_config_to_sweep_values(sweep_config)
            self.args.update_from_dict(sweep_values)
        else:
            self.is_sweeping = False

        if self.args.manual_seed:
            random.seed(self.args.manual_seed)
            np.random.seed(self.args.manual_seed)
            torch.manual_seed(self.args.manual_seed)
            if self.args.n_gpu > 0:
                torch.cuda.manual_seed_all(self.args.manual_seed)

        config_class, model_class, tokenizer_class = MODEL_CLASSES[model_type]
        self.config = config_class.from_pretrained(model_name, **self.args.config)
        if use_cuda:
            if torch.cuda.is_available():
                if cuda_device == -1:
                    self.device = torch.device("cuda")
                else:
                    self.device = torch.device(f"cuda:{cuda_device}")
            else:
                raise ValueError(
                    "'use_cuda' set to True when cuda is unavailable."
                    " Make sure CUDA is available or set use_cuda=False."
                )
        else:
            self.device = "cpu"

        self.model = model_class.from_pretrained(
            model_name, config=self.config, **kwargs
        )

        self.results = {}

        if not use_cuda:
            self.args.fp16 = False

        self.tokenizer = tokenizer_class.from_pretrained(
            model_name, do_lower_case=self.args.do_lower_case, **kwargs
        )

        self.args.model_name = model_name
        self.args.model_type = model_type

        if self.args.wandb_project and not wandb_available:
            warnings.warn(
                "wandb_project specified but wandb is not available. Wandb disabled."
            )
            self.args.wandb_project = None
        if self.args.model_type == "gpt2":
            # should we add a custom tokenizer for this model?
            self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})
            self.model.resize_token_embeddings(len(self.tokenizer))

    def _tokenize(self, text_list):
        # Tokenize the text with the provided tokenizer
        encoded = self.tokenizer.batch_encode_plus(
            text_list,
            add_special_tokens=True,
            max_length=self.args.max_seq_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        return encoded

    def encode_sentences(self, text_list, combine_strategy=None, batch_size=32):
        """
        Generates list of contextual word or sentence embeddings using the model passed to class constructor
        :param text_list: list of text sentences
        :param combine_strategy: strategy for combining word vectors, supported values: None, "mean", "concat",
        or an int value to select a specific embedding (e.g. 0 for [CLS] or -1 for the last one)
        :param batch_size
        :return: list of lists of sentence embeddings (if `combine_strategy=None`) OR list of sentence
        embeddings (if `combine_strategy!=None`)
        """

        if combine_strategy is not None:
            if type(combine_strategy) == int:
                embedding_func = partial(select_a_token, token_index=combine_strategy)
            else:
                embedding_func_mapping = {
                    "mean": mean_across_all_tokens,
                    "concat": concat_all_tokens,
                }
                try:
                    embedding_func = embedding_func_mapping[combine_strategy]
                except KeyError:
                    raise ValueError(
                        "Provided combine_strategy is not valid."
                        "supported values are: 'concat', 'mean' and None."
                    )
        else:
            embedding_func = get_all_tokens

        self.model.to(self.device)
        self.model.eval()
        batches = batch_iterable(text_list, batch_size=batch_size)
        embeddings = list()
        for batch in batches:
            encoded = self._tokenize(batch)
            with torch.no_grad():
                if self.args.model_type not in ["roberta", "gpt2"]:
                    token_vectors = self.model(
                        input_ids=encoded["input_ids"].to(self.device),
                        attention_mask=encoded["attention_mask"].to(self.device),
                        token_type_ids=encoded["token_type_ids"].to(self.device),
                    )
                else:
                    token_vectors = self.model(
                        input_ids=encoded["input_ids"].to(self.device),
                        attention_mask=encoded["attention_mask"].to(self.device),
                    )
            embeddings.append(embedding_func(token_vectors).cpu().detach().numpy())
        embeddings = np.concatenate(embeddings, axis=0)

        return embeddings

    def _load_model_args(self, input_dir):
        args = ModelArgs()
        args.load(input_dir)
        return args
