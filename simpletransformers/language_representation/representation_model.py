#!/usr/bin/env python
# coding: utf-8


from __future__ import absolute_import, division, print_function

import logging
import random
import warnings

import numpy as np
import torch
from transformers import BertConfig, BertTokenizer, GPT2Config, GPT2Tokenizer, RobertaConfig, RobertaTokenizer

from simpletransformers.config.model_args import ModelArgs
from simpletransformers.language_representation.transformer_models.bert_model import BertForTextRepresentation
from simpletransformers.language_representation.transformer_models.gpt2_model import GPT2ForTextRepresentation

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


def batch_iterable(iterable, batch_size=1):
    l = len(iterable)
    for ndx in range(0, l, batch_size):
        yield iterable[ndx : min(ndx + batch_size, l)]


class RepresentationModel:
    def __init__(
        self, model_type, model_name, args=None, use_cuda=True, cuda_device=-1, **kwargs,
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
            sweep_config = kwargs.pop("sweep_config")
            sweep_values = {key: value["value"] for key, value in sweep_config.as_dict().items() if key != "_wandb"}
            self.args.update_from_dict(sweep_values)

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

        self.model = model_class.from_pretrained(model_name, config=self.config, **kwargs)

        self.results = {}

        if not use_cuda:
            self.args.fp16 = False

        self.tokenizer = tokenizer_class.from_pretrained(model_name, do_lower_case=self.args.do_lower_case, **kwargs)

        self.args.model_name = model_name
        self.args.model_type = model_type

        if self.args.wandb_project and not wandb_available:
            warnings.warn("wandb_project specified but wandb is not available. Wandb disabled.")
            self.args.wandb_project = None
        if self.args.model_type == "gpt2":
            # should we add a custom tokenizer for this model?
            self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})
            self.model.resize_token_embeddings(len(self.tokenizer))

    def _tokenize(self, text_list):
        # Tokenize the text with the provided tokenizer
        input_ids = self.tokenizer.batch_encode_plus(
            text_list, add_special_tokens=True, max_length=self.args.max_seq_length, padding=True, truncation=True
        )["input_ids"]
        return torch.LongTensor(input_ids)

    def encode_sentences(self, text_list, combine_strategy=None, batch_size=32):
        """

        Generates list of contextual word or sentence embeddings using the model passed to class constructor
        :param text_list: list of text sentences
        :param combine_strategy: strategy for combining word vectors, supported values: None, "mean", "concat"
        :param batch_size
        :return: list of lists of sentence embeddings(if `combine_strategy=None`) OR list of sentence embeddings(if `combine_strategy!=None`)
        """
        batches = batch_iterable(text_list, batch_size=batch_size)
        embeddings = np.array([])
        for batch in batches:
            input_ids_tensor = self._tokenize(batch)
            token_vectors = self.model(input_ids=input_ids_tensor)
            if combine_strategy:
                embedding_func_mapping = {"mean": mean_across_all_tokens, "concat": concat_all_tokens}
                if embedding_func_mapping[combine_strategy]:
                    embedding_func = embedding_func_mapping[combine_strategy]
                else:
                    raise ValueError(
                        "Provided combine_strategy is not valid." "supported values are: 'concat', 'mean' and None."
                    )
                batch_embeddings = embedding_func(token_vectors).detach().numpy()
            else:
                batch_embeddings = token_vectors.detach().numpy()
            if len(embeddings) == 0:
                embeddings = batch_embeddings
            else:
                embeddings = np.concatenate((embeddings, batch_embeddings), axis=0)

        return embeddings

    def _load_model_args(self, input_dir):
        args = ModelArgs()
        args.load(input_dir)
        return args
