import json
import logging
import math
import os
import random
import warnings
from multiprocessing import cpu_count
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm, trange

import pandas as pd
import torch
from simpletransformers.config.global_args import global_args
from simpletransformers.t5.t5_utils import T5Dataset
from tensorboardX import SummaryWriter
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from transformers import AdamW, EncoderDecoderModel, EncoderDecoderConfig, get_linear_schedule_with_warmup

try:
    import wandb

    wandb_available = True
except ImportError:
    wandb_available = False

logger = logging.getLogger(__name__)


class Seq2SeqModel:
    def __init__(
        self, encoder_name=None, decoder_name=None, config=None, args=None, use_cuda=True, cuda_device=-1, **kwargs
    ):

        if not config:
            if not (encoder_name and decoder_name):
                raise ValueError(
                    "You must specify either a Seq2Seq config or " "you must specify encoder_name and decoder_name"
                )

        if args and "manual_seed" in args:
            random.seed(args["manual_seed"])
            np.random.seed(args["manual_seed"])
            torch.manual_seed(args["manual_seed"])
            if "n_gpu" in args and args["n_gpu"] > 0:
                torch.cuda.manual_seed_all(args["manual_seed"])

        self.args = {
            "dataset_class": None,
            "do_sample": False,
            "max_steps": -1,
            "evaluate_generated_text": False,
            "num_beams": 1,
            "max_length": 20,
            "repetition_penalty": 1.0,
            "length_penalty": 2.0,
            "early_stopping": True,
            "preprocess_inputs": True,
        }

        self.args.update(global_args)

        if args:
            self.args.update(args)

        if use_cuda:
            if torch.cuda.is_available():
                if cuda_device == -1:
                    self.device = torch.device("cuda")
                else:
                    self.device = torch.device(f"cuda:{cuda_device}")
            else:
                raise ValueError(
                    "'use_cuda' set to True when cuda is unavailable."
                    "Make sure CUDA is available or set `use_cuda=False`."
                )
        else:
            self.device = "cpu"

        self.results = {}

        self.model = EncoderDecoderModel.from_encoder_decoder_pretrained(
            config=config, encoder=encoder_name, decoder=decoder_name
        )
