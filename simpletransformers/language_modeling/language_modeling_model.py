#!/usr/bin/env python
# coding: utf-8


from __future__ import absolute_import, division, print_function

import json
import logging
import math
import os
import random
import warnings
from dataclasses import asdict
from multiprocessing import cpu_count
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    confusion_matrix,
    label_ranking_average_precision_score,
    matthews_corrcoef,
    mean_squared_error,
)
from torch.utils.tensorboard import SummaryWriter
from tokenizers import BertWordPieceTokenizer, ByteLevelBPETokenizer
from tokenizers.implementations import (
    SentencePieceBPETokenizer,
    SentencePieceUnigramTokenizer,
    # BertWordPieceTokenizer,
    # ByteLevelBPETokenizer,
)
from tokenizers.processors import BertProcessing
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm.auto import tqdm, trange
from transformers.optimization import (
    get_constant_schedule,
    get_constant_schedule_with_warmup,
    get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    get_cosine_with_hard_restarts_schedule_with_warmup,
    get_polynomial_decay_schedule_with_warmup,
)
from torch.optim import AdamW
from transformers.optimization import Adafactor
from simpletransformers.custom_models.models import RobertaWithAutoEncoderForMaskedLM

from transformers import DummyObject, requires_backends
from torch.utils.data import DataLoader, TensorDataset


class NystromformerTokenizer(metaclass=DummyObject):
    _backends = ["sentencepiece"]

    def __init__(self, *args, **kwargs):
        requires_backends(self, ["sentencepiece"])


from transformers import (
    WEIGHTS_NAME,
    AutoConfig,
    AutoModelWithLMHead,
    AutoTokenizer,
    AutoModelForCausalLM,
    BertConfig,
    BertForMaskedLM,
    BertTokenizer,
    BigBirdConfig,
    BigBirdForMaskedLM,
    BigBirdTokenizer,
    CamembertConfig,
    CamembertForMaskedLM,
    CamembertTokenizer,
    DistilBertConfig,
    DistilBertForMaskedLM,
    DistilBertTokenizer,
    ElectraConfig,
    ElectraForMaskedLM,
    ElectraForPreTraining,
    ElectraTokenizer,
    GPT2Config,
    GPT2LMHeadModel,
    GPT2Tokenizer,
    LongformerConfig,
    LongformerForMaskedLM,
    LongformerTokenizer,
    NystromformerConfig,
    NystromformerForMaskedLM,
    # NystromformerTokenizer,
    OpenAIGPTConfig,
    OpenAIGPTLMHeadModel,
    OpenAIGPTTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    RemBertConfig,
    RemBertForMaskedLM,
    RemBertTokenizer,
    RobertaConfig,
    RobertaForMaskedLM,
    RobertaTokenizer,
    XLMRobertaConfig,
    XLMRobertaForMaskedLM,
    XLMRobertaTokenizer,
    GenerationConfig,
)
from transformers.data.datasets.language_modeling import (
    LineByLineTextDataset,
    TextDataset,
)

from simpletransformers.config.global_args import global_args
from simpletransformers.config.model_args import LanguageModelingArgs, GenerationArgs
from simpletransformers.config.utils import sweep_config_to_sweep_values
from simpletransformers.custom_models.models import ElectraForLanguageModelingModel
from simpletransformers.language_modeling.language_modeling_utils import (
    SimpleDataset,
    apply_chat_template_to_inputs,
    load_hf_dataset,
    mask_tokens,
)

try:
    import wandb

    wandb_available = True
except ImportError:
    wandb_available = False

logger = logging.getLogger(__name__)

MODEL_CLASSES = {
    "auto": (AutoConfig, AutoModelWithLMHead, AutoTokenizer),
    "bert": (BertConfig, BertForMaskedLM, BertTokenizer),
    "bigbird": (BigBirdConfig, BigBirdForMaskedLM, BigBirdTokenizer),
    "camembert": (CamembertConfig, CamembertForMaskedLM, CamembertTokenizer),
    "causal": (AutoConfig, AutoModelForCausalLM, AutoTokenizer),
    "distilbert": (DistilBertConfig, DistilBertForMaskedLM, DistilBertTokenizer),
    "electra": (ElectraConfig, ElectraForLanguageModelingModel, ElectraTokenizer),
    "gpt2": (GPT2Config, GPT2LMHeadModel, GPT2Tokenizer),
    "longformer": (LongformerConfig, LongformerForMaskedLM, LongformerTokenizer),
    "nystromformer": (NystromformerConfig, NystromformerForMaskedLM, BigBirdTokenizer),
    "openai-gpt": (OpenAIGPTConfig, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer),
    "rembert": (RemBertConfig, RemBertForMaskedLM, RemBertTokenizer),
    "roberta": (RobertaConfig, RobertaForMaskedLM, RobertaTokenizer),
    "roberta-autoencoder": (
        RobertaConfig,
        RobertaWithAutoEncoderForMaskedLM,
        RobertaTokenizer,
    ),
    "xlmroberta": (XLMRobertaConfig, XLMRobertaForMaskedLM, XLMRobertaTokenizer),
}


class LanguageModelingModel:
    def __init__(
        self,
        model_type,
        model_name,
        generator_name=None,
        discriminator_name=None,
        train_files=None,
        args=None,
        use_cuda=True,
        retrieval_model=None,
        adapter_name=None,
        # autoencoder_model=None,
        cuda_device=-1,
        **kwargs,
    ):
        """
        Initializes a LanguageModelingModel.

        Args:
            model_type: The type of model (gpt2, openai-gpt, bert, roberta, distilbert, camembert)
            model_name: Default Transformer model name or path to a directory containing Transformer model file (pytorch_nodel.bin).
            generator_name (optional): A pretrained model name or path to a directory containing an ELECTRA generator model.
            discriminator_name (optional): A pretrained model name or path to a directory containing an ELECTRA discriminator model.
            args (optional): Default args will be used if this parameter is not provided. If provided, it should be a dict containing the args that should be changed in the default args.
            train_files (optional): List of files to be used when training the tokenizer.
            rag_corpus (optional): A collection of documents to be used for Retrieval-Augmented Generation. This may
            retrieval_model (optional): A pretrained model name or path to a directory containing a retrieval model. This should be preloaded with a knowledge index.
            use_cuda (optional): Use GPU if available. Setting to False will force model to use CPU only.
            cuda_device (optional): Specific GPU that should be used. Will use the first available GPU by default.
            **kwargs (optional): For providing proxies, force_download, resume_download, cache_dir and other options specific to the 'from_pretrained' implementation where this will be supplied.
        """  # noqa: ignore flake8"

        self.args = self._load_model_args(model_name)

        if isinstance(args, dict):
            self.args.update_from_dict(args)
        elif isinstance(args, LanguageModelingArgs):
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

        if self.args.local_rank != -1:
            logger.info(f"local_rank: {self.args.local_rank}")
            torch.distributed.init_process_group(backend="nccl")
            cuda_device = self.args.local_rank

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

        self.results = {}

        if not use_cuda:
            self.args.fp16 = False

        self.args.model_name = model_name
        self.args.model_type = model_type

        config_class, model_class, tokenizer_class = MODEL_CLASSES[model_type]
        self.tokenizer_class = tokenizer_class
        new_tokenizer = False

        if self.args.tokenizer_name:
            self.tokenizer = tokenizer_class.from_pretrained(
                self.args.tokenizer_name, cache_dir=self.args.cache_dir
            )
        elif self.args.model_name:
            if self.args.model_name == "electra":
                self.tokenizer = tokenizer_class.from_pretrained(
                    generator_name, cache_dir=self.args.cache_dir, **kwargs
                )
                self.args.tokenizer_name = self.args.model_name
            else:
                self.tokenizer = tokenizer_class.from_pretrained(
                    model_name, cache_dir=self.args.cache_dir, **kwargs
                )
                self.args.tokenizer_name = self.args.model_name
        else:
            if not train_files:
                raise ValueError(
                    "model_name and tokenizer_name are not specified."
                    "You must specify train_files to train a Tokenizer."
                )
            else:
                self.train_tokenizer(train_files)
                new_tokenizer = True

        if self.args.config_name:
            self.config = config_class.from_pretrained(
                self.args.config_name,
                cache_dir=self.args.cache_dir,
                trust_remote_code=self.args.trust_remote_code,
                **kwargs,
            )
        elif self.args.model_name and self.args.model_name != "electra":
            self.config = config_class.from_pretrained(
                model_name,
                cache_dir=self.args.cache_dir,
                trust_remote_code=self.args.trust_remote_code,
                **kwargs,
            )
        else:
            self.config = config_class(**self.args.config, **kwargs)
        if self.args.vocab_size:
            self.config.vocab_size = self.args.vocab_size
        if new_tokenizer:
            self.config.vocab_size = len(self.tokenizer)

        if self.args.model_type == "electra":
            if generator_name:
                self.generator_config = ElectraConfig.from_pretrained(generator_name)
            elif self.args.model_name:
                self.generator_config = ElectraConfig.from_pretrained(
                    os.path.join(self.args.model_name, "generator_config"),
                    **kwargs,
                )
            else:
                self.generator_config = ElectraConfig(
                    **self.args.generator_config, **kwargs
                )
                if new_tokenizer:
                    self.generator_config.vocab_size = len(self.tokenizer)

            if discriminator_name:
                self.discriminator_config = ElectraConfig.from_pretrained(
                    discriminator_name
                )
            elif self.args.model_name:
                self.discriminator_config = ElectraConfig.from_pretrained(
                    os.path.join(self.args.model_name, "discriminator_config"),
                    **kwargs,
                )
            else:
                self.discriminator_config = ElectraConfig(
                    **self.args.discriminator_config, **kwargs
                )
                if new_tokenizer:
                    self.discriminator_config.vocab_size = len(self.tokenizer)

        if self.args.block_size <= 0:
            self.args.block_size = min(
                self.args.max_seq_length, self.tokenizer.model_max_length
            )
        else:
            self.args.block_size = min(
                self.args.block_size,
                self.tokenizer.model_max_length,
                self.args.max_seq_length,
            )

        if self.args.model_name:
            if self.args.model_type == "electra":
                if self.args.model_name == "electra":
                    generator_model = ElectraForMaskedLM.from_pretrained(generator_name)
                    discriminator_model = ElectraForPreTraining.from_pretrained(
                        discriminator_name
                    )
                    self.model = ElectraForLanguageModelingModel(
                        config=self.config,
                        generator_model=generator_model,
                        discriminator_model=discriminator_model,
                        generator_config=self.generator_config,
                        discriminator_config=self.discriminator_config,
                        tie_generator_and_discriminator_embeddings=self.args.tie_generator_and_discriminator_embeddings,
                    )
                    model_to_resize = (
                        self.model.generator_model.module
                        if hasattr(self.model.generator_model, "module")
                        else self.model.generator_model
                    )
                    model_to_resize.resize_token_embeddings(len(self.tokenizer))

                    model_to_resize = (
                        self.model.discriminator_model.module
                        if hasattr(self.model.discriminator_model, "module")
                        else self.model.discriminator_model
                    )
                    model_to_resize.resize_token_embeddings(len(self.tokenizer))
                    self.model.generator_model = generator_model
                    self.model.discriminator_model = discriminator_model
                else:
                    self.model = model_class.from_pretrained(
                        model_name,
                        config=self.config,
                        cache_dir=self.args.cache_dir,
                        generator_config=self.generator_config,
                        discriminator_config=self.discriminator_config,
                        **kwargs,
                    )
                    self.model.load_state_dict(
                        torch.load(
                            os.path.join(self.args.model_name, "pytorch_model.bin"),
                            map_location=self.device,
                        )
                    )
            else:
                if self.args.nf4:
                    from transformers import BitsAndBytesConfig

                    nf4_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_compute_dtype=torch.bfloat16,
                    )
                    self.model = model_class.from_pretrained(
                        model_name,
                        quantization_config=nf4_config,
                        trust_remote_code=self.args.trust_remote_code,
                    )
                else:
                    self.model = model_class.from_pretrained(
                        model_name,
                        config=self.config,
                        cache_dir=self.args.cache_dir,
                        **kwargs,
                    )

            if self.args.peft:
                from peft import LoraConfig, get_peft_model, LoftQConfig
                from peft.peft_model import PeftModel

                if self.args.qlora:
                    if self.args.nf4:
                        raise ValueError(
                            "PEFT and QLORA cannot be used together with NF4"
                        )
                    loftq_config = LoftQConfig(
                        loftq_bits=self.args.loftq_bits, **self.args.loftq_config
                    )
                    self.lora_config = LoraConfig(
                        init_lora_weights="loftq",
                        target_modules="all-linear",
                        loftq_config=loftq_config,
                        **self.args.lora_config,
                    )
                    self.args.fp16 = False
                else:
                    self.lora_config = LoraConfig(
                        use_rslora=True, target_modules="all-linear"
                    )
                self.model.gradient_checkpointing_enable()
                self.model.enable_input_require_grads()
                if adapter_name is not None:
                    self.model = PeftModel.from_pretrained(
                        self.model,
                        model_id=adapter_name,
                        adapter_name=adapter_name,
                    )
                    self.adapter_name = adapter_name
                else:
                    self.adapter_name = None
                    self.model = get_peft_model(self.model, self.lora_config)
                    self.model.print_trainable_parameters()

        else:
            logger.info(" Training language model from scratch")
            if self.args.model_type == "electra":
                generator_model = ElectraForMaskedLM(config=self.generator_config)
                discriminator_model = ElectraForPreTraining(
                    config=self.discriminator_config
                )
                self.model = ElectraForLanguageModelingModel(
                    config=self.config,
                    generator_model=generator_model,
                    discriminator_model=discriminator_model,
                    generator_config=self.generator_config,
                    discriminator_config=self.discriminator_config,
                    tie_generator_and_discriminator_embeddings=self.args.tie_generator_and_discriminator_embeddings,
                )
                model_to_resize = (
                    self.model.generator_model.module
                    if hasattr(self.model.generator_model, "module")
                    else self.model.generator_model
                )
                model_to_resize.resize_token_embeddings(len(self.tokenizer))

                model_to_resize = (
                    self.model.discriminator_model.module
                    if hasattr(self.model.discriminator_model, "module")
                    else self.model.discriminator_model
                )
                model_to_resize.resize_token_embeddings(len(self.tokenizer))
            else:
                self.model = model_class(config=self.config)
                model_to_resize = (
                    self.model.module if hasattr(self.model, "module") else self.model
                )
                model_to_resize.resize_token_embeddings(len(self.tokenizer))

        # if self.args.use_autoencoder:
        #     self.autoencoder_model = Autoencoder()
        #     if autoencoder_model is not None:
        #         # Load with PyTorch
        #         self.autoencoder_model.load_state_dict(
        #             torch.load(os.path.join(autoencoder_model, "pytorch_model.bin"))
        #         )
        #     elif model_name:
        #         # PyTorch model from a PyTorch checkpoint
        #         self.autoencoder_model.load_state_dict(
        #             torch.load(
        #                 os.path.join(
        #                     model_name, "autoencoder_model", "pytorch_model.bin"
        #                 )
        #             )
        #         )
        # else:
        #     self.autoencoder_model = None

        if model_type in ["camembert", "xlmroberta"]:
            warnings.warn(
                f"use_multiprocessing automatically disabled as {model_type}"
                " fails when using multiprocessing for feature conversion."
            )
            self.args.use_multiprocessing = False

        if self.args.wandb_project and not wandb_available:
            warnings.warn(
                "wandb_project specified but wandb is not available. Wandb disabled."
            )
            self.args.wandb_project = None

        if self.args.rag:
            if retrieval_model:
                self.retrieval_model = retrieval_model
            else:
                raise ValueError(
                    "RAG is enabled but no retrieval model is specified."
                    " Pass a retrieval model when instantiating the LanguageModelingModel to use RAG."
                )

    def train_model(
        self,
        train_file,
        output_dir=None,
        show_running_loss=True,
        args=None,
        eval_file=None,
        verbose=True,
        **kwargs,
    ):
        """
        Trains the model using 'train_file'

        Args:
            train_file: Path to text file containing the text to train the language model on.
            output_dir: The directory where model files will be saved. If not given, self.args.output_dir will be used.
            show_running_loss (optional): Set to False to prevent running loss from being printed to console. Defaults to True.
            args (optional): Optional changes to the args dict of the model. Any changes made will persist for the model.
            eval_file (optional): Path to eval file containing the text to evaluate the language model on.

        Returns:
            global_step: Number of global steps trained
            training_details: Average training loss if evaluate_during_training is False or full training progress scores if evaluate_during_training is True
        """  # noqa: ignore flake8"

        if args:
            self.args.update_from_dict(args)

        if self.args.silent:
            show_running_loss = False

        if self.args.evaluate_during_training and eval_file is None:
            raise ValueError(
                "evaluate_during_training is enabled but eval_file is not specified."
                " Pass eval_file to model.train_model() if using evaluate_during_training."
            )

        if not output_dir:
            output_dir = self.args.output_dir

        if (
            os.path.exists(output_dir)
            and os.listdir(output_dir)
            and not self.args.overwrite_output_dir
        ):
            raise ValueError(
                "Output directory ({}) already exists and is not empty."
                " Set args.overwrite_output_dir = True to overcome.".format(output_dir)
            )

        self._move_model_to_device()

        train_dataset = self.load_and_cache_examples(train_file, verbose=verbose)

        os.makedirs(output_dir, exist_ok=True)

        global_step, training_details = self.train(
            train_dataset,
            output_dir,
            show_running_loss=show_running_loss,
            eval_file=eval_file,
            verbose=verbose,
            **kwargs,
        )

        self.save_model(output_dir, model=self.model)
        if self.args.model_type == "electra":
            self.save_discriminator()
            self.save_generator()
        # model_to_save = self.model.module if hasattr(self.model, "module") else self.model
        # model_to_save.save_pretrained(output_dir)
        # self.tokenizer.save_pretrained(output_dir)
        # torch.save(self.args, os.path.join(output_dir, "training_args.bin"))

        if verbose:
            logger.info(
                " Training of {} model complete. Saved to {}.".format(
                    self.args.model_type, output_dir
                )
            )

        return global_step, training_details

    def train(
        self,
        train_dataset,
        output_dir,
        show_running_loss=True,
        eval_file=None,
        verbose=True,
        **kwargs,
    ):
        """
        Trains the model on train_dataset.

        Utility function to be used by the train_model() method. Not intended to be used directly.
        """

        model = self.model
        args = self.args
        tokenizer = self.tokenizer

        def collate(examples: List[torch.Tensor]):
            if tokenizer.pad_token is None:
                return pad_sequence(examples, batch_first=True)
            return pad_sequence(
                examples, batch_first=True, padding_value=tokenizer.pad_token_id
            )

        if self.is_world_master():
            tb_writer = SummaryWriter(log_dir=args.tensorboard_dir)
        if not self.args.stream_hf_datasets:
            train_sampler = (
                RandomSampler(train_dataset)
                if args.local_rank == -1
                else DistributedSampler(train_dataset)
            )
        if self.args.use_hf_datasets:
            # Inputs are already padded so default collation is fine
            train_dataloader = DataLoader(
                train_dataset,
                batch_size=args.train_batch_size,
                sampler=train_sampler if not self.args.stream_hf_datasets else None,
            )
        else:
            train_dataloader = DataLoader(
                train_dataset,
                batch_size=args.train_batch_size,
                sampler=train_sampler,
                collate_fn=collate,
            )

        if args.max_steps > 0:
            t_total = args.max_steps
            try:
                args.num_train_epochs = (
                    args.max_steps
                    // (len(train_dataloader) // args.gradient_accumulation_steps)
                    + 1
                )
            except TypeError:
                pass
        else:
            t_total = (
                len(train_dataloader)
                // args.gradient_accumulation_steps
                * args.num_train_epochs
            )

        optimizer_grouped_parameters = self.get_optimizer_parameters(model, args)

        warmup_steps = math.ceil(t_total * args.warmup_ratio)
        args.warmup_steps = (
            warmup_steps if args.warmup_steps == 0 else args.warmup_steps
        )

        if args.optimizer == "AdamW":
            optimizer = AdamW(
                optimizer_grouped_parameters,
                lr=args.learning_rate,
                eps=args.adam_epsilon,
                betas=args.adam_betas,
            )
        elif args.optimizer == "Adafactor":
            optimizer = Adafactor(
                optimizer_grouped_parameters,
                lr=args.learning_rate,
                eps=args.adafactor_eps,
                clip_threshold=args.adafactor_clip_threshold,
                decay_rate=args.adafactor_decay_rate,
                beta1=args.adafactor_beta1,
                weight_decay=args.weight_decay,
                scale_parameter=args.adafactor_scale_parameter,
                relative_step=args.adafactor_relative_step,
                warmup_init=args.adafactor_warmup_init,
            )
        elif args.optimizer == "Adam8bit":
            from bitsandbytes.optim import Adam8bit

            optimizer = Adam8bit(
                optimizer_grouped_parameters,
                lr=args.learning_rate,
                eps=args.adam_epsilon,
                betas=args.adam_betas,
            )

        else:
            raise ValueError(
                "{} is not a valid optimizer class. Please use one of ('AdamW', 'Adafactor') instead.".format(
                    args.optimizer
                )
            )

        if args.scheduler == "constant_schedule":
            scheduler = get_constant_schedule(optimizer)

        elif args.scheduler == "constant_schedule_with_warmup":
            scheduler = get_constant_schedule_with_warmup(
                optimizer, num_warmup_steps=args.warmup_steps
            )

        elif args.scheduler == "linear_schedule_with_warmup":
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=args.warmup_steps,
                num_training_steps=t_total,
            )

        elif args.scheduler == "cosine_schedule_with_warmup":
            scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=args.warmup_steps,
                num_training_steps=t_total,
                num_cycles=args.cosine_schedule_num_cycles,
            )

        elif args.scheduler == "cosine_with_hard_restarts_schedule_with_warmup":
            scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(
                optimizer,
                num_warmup_steps=args.warmup_steps,
                num_training_steps=t_total,
                num_cycles=args.cosine_schedule_num_cycles,
            )

        elif args.scheduler == "polynomial_decay_schedule_with_warmup":
            scheduler = get_polynomial_decay_schedule_with_warmup(
                optimizer,
                num_warmup_steps=args.warmup_steps,
                num_training_steps=t_total,
                lr_end=args.polynomial_decay_schedule_lr_end,
                power=args.polynomial_decay_schedule_power,
            )

        else:
            raise ValueError("{} is not a valid scheduler.".format(args.scheduler))

        if (
            args.model_name
            and os.path.isfile(os.path.join(args.model_name, "optimizer.pt"))
            and os.path.isfile(os.path.join(args.model_name, "scheduler.pt"))
        ):
            # Load in optimizer and scheduler states
            optimizer.load_state_dict(
                torch.load(os.path.join(args.model_name, "optimizer.pt"))
            )
            scheduler.load_state_dict(
                torch.load(os.path.join(args.model_name, "scheduler.pt"))
            )

        if args.n_gpu > 1:
            model = torch.nn.DataParallel(model)

        # Distributed training
        if args.local_rank != -1:
            model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[args.local_rank],
                output_device=args.local_rank,
                find_unused_parameters=True,
            )

        logger.info(" Training started")

        global_step = 0
        training_progress_scores = None
        tr_loss, logging_loss = 0.0, 0.0
        model.zero_grad()
        train_iterator = trange(
            int(args.num_train_epochs), desc="Epoch", disable=args.silent, mininterval=0
        )
        epoch_number = 0
        best_eval_metric = None
        early_stopping_counter = 0
        steps_trained_in_current_epoch = 0
        epochs_trained = 0

        if args.model_name and os.path.exists(args.model_name):
            try:
                # set global_step to gobal_step of last saved checkpoint from model path
                checkpoint_suffix = args.model_name.split("/")[-1].split("-")
                if len(checkpoint_suffix) > 2:
                    checkpoint_suffix = checkpoint_suffix[1]
                else:
                    checkpoint_suffix = checkpoint_suffix[-1]
                global_step = int(checkpoint_suffix)
                epochs_trained = global_step // (
                    len(train_dataloader) // args.gradient_accumulation_steps
                )
                steps_trained_in_current_epoch = global_step % (
                    len(train_dataloader) // args.gradient_accumulation_steps
                )

                logger.info(
                    "   Continuing training from checkpoint, will skip to saved global_step"
                )
                logger.info("   Continuing training from epoch %d", epochs_trained)
                logger.info("   Continuing training from global step %d", global_step)
                logger.info(
                    "   Will skip the first %d steps in the current epoch",
                    steps_trained_in_current_epoch,
                )
            except ValueError:
                logger.info("   Starting fine-tuning.")

        if args.evaluate_during_training:
            training_progress_scores = self._create_training_progress_scores(**kwargs)

        if args.wandb_project:
            wandb.init(
                project=args.wandb_project, config={**asdict(args)}, **args.wandb_kwargs
            )
            wandb.run._label(repo="simpletransformers")
            wandb.watch(self.model)
            self.wandb_run_id = wandb.run.id

        if args.fp16:
            from torch.cuda import amp

            scaler = amp.GradScaler()

        for current_epoch in train_iterator:
            model.train()
            if isinstance(train_dataloader, DataLoader) and isinstance(
                train_dataloader.sampler, DistributedSampler
            ):
                train_dataloader.sampler.set_epoch(current_epoch)
            if epochs_trained > 0:
                epochs_trained -= 1
                continue
            train_iterator.set_description(
                f"Epoch {epoch_number + 1} of {args.num_train_epochs}"
            )
            batch_iterator = tqdm(
                train_dataloader,
                desc=f"Running Epoch {epoch_number + 1} of {args.num_train_epochs}",
                disable=args.silent,
                mininterval=0,
            )
            for step, batch in enumerate(batch_iterator):
                if steps_trained_in_current_epoch > 0:
                    steps_trained_in_current_epoch -= 1
                    continue

                # TODO: Move this to _get_inputs_dict and keep the attention masks
                if self.args.use_hf_datasets:
                    if self.args.stream_hf_datasets:
                        batch["input_ids"] = torch.stack(batch["input_ids"])
                        batch["attention_mask"] = torch.stack(batch["attention_mask"])
                        if self.args.model_type in [
                            "roberta",
                            "roberta-autoencoder",
                            "xlmroberta",
                        ]:
                            # We need a list of zeros the same shape as attention_mask for RoBERTa
                            batch["token_type_ids"] = [
                                torch.zeros_like(attention_mask)
                                for attention_mask in batch["attention_mask"]
                            ]
                            batch["token_type_ids"] = torch.stack(
                                batch["token_type_ids"]
                            )
                        else:
                            batch["token_type_ids"] = torch.stack(
                                batch["token_type_ids"]
                            )
                    input_ids = batch["input_ids"]
                else:
                    input_ids = batch

                inputs, labels = (
                    mask_tokens(input_ids, tokenizer, args)
                    if args.mlm
                    else (input_ids, input_ids)
                )
                inputs = inputs.to(self.device)
                attention_mask = (
                    batch["attention_mask"].to(self.device)
                    if self.args.use_hf_datasets
                    else None
                )
                token_type_ids = (
                    batch["token_type_ids"].to(self.device)
                    if self.args.use_hf_datasets and "token_type_ids" in batch
                    else None
                )
                labels = labels.to(self.device)

                if token_type_ids is None:
                    inputs_dict = {
                        "input_ids": inputs,
                        "attention_mask": attention_mask,
                    }
                else:
                    inputs_dict = {
                        "input_ids": inputs,
                        "attention_mask": attention_mask,
                        "token_type_ids": token_type_ids,
                    }

                if args.fp16:
                    with amp.autocast():
                        if args.model_type == "longformer":
                            outputs = model(inputs, labels=labels)
                        elif args.model_type == "electra":
                            outputs = model(
                                inputs,
                                labels,
                                attention_mask=attention_mask,
                                token_type_ids=token_type_ids,
                            )
                        else:
                            outputs = (
                                model(**inputs_dict, labels=labels)
                                if args.mlm
                                else model(**inputs_dict, labels=labels)
                            )
                        # model outputs are always tuple in pytorch-transformers (see doc)
                        if args.model_type == "electra":
                            g_loss = outputs[0]
                            d_loss = outputs[1]
                            loss = g_loss + args.discriminator_loss_weight * d_loss
                        else:
                            loss = outputs[0]
                else:
                    if args.model_type == "longformer":
                        outputs = model(**inputs_dict, labels=labels)
                    elif args.model_type == "electra":
                        outputs = model(
                            input_ids,
                            labels,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                        )
                    else:
                        outputs = (
                            model(**inputs_dict, labels=labels)
                            if args.mlm
                            else model(**inputs_dict, labels=labels)
                        )
                    # model outputs are always tuple in pytorch-transformers (see doc)
                    if args.model_type == "electra":
                        g_loss = outputs[0]
                        d_loss = outputs[1]
                        loss = g_loss + args.discriminator_loss_weight * d_loss
                    else:
                        loss = outputs[0]
                    # if loss.item() < 1:
                    #     masked = (labels[0] != -100).nonzero()
                    #     print(labels[0][masked])
                    #     preds = outputs[1][0, masked, :].clone().detach().cpu().numpy()
                    #     print(np.argmax(preds, axis=2))

                if args.n_gpu > 1:
                    loss = (
                        loss.mean()
                    )  # mean() to average on multi-gpu parallel training

                current_loss = loss.item()

                if show_running_loss:
                    batch_iterator.set_description(
                        f"Epochs {epoch_number + 1}/{args.num_train_epochs}. Running Loss: {current_loss:9.4f}"
                    )

                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                if args.fp16:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

                tr_loss += loss.item()
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    if args.fp16:
                        scaler.unscale_(optimizer)
                    if args.optimizer == "AdamW":
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(), args.max_grad_norm
                        )

                    if args.fp16:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    scheduler.step()  # Update learning rate schedule
                    model.zero_grad()
                    global_step += 1

                    if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                        # Log metrics
                        if self.is_world_master():
                            tb_writer.add_scalar(
                                "lr", scheduler.get_last_lr()[0], global_step
                            )
                            tb_writer.add_scalar(
                                "loss",
                                (tr_loss - logging_loss) / args.logging_steps,
                                global_step,
                            )
                        logging_loss = tr_loss
                        if args.wandb_project or self.is_sweeping:
                            wandb.log(
                                {
                                    "Training loss": current_loss,
                                    "lr": scheduler.get_last_lr()[0],
                                    "global_step": global_step,
                                }
                            )

                    if args.save_steps > 0 and global_step % args.save_steps == 0:
                        # Save model checkpoint
                        output_dir_current = os.path.join(
                            output_dir, "checkpoint-{}".format(global_step)
                        )

                        self.save_model(
                            output_dir_current, optimizer, scheduler, model=model
                        )

                    if args.evaluate_during_training and (
                        args.evaluate_during_training_steps > 0
                        and global_step % args.evaluate_during_training_steps == 0
                    ):
                        # Only evaluate when single GPU otherwise metrics may not average well
                        results = self.eval_model(
                            eval_file,
                            verbose=verbose and args.evaluate_during_training_verbose,
                            silent=args.evaluate_during_training_silent,
                            **kwargs,
                        )

                        if self.is_world_master():
                            for key, value in results.items():
                                try:
                                    tb_writer.add_scalar(
                                        "eval_{}".format(key), value, global_step
                                    )
                                except (NotImplementedError, AssertionError):
                                    pass

                        output_dir_current = os.path.join(
                            output_dir, "checkpoint-{}".format(global_step)
                        )

                        if args.save_eval_checkpoints:
                            self.save_model(
                                output_dir_current,
                                optimizer,
                                scheduler,
                                model=model,
                                results=results,
                            )

                        training_progress_scores["global_step"].append(global_step)
                        training_progress_scores["train_loss"].append(current_loss)
                        for key in results:
                            training_progress_scores[key].append(results[key])
                        report = pd.DataFrame(training_progress_scores)
                        report.to_csv(
                            os.path.join(
                                args.output_dir, "training_progress_scores.csv"
                            ),
                            index=False,
                        )

                        if args.wandb_project or self.is_sweeping:
                            wandb.log(self._get_last_metrics(training_progress_scores))

                        if not best_eval_metric:
                            best_eval_metric = results[args.early_stopping_metric]
                            self.save_model(
                                args.best_model_dir,
                                optimizer,
                                scheduler,
                                model=model,
                                results=results,
                            )
                        if best_eval_metric and args.early_stopping_metric_minimize:
                            if (
                                results[args.early_stopping_metric] - best_eval_metric
                                < args.early_stopping_delta
                            ):
                                best_eval_metric = results[args.early_stopping_metric]
                                self.save_model(
                                    args.best_model_dir,
                                    optimizer,
                                    scheduler,
                                    model=model,
                                    results=results,
                                )
                                early_stopping_counter = 0
                            else:
                                if args.use_early_stopping:
                                    if (
                                        early_stopping_counter
                                        < args.early_stopping_patience
                                    ):
                                        early_stopping_counter += 1
                                        if verbose:
                                            logger.info(
                                                f" No improvement in {args.early_stopping_metric}"
                                            )
                                            logger.info(
                                                f" Current step: {early_stopping_counter}"
                                            )
                                            logger.info(
                                                f" Early stopping patience: {args.early_stopping_patience}"
                                            )
                                    else:
                                        if verbose:
                                            logger.info(
                                                f" Patience of {args.early_stopping_patience} steps reached."
                                            )
                                            logger.info(" Training terminated.")
                                            train_iterator.close()
                                        return (
                                            global_step,
                                            (
                                                tr_loss / global_step
                                                if not self.args.evaluate_during_training
                                                else training_progress_scores
                                            ),
                                        )
                        else:
                            if (
                                results[args.early_stopping_metric] - best_eval_metric
                                > args.early_stopping_delta
                            ):
                                best_eval_metric = results[args.early_stopping_metric]
                                self.save_model(
                                    args.best_model_dir,
                                    optimizer,
                                    scheduler,
                                    model=model,
                                    results=results,
                                )
                                early_stopping_counter = 0
                            else:
                                if args.use_early_stopping:
                                    if (
                                        early_stopping_counter
                                        < args.early_stopping_patience
                                    ):
                                        early_stopping_counter += 1
                                        if verbose:
                                            logger.info(
                                                f" No improvement in {args.early_stopping_metric}"
                                            )
                                            logger.info(
                                                f" Current step: {early_stopping_counter}"
                                            )
                                            logger.info(
                                                f" Early stopping patience: {args.early_stopping_patience}"
                                            )
                                    else:
                                        if verbose:
                                            logger.info(
                                                f" Patience of {args.early_stopping_patience} steps reached."
                                            )
                                            logger.info(" Training terminated.")
                                            train_iterator.close()
                                        return (
                                            global_step,
                                            (
                                                tr_loss / global_step
                                                if not self.args.evaluate_during_training
                                                else training_progress_scores
                                            ),
                                        )
                        model.train()

                if args.max_steps > 0 and global_step > args.max_steps:
                    return (
                        global_step,
                        (
                            tr_loss / global_step
                            if not self.args.evaluate_during_training
                            else training_progress_scores
                        ),
                    )

            epoch_number += 1
            output_dir_current = os.path.join(
                output_dir,
                "checkpoint-{}-epoch-{}".format(global_step, epoch_number),
            )

            if args.save_model_every_epoch or args.evaluate_during_training:
                os.makedirs(output_dir_current, exist_ok=True)

            if args.save_model_every_epoch:
                self.save_model(output_dir_current, optimizer, scheduler, model=model)

            if args.evaluate_during_training and args.evaluate_each_epoch:
                results = self.eval_model(
                    eval_file,
                    verbose=verbose and args.evaluate_during_training_verbose,
                    silent=args.evaluate_during_training_silent,
                    **kwargs,
                )

                self.save_model(
                    output_dir_current, optimizer, scheduler, results=results
                )

                training_progress_scores["global_step"].append(global_step)
                training_progress_scores["train_loss"].append(current_loss)
                for key in results:
                    training_progress_scores[key].append(results[key])
                report = pd.DataFrame(training_progress_scores)
                report.to_csv(
                    os.path.join(args.output_dir, "training_progress_scores.csv"),
                    index=False,
                )

                if args.wandb_project or self.is_sweeping:
                    wandb.log(self._get_last_metrics(training_progress_scores))

                if not best_eval_metric:
                    best_eval_metric = results[args.early_stopping_metric]
                    self.save_model(
                        args.best_model_dir,
                        optimizer,
                        scheduler,
                        model=model,
                        results=results,
                    )
                if best_eval_metric and args.early_stopping_metric_minimize:
                    if (
                        results[args.early_stopping_metric] - best_eval_metric
                        < args.early_stopping_delta
                    ):
                        best_eval_metric = results[args.early_stopping_metric]
                        self.save_model(
                            args.best_model_dir,
                            optimizer,
                            scheduler,
                            model=model,
                            results=results,
                        )
                        early_stopping_counter = 0
                    else:
                        if (
                            args.use_early_stopping
                            and args.early_stopping_consider_epochs
                        ):
                            if early_stopping_counter < args.early_stopping_patience:
                                early_stopping_counter += 1
                                if verbose:
                                    logger.info(
                                        f" No improvement in {args.early_stopping_metric}"
                                    )
                                    logger.info(
                                        f" Current step: {early_stopping_counter}"
                                    )
                                    logger.info(
                                        f" Early stopping patience: {args.early_stopping_patience}"
                                    )
                            else:
                                if verbose:
                                    logger.info(
                                        f" Patience of {args.early_stopping_patience} steps reached"
                                    )
                                    logger.info(" Training terminated.")
                                    train_iterator.close()
                                return (
                                    global_step,
                                    (
                                        tr_loss / global_step
                                        if not self.args.evaluate_during_training
                                        else training_progress_scores
                                    ),
                                )
                else:
                    if (
                        results[args.early_stopping_metric] - best_eval_metric
                        > args.early_stopping_delta
                    ):
                        best_eval_metric = results[args.early_stopping_metric]
                        self.save_model(
                            args.best_model_dir,
                            optimizer,
                            scheduler,
                            model=model,
                            results=results,
                        )
                        early_stopping_counter = 0
                    else:
                        if (
                            args.use_early_stopping
                            and args.early_stopping_consider_epochs
                        ):
                            if early_stopping_counter < args.early_stopping_patience:
                                early_stopping_counter += 1
                                if verbose:
                                    logger.info(
                                        f" No improvement in {args.early_stopping_metric}"
                                    )
                                    logger.info(
                                        f" Current step: {early_stopping_counter}"
                                    )
                                    logger.info(
                                        f" Early stopping patience: {args.early_stopping_patience}"
                                    )
                            else:
                                if verbose:
                                    logger.info(
                                        f" Patience of {args.early_stopping_patience} steps reached"
                                    )
                                    logger.info(" Training terminated.")
                                    train_iterator.close()
                                return (
                                    global_step,
                                    (
                                        tr_loss / global_step
                                        if not self.args.evaluate_during_training
                                        else training_progress_scores
                                    ),
                                )

            if args.max_steps > 0 and global_step > args.max_steps:
                return (
                    global_step,
                    (
                        tr_loss / global_step
                        if not self.args.evaluate_during_training
                        else training_progress_scores
                    ),
                )

        return (
            global_step,
            (
                tr_loss / global_step
                if not self.args.evaluate_during_training
                else training_progress_scores
            ),
        )

    def eval_model(
        self,
        eval_file,
        output_dir=None,
        evaluate_generated_text=False,
        verbose=True,
        silent=False,
        **kwargs,
    ):
        """
        Evaluates the model on eval_df. Saves results to args.output_dir
            result: Dictionary containing evaluation results.
        """  # noqa: ignore flake8"

        if not output_dir:
            output_dir = self.args.output_dir

        self._move_model_to_device()

        if evaluate_generated_text:
            raise NotImplementedError(
                "evaluate_generated_text is not yet implemented for this model type."
            )
        else:
            eval_dataset = self.load_and_cache_examples(
                eval_file, evaluate=True, verbose=verbose, silent=silent
            )
            os.makedirs(output_dir, exist_ok=True)

            result = self.evaluate(
                eval_dataset, output_dir, verbose=verbose, silent=silent, **kwargs
            )
            self.results.update(result)

            if verbose:
                logger.info(self.results)

            return result

    def evaluate(
        self,
        eval_dataset,
        output_dir,
        prefix="",
        verbose=True,
        silent=False,
        **kwargs,
    ):
        """
        Evaluates the model on eval_dataset.

        Utility function to be used by the eval_model() method. Not intended to be used directly.
        """

        model = self.model
        args = self.args
        eval_output_dir = output_dir
        tokenizer = self.tokenizer

        results = {}

        def collate(examples: List[torch.Tensor]):
            if tokenizer.pad_token is None:
                return pad_sequence(examples, batch_first=True)
            return pad_sequence(
                examples, batch_first=True, padding_value=tokenizer.pad_token_id
            )

        eval_sampler = SequentialSampler(eval_dataset)
        if self.args.use_hf_datasets:
            # Inputs are already padded so default collation is fine
            eval_dataloader = DataLoader(
                eval_dataset, batch_size=args.train_batch_size, sampler=eval_sampler
            )
        else:
            eval_dataloader = DataLoader(
                eval_dataset,
                batch_size=args.train_batch_size,
                sampler=eval_sampler,
                collate_fn=collate,
            )

        if args.n_gpu > 1:
            model = torch.nn.DataParallel(model)

        eval_loss = 0.0
        nb_eval_steps = 0
        model.eval()

        for batch in tqdm(
            eval_dataloader, disable=args.silent or silent, desc="Running Evaluation"
        ):
            if self.args.use_hf_datasets:
                input_ids = batch["input_ids"]

            inputs, labels = (
                mask_tokens(batch, tokenizer, args)
                if args.mlm
                else (input_ids, input_ids)
            )
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            if "token_type_ids" in batch:
                inputs_dict = {
                    "input_ids": inputs,
                    "attention_mask": (
                        batch["attention_mask"].to(self.device)
                        if self.args.use_hf_datasets
                        else None
                    ),
                    "token_type_ids": (
                        batch["token_type_ids"].to(self.device)
                        if self.args.use_hf_datasets
                        else None
                    ),
                }
            else:
                inputs_dict = {
                    "input_ids": inputs,
                    "attention_mask": (
                        batch["attention_mask"].to(self.device)
                        if self.args.use_hf_datasets
                        else None
                    ),
                }

            with torch.no_grad():
                outputs = (
                    model(**inputs_dict, labels=labels)
                    if args.mlm
                    else model(**inputs_dict, labels=labels)
                )
                if args.model_type == "electra":
                    g_loss = outputs[0]
                    d_loss = outputs[1]
                    lm_loss = g_loss + args.discriminator_loss_weight * d_loss
                else:
                    lm_loss = outputs[0]
                if self.args.n_gpu > 1:
                    lm_loss = lm_loss.mean()
                eval_loss += lm_loss.item()
            nb_eval_steps += 1

        eval_loss = eval_loss / nb_eval_steps
        perplexity = torch.exp(torch.tensor(eval_loss))

        results["eval_loss"] = eval_loss
        results["perplexity"] = perplexity

        output_eval_file = os.path.join(eval_output_dir, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            for key in sorted(results.keys()):
                writer.write("{} = {}\n".format(key, str(results[key])))

        return results

    def predict(
        self,
        to_predict,
        generation_args=None,
        rag_queries=None,
        knowledge_dataset=None,
        user_role="user",
        system_role="system",
        system_prompt="",
        apply_chat_template=False,
        **kwargs,
    ):
        """
        Performs text completions on a list of text. To be used with language models.

        Args:

        to_predict: A list of text to make predictions on.
        generation_args: An instance of the `GenerationArgs` class containing the generation arguments for the model.
        rag_queries (optional): A list of text to be used as queries for the RAG model. Only applicable if rag is enabled.
        knowledge_dataset (optional): A list of text to be used as knowledge for the RAG model. Only applicable if the model is a RAG model.
        **kwargs: Additional arguments to be passed to the models `generate()` method during inference.

        Returns:
        preds: A list of the predicted sequences.
        """
        self._move_model_to_device()

        if not generation_args:
            generation_args = GenerationArgs()

        if self.args.peft and self.adapter_name:
            logger.info(
                "Merging adapter with model for faster inference. Continuing training from this point may result in unexpected behavior."
            )
            self.model = self.model.merge_and_unload()

        self.tokenizer.padding_side = "left"

        if self.args.rag:
            if not rag_queries:
                rag_queries = to_predict
                raise Warning(
                    "No `rag_queries` provided. Using `to_predict` as `rag_queries`."
                )

            context_docs = self.retrieval_model.predict(
                rag_queries,
                passages_only=True,
                prediction_passages=knowledge_dataset,
            )

            to_predict = [
                f"Context: {' '.join(context_doc)} {text}"
                for context_doc, text in zip(context_docs, to_predict)
            ]
            # TODO:
            # - Simplest option is to just prepend context: context_docs to to_predict
            # - Advanced option is to have <CONTEXT_1> ... <CONTEXT_n> in to_predict and then replace <CONTEXT_1> ... <CONTEXT_n> with context_docs

        if apply_chat_template:
            to_predict = apply_chat_template_to_inputs(
                to_predict, user_role, system_role, system_prompt, self.tokenizer
            )

        try:
            inputs = self.tokenizer(
                to_predict,
                padding=True,
                return_tensors="pt",
            )
        except ValueError:
            if not self.tokenizer.pad_token:
                warnings.warn(
                    "The tokenizer you are using does not have a pad_token assigned. Setting to `eos_token`."
                )
                self.tokenizer.pad_token = self.tokenizer.eos_token
            inputs = self.tokenizer(
                to_predict,
                padding=True,
                return_tensors="pt",
            )

        input_ids_tensor = inputs["input_ids"]
        attention_mask_tensor = inputs["attention_mask"]

        # Create a TensorDataset
        dataset = TensorDataset(input_ids_tensor, attention_mask_tensor)

        # Define batch size

        # Create the dataloader
        predict_dataloader = DataLoader(
            dataset, batch_size=self.args.eval_batch_size, shuffle=False
        )

        # Put model in evaluation mode
        self.model.eval()

        # Predict
        responses = []
        outputs = []
        for batch in tqdm(
            predict_dataloader, desc="Generating outputs", disable=self.args.silent
        ):
            batch = tuple(t.to(self.device) for t in batch)
            input_ids, attention_mask = batch

            generation_output = self.model.generate(
                input_ids,
                attention_mask=attention_mask,
                return_dict_in_generate=True,
                output_scores=True,
                **generation_args.get_dict(),
                **kwargs,
            )

            # response_tests = self.tokenizer.batch_decode(generation_output.sequences[:, input_ids.shape[1]:], skip_special_tokens=True)

            for i, s in enumerate(generation_output.sequences):
                output = self.tokenizer.decode(
                    s[input_ids[i].shape[0] :], skip_special_tokens=True
                )
                responses.append(output)

            # responses.extend(response_tests)
            outputs.extend(generation_output)

        return responses, generation_output

    def load_and_cache_examples(
        self, file_path, evaluate=False, no_cache=False, verbose=True, silent=False
    ):
        """
        Reads a text file from file_path and creates training features.

        Utility function for train() and eval() methods. Not intended to be used directly.
        """

        tokenizer = self.tokenizer
        args = self.args

        if not no_cache:
            no_cache = args.no_cache

        if not no_cache:
            os.makedirs(self.args.cache_dir, exist_ok=True)

        mode = "dev" if evaluate else "train"

        if self.args.use_hf_datasets:
            dataset = load_hf_dataset(
                file_path, tokenizer, self.args, retrieval_model=self.retrieval_model
            )
            return dataset
        elif args.dataset_class:
            CustomDataset = args.dataset_class
            return CustomDataset(tokenizer, args, file_path, mode, args.block_size)
        else:
            dataset_type = args.dataset_type
            if dataset_type == "text":
                return TextDataset(
                    tokenizer, file_path, args.block_size, overwrite_cache=True
                )
            elif dataset_type == "line_by_line":
                return LineByLineTextDataset(tokenizer, file_path, args.block_size)
            else:
                special_tokens_count = (
                    3
                    if bool(args.model_type in ["roberta", "camembert", "xlmroberta"])
                    else 2
                )
                if self.args.max_seq_length > 509 and self.args.model_type not in [
                    "longformer",
                    "bigbird",
                    "nystromformer",
                ]:
                    self.args.max_seq_length = (
                        509
                        if bool(
                            args.model_type in ["roberta", "camembert", "xlmroberta"]
                        )
                        else 510
                    )
                    self.args.block_size = (
                        509
                        if bool(
                            args.model_type in ["roberta", "camembert", "xlmroberta"]
                        )
                        else 510
                    )
                return SimpleDataset(
                    tokenizer,
                    self.args,
                    file_path,
                    mode,
                    args.block_size,
                    special_tokens_count,
                    sliding_window=args.sliding_window,
                )

    def train_tokenizer(
        self,
        train_files,
        tokenizer_name=None,
        output_dir=None,
        use_trained_tokenizer=True,
    ):
        """
        Train a new tokenizer on `train_files`.

        Args:

        - train_files: List of files to be used when training the tokenizer.

        - tokenizer_name: Name of a pretrained tokenizer or a path to a directory containing a tokenizer.

        - output_dir (optional): The directory where model files will be saved. If not given, self.args.output_dir
        will be used.

        - use_trained_tokenizer (optional): Load the trained tokenizer once training completes.

        Returns: None
        """

        if not self.args.vocab_size:
            raise AttributeError(
                "Cannot train a new tokenizer as vocab_size is not specified in args dict. "
                "Either provide a tokenizer or specify vocab_size."
            )

        if not isinstance(train_files, list):
            train_files = [train_files]

        if not output_dir:
            output_dir = self.args.output_dir

        if self.args.model_type in ["bert", "electra"]:
            tokenizer = BertWordPieceTokenizer(
                clean_text=self.args.clean_text,
                handle_chinese_chars=self.args.handle_chinese_chars,
                strip_accents=self.args.strip_accents,
                lowercase=self.args.do_lower_case,
            )
            self.args.special_tokens = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
            self.args.wordpieces_prefix = "##"

            tokenizer.train(
                files=train_files,
                vocab_size=self.args.vocab_size,
                min_frequency=self.args.min_frequency,
                special_tokens=self.args.special_tokens,
                wordpieces_prefix="##",
            )
        elif self.args.model_type in ["bigbird", "xlmroberta", "nystromformer"]:
            # The google BigBird way
            # Tokenizers sentencepiece does not build a BigBird compatible vocabulary model
            import sentencepiece as spm
            import shutil

            os.makedirs(output_dir, exist_ok=True)
            files = ",".join(train_files)

            if self.args.model_type in ["xlmroberta"]:
                # </s>,<s>,<unk>,<pad> are built in -- leave as default
                # XLMRoberta uses sentencepiece.bpe as a vocab model prefix
                prefix = "sentencepiece.bpe"
                self.args.special_tokens = [
                    "<s>",
                    "</s>",
                    "<pad>",
                    "<mask>",
                    "<s>NOTUSED",
                    "</s>NOTUSED",
                ]
                spm.SentencePieceTrainer.Train(
                    f"--input={files} --user_defined_symbols=<pad>,<mask>,<s>NOTUSED,</s>NOTUSED --model_type=bpe --model_prefix={prefix} --vocab_size={self.args.vocab_size - 2} --shuffle_input_sentence=true --max_sentence_length=10000"
                )
            else:
                # </s>,<s>,<unk>,<pad> are built in -- leave as default
                # BigBird uses spiece as a vocab model prefix
                # Nystromformer uses spiece as a vocab model prefix
                prefix = "spiece"
                self.args.special_tokens = [
                    "<s>",
                    "</s>",
                    "<pad>",
                    "[SEP]",
                    "[CLS]",
                    "[MASK]",
                ]
                spm.SentencePieceTrainer.Train(
                    f"--input={files} --user_defined_symbols=<pad>,[SEP],[CLS],[MASK] --model_type=bpe --model_prefix=spiece --vocab_size={self.args.vocab_size} --shuffle_input_sentence=true --max_sentence_length=10000"
                )

            # SentencePiece There is no option for output path https://github.com/google/sentencepiece/blob/master/doc/options.md
            if os.path.exists(output_dir + "/" + f"{prefix}.model"):
                os.remove(output_dir + "/" + f"{prefix}.model")
            shutil.move(src=f"{prefix}.model", dst=output_dir)

            if os.path.exists(output_dir + "/" + f"{prefix}.vocab"):
                os.remove(output_dir + "/" + f"{prefix}.vocab")
            shutil.move(src=f"{prefix}.vocab", dst=output_dir)
        else:
            tokenizer = ByteLevelBPETokenizer(lowercase=self.args.do_lower_case)

            tokenizer.train(
                files=train_files,
                vocab_size=self.args.vocab_size,
                min_frequency=self.args.min_frequency,
                special_tokens=self.args.special_tokens,
            )

        if self.args.model_type not in ["bigbird", "xlmroberta", "nystromformer"]:
            os.makedirs(output_dir, exist_ok=True)

            tokenizer.save_model(output_dir)
            logger.info(
                " Training of {} tokenizer complete. Saved to {}.".format(
                    tokenizer_name, output_dir
                )
            )

        _, _, tokenizer_class = MODEL_CLASSES[self.args.model_type]
        tokenizer = tokenizer_class.from_pretrained(output_dir)

        if use_trained_tokenizer:
            self.tokenizer = tokenizer
            self.args.tokenizer_name = output_dir
            try:
                if self.args.model_type == "electra":
                    model_to_resize = (
                        self.model.generator_model.module
                        if hasattr(self.model.generator_model, "module")
                        else self.model.generator_model
                    )
                    model_to_resize.resize_token_embeddings(len(self.tokenizer))

                    model_to_resize = (
                        self.model.discriminator_model.module
                        if hasattr(self.model.discriminator_model, "module")
                        else self.model.discriminator_model
                    )
                    model_to_resize.resize_token_embeddings(len(self.tokenizer))

                model_to_resize = (
                    self.model.module if hasattr(self.model, "module") else self.model
                )

                model_to_resize.resize_token_embeddings(len(self.tokenizer))
            except AttributeError:
                pass

    def save_discriminator(self, output_dir=None):
        if self.args.model_type == "electra":
            if not self.args.no_save:
                if not output_dir:
                    output_dir = os.path.join(
                        self.args.output_dir, "discriminator_model"
                    )
                os.makedirs(output_dir, exist_ok=True)
                model_to_save = (
                    self.model.discriminator_model.module
                    if hasattr(self.model.discriminator_model, "module")
                    else self.model.discriminator_model
                )
                model_to_save.save_pretrained(output_dir)
                self.tokenizer.save_pretrained(output_dir)
        else:
            raise ValueError("Model must be of ElectraForLanguageModelingModel type")

    def save_generator(self, output_dir=None):
        if self.args.model_type == "electra":
            if not self.args.no_save:
                if not output_dir:
                    output_dir = os.path.join(self.args.output_dir, "generator_model")
                os.makedirs(output_dir, exist_ok=True)
                model_to_save = (
                    self.model.generator_model.module
                    if hasattr(self.model.generator_model, "module")
                    else self.model.generator_model
                )
                model_to_save.save_pretrained(output_dir)
                self.tokenizer.save_pretrained(output_dir)
        else:
            raise ValueError("Model must be of ElectraForLanguageModelingModel type")

    def _threshold(self, x, threshold):
        if x >= threshold:
            return 1
        return 0

    def _move_model_to_device(self):
        if not self.args.qlora and not self.args.nf4:
            self.model.to(self.device)

    def _create_training_progress_scores(self, **kwargs):
        extra_metrics = {key: [] for key in kwargs}
        training_progress_scores = {
            "global_step": [],
            "perplexity": [],
            "eval_loss": [],
            "train_loss": [],
            **extra_metrics,
        }

        return training_progress_scores

    def _get_last_metrics(self, metric_values):
        return {metric: values[-1] for metric, values in metric_values.items()}

    def save_model(
        self, output_dir=None, optimizer=None, scheduler=None, model=None, results=None
    ):
        if not self.is_world_master():
            return
        if not output_dir:
            output_dir = self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)

        if model and not self.args.no_save:
            # Take care of distributed/parallel training
            model_to_save = model.module if hasattr(model, "module") else model
            if self.args.model_type in "electra":
                os.makedirs(os.path.join(output_dir, "generator_config"), exist_ok=True)
                os.makedirs(
                    os.path.join(output_dir, "discriminator_config"), exist_ok=True
                )
                self.generator_config.save_pretrained(
                    os.path.join(output_dir, "generator_config")
                )
                self.discriminator_config.save_pretrained(
                    os.path.join(output_dir, "discriminator_config")
                )
            model_to_save.save_pretrained(output_dir)
            self.tokenizer.save_pretrained(output_dir)
            torch.save(self.args, os.path.join(output_dir, "training_args.bin"))
            if optimizer and scheduler and self.args.save_optimizer_and_scheduler:
                torch.save(
                    optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt")
                )
                torch.save(
                    scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt")
                )
            self.save_model_args(output_dir)

        if results:
            output_eval_file = os.path.join(output_dir, "eval_results.txt")
            with open(output_eval_file, "w") as writer:
                for key in sorted(results.keys()):
                    writer.write("{} = {}\n".format(key, str(results[key])))

    def save_model_args(self, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        self.args.save(output_dir)

    def _load_model_args(self, input_dir):
        args = LanguageModelingArgs()
        args.load(input_dir)
        return args

    def is_world_master(self) -> bool:
        """
        This will be True only in one process, even in distributed mode,
        even when training on multiple machines.
        """
        return self.args.local_rank == -1 or torch.distributed.get_rank() == 0

    def get_optimizer_parameters(self, model, args):
        no_decay = ["bias", "LayerNorm.weight"]

        optimizer_grouped_parameters = []
        custom_parameter_names = set()
        for group in self.args.custom_parameter_groups:
            params = group.pop("params")
            custom_parameter_names.update(params)
            param_group = {**group}
            param_group["params"] = [
                p for n, p in model.named_parameters() if n in params
            ]
            optimizer_grouped_parameters.append(param_group)

        for group in self.args.custom_layer_parameters:
            layer_number = group.pop("layer")
            layer = f"layer.{layer_number}."
            group_d = {**group}
            group_nd = {**group}
            group_nd["weight_decay"] = 0.0
            params_d = []
            params_nd = []
            for n, p in model.named_parameters():
                if n not in custom_parameter_names and layer in n:
                    if any(nd in n for nd in no_decay):
                        params_nd.append(p)
                    else:
                        params_d.append(p)
                    custom_parameter_names.add(n)
            group_d["params"] = params_d
            group_nd["params"] = params_nd

            optimizer_grouped_parameters.append(group_d)
            optimizer_grouped_parameters.append(group_nd)

        if not self.args.train_custom_parameters_only:
            optimizer_grouped_parameters.extend(
                [
                    {
                        "params": [
                            p
                            for n, p in model.named_parameters()
                            if n not in custom_parameter_names
                            and not any(nd in n for nd in no_decay)
                        ],
                        "weight_decay": args.weight_decay,
                    },
                    {
                        "params": [
                            p
                            for n, p in model.named_parameters()
                            if n not in custom_parameter_names
                            and any(nd in n for nd in no_decay)
                        ],
                        "weight_decay": 0.0,
                    },
                ]
            )

            # if self.args.use_autoencoder:
            #     optimizer_grouped_parameters.extend(
            #         [
            #             {
            #                 "params": [
            #                     p
            #                     for n, p in self.autoencoder_model.named_parameters()
            #                     if n not in custom_parameter_names
            #                     and not any(nd in n for nd in no_decay)
            #                 ],
            #                 "weight_decay": args.weight_decay,
            #             },
            #             {
            #                 "params": [
            #                     p
            #                     for n, p in self.autoencoder_model.named_parameters()
            #                     if n not in custom_parameter_names
            #                     and any(nd in n for nd in no_decay)
            #                 ],
            #                 "weight_decay": 0.0,
            #             },
            #         ]
            #     )

        return optimizer_grouped_parameters

    def get_named_parameters(self):
        return [n for n, p in self.model.named_parameters()]
