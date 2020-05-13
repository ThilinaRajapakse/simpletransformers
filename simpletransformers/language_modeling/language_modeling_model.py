#!/usr/bin/env python
# coding: utf-8


from __future__ import absolute_import, division, print_function

import json
import logging
import math
import os
import random
import warnings
from multiprocessing import cpu_count
from typing import Dict, List

import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    label_ranking_average_precision_score,
    matthews_corrcoef,
    mean_squared_error,
)
from tqdm.auto import tqdm, trange

import pandas as pd
import torch
from simpletransformers.config.global_args import global_args
from simpletransformers.custom_models.models import ElectraForLanguageModelingModel
from simpletransformers.language_modeling.language_modeling_utils import (
    LineByLineTextDataset,
    SimpleDataset,
    TextDataset,
    mask_tokens,
)
from tensorboardX import SummaryWriter
from tokenizers import BertWordPieceTokenizer, ByteLevelBPETokenizer
from tokenizers.processors import BertProcessing
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from transformers import (
    WEIGHTS_NAME,
    AdamW,
    AutoConfig,
    AutoTokenizer,
    AutoModelWithLMHead,
    BertConfig,
    BertForMaskedLM,
    BertTokenizer,
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
    OpenAIGPTConfig,
    OpenAIGPTLMHeadModel,
    OpenAIGPTTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    RobertaConfig,
    RobertaForMaskedLM,
    RobertaTokenizer,
    get_linear_schedule_with_warmup,
)
# for using TPU
import torch_xla
import torch_xla.core.xla_model as xm

try:
    import wandb

    wandb_available = True
except ImportError:
    wandb_available = False

logger = logging.getLogger(__name__)

MODEL_CLASSES = {
    "auto": (AutoConfig, AutoModelWithLMHead, AutoTokenizer),
    "bert": (BertConfig, BertForMaskedLM, BertTokenizer),
    "camembert": (CamembertConfig, CamembertForMaskedLM, CamembertTokenizer),
    "distilbert": (DistilBertConfig, DistilBertForMaskedLM, DistilBertTokenizer),
    "electra": (ElectraConfig, ElectraForLanguageModelingModel, ElectraTokenizer),
    "gpt2": (GPT2Config, GPT2LMHeadModel, GPT2Tokenizer),
    "openai-gpt": (OpenAIGPTConfig, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer),
    "roberta": (RobertaConfig, RobertaForMaskedLM, RobertaTokenizer),
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
            use_cuda (optional): Use GPU if available. Setting to False will force model to use CPU only.
            cuda_device (optional): Specific GPU that should be used. Will use the first available GPU by default.
            **kwargs (optional): For providing proxies, force_download, resume_download, cache_dir and other options specific to the 'from_pretrained' implementation where this will be supplied.
        """  # noqa: ignore flake8"

        if args and "manual_seed" in args:
            random.seed(args["manual_seed"])
            np.random.seed(args["manual_seed"])
            torch.manual_seed(args["manual_seed"])
            if "n_gpu" in args and args["n_gpu"] > 0:
                torch.cuda.manual_seed_all(args["manual_seed"])

        # to use this add "use_tpu": True to  train_args 
        if args['use_tpu']:
            print('using TPU')
            self.device = xm.xla_device()
        elif use_cuda:
            if torch.cuda.is_available():
                print('using GPU')
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
            print('using CPU')
            self.device = "cpu"

        self.results = {}

        self.args = {
            "dataset_type": "None",
            "dataset_class": None,
            "custom_tokenizer": None,
            "block_size": -1,
            "mlm": True,
            "mlm_probability": 0.15,
            "max_steps": -1,
            "config_name": None,
            "tokenizer_name": None,
            "min_frequency": 2,
            "special_tokens": ["<s>", "<pad>", "</s>", "<unk>", "<mask>"],
            "sliding_window": False,
            "stride": 0.8,
            "generator_config": {},
            "discriminator_config": {},
            "vocab_size": None,
        }

        self.args.update(global_args)

        saved_model_args = self._load_model_args(model_name)
        if saved_model_args:
            self.args.update(saved_model_args)

        if args:
            self.args.update(args)

        if not use_cuda:
            self.args["fp16"] = False

        if args:
            self.args.update(args)

        self.args["model_name"] = model_name
        self.args["model_type"] = model_type

        config_class, model_class, tokenizer_class = MODEL_CLASSES[model_type]
        self.tokenizer_class = tokenizer_class
        new_tokenizer = False

        if self.args["tokenizer_name"]:
            self.tokenizer = tokenizer_class.from_pretrained(
                self.args["tokenizer_name"], cache_dir=self.args["cache_dir"]
            )
        elif self.args["model_name"]:
            self.tokenizer = tokenizer_class.from_pretrained(model_name, cache_dir=self.args["cache_dir"], **kwargs)
            self.args["tokenizer_name"] = self.args["model_name"]
        else:
            if not train_files:
                raise ValueError(
                    "model_name and tokenizer_name are not specified."
                    "You must specify train_files to train a Tokenizer."
                )
            else:
                self.train_tokenizer(train_files)
                new_tokenizer = True

        if self.args["config_name"]:
            self.config = config_class.from_pretrained(self.args["config_name"], cache_dir=self.args["cache_dir"])
        elif self.args["model_name"]:
            self.config = config_class.from_pretrained(model_name, cache_dir=self.args["cache_dir"], **kwargs)
        else:
            self.config = config_class(**self.args["config"], **kwargs)
        if self.args["vocab_size"]:
            self.config.vocab_size = self.args["vocab_size"]
        if new_tokenizer:
            self.config.vocab_size = len(self.tokenizer)

        if self.args["model_type"] == "electra":
            if generator_name:
                self.generator_config = ElectraConfig.from_pretrained(generator_name)
            elif self.args["model_name"]:
                self.generator_config = ElectraConfig.from_pretrained(
                    os.path.join(self.args["model_name"], "generator_config"), **kwargs,
                )
            else:
                self.generator_config = ElectraConfig(**self.args["generator_config"], **kwargs)
                if new_tokenizer:
                    self.generator_config.vocab_size = len(self.tokenizer)

            if discriminator_name:
                self.discriminator_config = ElectraConfig.from_pretrained(discriminator_name)
            elif self.args["model_name"]:
                self.discriminator_config = ElectraConfig.from_pretrained(
                    os.path.join(self.args["model_name"], "discriminator_config"), **kwargs,
                )
            else:
                self.discriminator_config = ElectraConfig(**self.args["discriminator_config"], **kwargs)
                if new_tokenizer:
                    self.discriminator_config.vocab_size = len(self.tokenizer)

        if self.args["block_size"] <= 0:
            self.args["block_size"] = min(self.args["max_seq_length"], self.tokenizer.max_len)
        else:
            self.args["block_size"] = min(self.args["block_size"], self.tokenizer.max_len, self.args["max_seq_length"])

        if self.args["model_name"]:
            if self.args["model_type"] == "electra":
                self.model = model_class.from_pretrained(
                    model_name,
                    config=self.config,
                    cache_dir=self.args["cache_dir"],
                    generator_config=self.generator_config,
                    discriminator_config=self.discriminator_config,
                    **kwargs,
                )
                self.model.load_state_dict(torch.load(os.path.join(self.args["model_name"], "pytorch_model.bin")))
            else:
                self.model = model_class.from_pretrained(
                    model_name, config=self.config, cache_dir=self.args["cache_dir"], **kwargs,
                )
        else:
            logger.info(" Training language model from scratch")
            if self.args["model_type"] == "electra":
                generator_model = ElectraForMaskedLM(config=self.generator_config)
                discriminator_model = ElectraForPreTraining(config=self.discriminator_config)
                self.model = ElectraForLanguageModelingModel(
                    config=self.config,
                    generator_model=generator_model,
                    discriminator_model=discriminator_model,
                    generator_config=self.generator_config,
                    discriminator_config=self.discriminator_config,
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
                model_to_resize = self.model.module if hasattr(self.model, "module") else self.model
                model_to_resize.resize_token_embeddings(len(self.tokenizer))

        if model_type in ["camembert", "xlmroberta"]:
            warnings.warn(
                f"use_multiprocessing automatically disabled as {model_type}"
                " fails when using multiprocessing for feature conversion."
            )
            self.args["use_multiprocessing"] = False

        if self.args["wandb_project"] and not wandb_available:
            warnings.warn("wandb_project specified but wandb is not available. Wandb disabled.")
            self.args["wandb_project"] = None

    def train_model(
        self, train_file, output_dir=None, show_running_loss=True, args=None, eval_file=None, verbose=True, **kwargs,
    ):
        """
        Trains the model using 'train_file'

        Args:
            train_file: Path to text file containing the text to train the language model on.
            output_dir: The directory where model files will be saved. If not given, self.args['output_dir'] will be used.
            show_running_loss (optional): Set to False to prevent running loss from being printed to console. Defaults to True.
            args (optional): Optional changes to the args dict of the model. Any changes made will persist for the model.
            eval_file (optional): Path to eval file containing the text to evaluate the language model on.

        Returns:
            None
        """  # noqa: ignore flake8"

        if args:
            self.args.update(args)

        if self.args["silent"]:
            show_running_loss = False

        if self.args["evaluate_during_training"] and eval_file is None:
            raise ValueError(
                "evaluate_during_training is enabled but eval_file is not specified."
                " Pass eval_file to model.train_model() if using evaluate_during_training."
            )

        if not output_dir:
            output_dir = self.args["output_dir"]

        if os.path.exists(output_dir) and os.listdir(output_dir) and not self.args["overwrite_output_dir"]:
            raise ValueError(
                "Output directory ({}) already exists and is not empty."
                " Set args['overwrite_output_dir'] = True to overcome.".format(output_dir)
            )

        self._move_model_to_device()

        train_dataset = self.load_and_cache_examples(train_file, verbose=verbose)

        os.makedirs(output_dir, exist_ok=True)

        global_step, tr_loss = self.train(
            train_dataset,
            output_dir,
            show_running_loss=show_running_loss,
            eval_file=eval_file,
            verbose=verbose,
            **kwargs,
        )

        self._save_model(output_dir, model=self.model)
        if self.args["model_type"] == "electra":
            self.save_discriminator()
            self.save_generator()
        # model_to_save = self.model.module if hasattr(self.model, "module") else self.model
        # model_to_save.save_pretrained(output_dir)
        # self.tokenizer.save_pretrained(output_dir)
        # torch.save(self.args, os.path.join(output_dir, "training_args.bin"))

        if verbose:
            logger.info(" Training of {} model complete. Saved to {}.".format(self.args["model_type"], output_dir))

    def train(
        self, train_dataset, output_dir, show_running_loss=True, eval_file=None, verbose=True, **kwargs,
    ):
        """
        Trains the model on train_dataset.

        Utility function to be used by the train_model() method. Not intended to be used directly.
        """

        model = self.model
        args = self.args
        tokenizer = self.tokenizer

        def collate(examples: List[torch.Tensor]):
            if tokenizer._pad_token is None:
                return pad_sequence(examples, batch_first=True)
            return pad_sequence(examples, batch_first=True, padding_value=tokenizer.pad_token_id)

        tb_writer = SummaryWriter(logdir=args["tensorboard_dir"])
        train_sampler = RandomSampler(train_dataset)
        train_dataloader = DataLoader(
            train_dataset, sampler=train_sampler, batch_size=args["train_batch_size"], collate_fn=collate
        )

        if args["max_steps"] > 0:
            t_total = args["max_steps"]
            args["num_train_epochs"] = (
                args["max_steps"] // (len(train_dataloader) // args["gradient_accumulation_steps"]) + 1
            )
        else:
            t_total = len(train_dataloader) // args["gradient_accumulation_steps"] * args["num_train_epochs"]

        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": args["weight_decay"],
            },
            {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)]},
        ]

        warmup_steps = math.ceil(t_total * args["warmup_ratio"])
        args["warmup_steps"] = warmup_steps if args["warmup_steps"] == 0 else args["warmup_steps"]

        optimizer = AdamW(optimizer_grouped_parameters, lr=args["learning_rate"], eps=args["adam_epsilon"])
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=args["warmup_steps"], num_training_steps=t_total
        )

        if (
            args["model_name"]
            and os.path.isfile(os.path.join(args["model_name"], "optimizer.pt"))
            and os.path.isfile(os.path.join(args["model_name"], "scheduler.pt"))
        ):
            # Load in optimizer and scheduler states
            optimizer.load_state_dict(torch.load(os.path.join(args["model_name"], "optimizer.pt")))
            scheduler.load_state_dict(torch.load(os.path.join(args["model_name"], "scheduler.pt")))

        if args["fp16"]:
            try:
                from apex import amp
            except ImportError:
                raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")

            model, optimizer = amp.initialize(model, optimizer, opt_level=args["fp16_opt_level"])

        if args["n_gpu"] > 1:
            model = torch.nn.DataParallel(model)

        logger.info(" Training started")

        global_step = 0
        tr_loss, logging_loss = 0.0, 0.0
        model.zero_grad()
        train_iterator = trange(int(args["num_train_epochs"]), desc="Epoch", disable=args["silent"], mininterval=0)
        epoch_number = 0
        best_eval_metric = None
        early_stopping_counter = 0
        steps_trained_in_current_epoch = 0
        epochs_trained = 0

        if args["model_name"] and os.path.exists(args["model_name"]):
            try:
                # set global_step to gobal_step of last saved checkpoint from model path
                checkpoint_suffix = args["model_name"].split("/")[-1].split("-")
                if len(checkpoint_suffix) > 2:
                    checkpoint_suffix = checkpoint_suffix[1]
                else:
                    checkpoint_suffix = checkpoint_suffix[-1]
                global_step = int(checkpoint_suffix)
                epochs_trained = global_step // (len(train_dataloader) // args["gradient_accumulation_steps"])
                steps_trained_in_current_epoch = global_step % (
                    len(train_dataloader) // args["gradient_accumulation_steps"]
                )

                logger.info("   Continuing training from checkpoint, will skip to saved global_step")
                logger.info("   Continuing training from epoch %d", epochs_trained)
                logger.info("   Continuing training from global step %d", global_step)
                logger.info("   Will skip the first %d steps in the current epoch", steps_trained_in_current_epoch)
            except ValueError:
                logger.info("   Starting fine-tuning.")

        if args["evaluate_during_training"]:
            training_progress_scores = self._create_training_progress_scores(**kwargs)

        if args["wandb_project"]:
            wandb.init(project=args["wandb_project"], config={**args}, **args["wandb_kwargs"])
            wandb.watch(self.model)

        model.train()
        for current_epoch in train_iterator:
            if epochs_trained > 0:
                epochs_trained -= 1
                continue
            # epoch_iterator = tqdm(train_dataloader, desc="Iteration")
            for step, batch in enumerate(tqdm(train_dataloader, desc="Current iteration", disable=args["silent"])):
                if steps_trained_in_current_epoch > 0:
                    steps_trained_in_current_epoch -= 1
                    continue

                inputs, labels = mask_tokens(batch, tokenizer, args) if args["mlm"] else (batch, batch)
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                outputs = model(inputs, masked_lm_labels=labels) if args["mlm"] else model(inputs, labels=labels)
                # model outputs are always tuple in pytorch-transformers (see doc)
                loss = outputs[0]
                # if loss.item() < 1:
                #     masked = (labels[0] != -100).nonzero()
                #     print(labels[0][masked])
                #     preds = outputs[1][0, masked, :].clone().detach().cpu().numpy()
                #     print(np.argmax(preds, axis=2))

                if args["n_gpu"] > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu parallel training

                current_loss = loss.item()

                if show_running_loss:
                    print("\rRunning loss: %f" % loss, end="")

                if args["gradient_accumulation_steps"] > 1:
                    loss = loss / args["gradient_accumulation_steps"]

                if args["fp16"]:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()

                tr_loss += loss.item()
                if (step + 1) % args["gradient_accumulation_steps"] == 0:
                    if args["fp16"]:
                        torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args["max_grad_norm"])
                    else:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args["max_grad_norm"])

                   # Update parameters and take a step using the computed gradient
                    if args['use_tpu']:
                        xm.optimizer_step(optimizer, barrier=True)
                    else: 
                        optimizer.step()
                    scheduler.step()  # Update learning rate schedule
                    model.zero_grad()
                    global_step += 1

                    if args["logging_steps"] > 0 and global_step % args["logging_steps"] == 0:
                        # Log metrics
                        tb_writer.add_scalar("lr", scheduler.get_lr()[0], global_step)
                        tb_writer.add_scalar("loss", (tr_loss - logging_loss) / args["logging_steps"], global_step)
                        logging_loss = tr_loss
                        if args["wandb_project"]:
                            wandb.log(
                                {
                                    "Training loss": current_loss,
                                    "lr": scheduler.get_lr()[0],
                                    "global_step": global_step,
                                }
                            )

                    if args["save_steps"] > 0 and global_step % args["save_steps"] == 0:
                        # Save model checkpoint
                        output_dir_current = os.path.join(output_dir, "checkpoint-{}".format(global_step))

                        self._save_model(output_dir_current, optimizer, scheduler, model=model)

                    if args["evaluate_during_training"] and (
                        args["evaluate_during_training_steps"] > 0
                        and global_step % args["evaluate_during_training_steps"] == 0
                    ):
                        # Only evaluate when single GPU otherwise metrics may not average well
                        results = self.eval_model(
                            eval_file,
                            verbose=verbose and args["evaluate_during_training_verbose"],
                            silent=True,
                            **kwargs,
                        )
                        for key, value in results.items():
                            tb_writer.add_scalar("eval_{}".format(key), value, global_step)

                        output_dir_current = os.path.join(output_dir, "checkpoint-{}".format(global_step))

                        if args["save_eval_checkpoints"]:
                            self._save_model(output_dir_current, optimizer, scheduler, model=model, results=results)

                        training_progress_scores["global_step"].append(global_step)
                        training_progress_scores["train_loss"].append(current_loss)
                        for key in results:
                            training_progress_scores[key].append(results[key])
                        report = pd.DataFrame(training_progress_scores)
                        report.to_csv(
                            os.path.join(args["output_dir"], "training_progress_scores.csv"), index=False,
                        )

                        if args["wandb_project"]:
                            wandb.log(self._get_last_metrics(training_progress_scores))

                        if not best_eval_metric:
                            best_eval_metric = results[args["early_stopping_metric"]]
                            self._save_model(
                                args["best_model_dir"], optimizer, scheduler, model=model, results=results
                            )
                        if best_eval_metric and args["early_stopping_metric_minimize"]:
                            if (
                                results[args["early_stopping_metric"]] - best_eval_metric
                                < args["early_stopping_delta"]
                            ):
                                best_eval_metric = results[args["early_stopping_metric"]]
                                self._save_model(
                                    args["best_model_dir"], optimizer, scheduler, model=model, results=results
                                )
                                early_stopping_counter = 0
                            else:
                                if args["use_early_stopping"]:
                                    if early_stopping_counter < args["early_stopping_patience"]:
                                        early_stopping_counter += 1
                                        if verbose:
                                            logger.info(f" No improvement in {args['early_stopping_metric']}")
                                            logger.info(f" Current step: {early_stopping_counter}")
                                            logger.info(f" Early stopping patience: {args['early_stopping_patience']}")
                                    else:
                                        if verbose:
                                            logger.info(
                                                f" Patience of {args['early_stopping_patience']} steps reached."
                                            )
                                            logger.info(" Training terminated.")
                                            train_iterator.close()
                                        return global_step, tr_loss / global_step
                        else:
                            if (
                                results[args["early_stopping_metric"]] - best_eval_metric
                                > args["early_stopping_delta"]
                            ):
                                best_eval_metric = results[args["early_stopping_metric"]]
                                self._save_model(
                                    args["best_model_dir"], optimizer, scheduler, model=model, results=results
                                )
                                early_stopping_counter = 0
                            else:
                                if args["use_early_stopping"]:
                                    if early_stopping_counter < args["early_stopping_patience"]:
                                        early_stopping_counter += 1
                                        if verbose:
                                            logger.info(f" No improvement in {args['early_stopping_metric']}")
                                            logger.info(f" Current step: {early_stopping_counter}")
                                            logger.info(f" Early stopping patience: {args['early_stopping_patience']}")
                                    else:
                                        if verbose:
                                            logger.info(
                                                f" Patience of {args['early_stopping_patience']} steps reached."
                                            )
                                            logger.info(" Training terminated.")
                                            train_iterator.close()
                                        return global_step, tr_loss / global_step

                if args["max_steps"] > 0 and global_step > args["max_steps"]:
                    return global_step, tr_loss / global_step

            epoch_number += 1
            output_dir_current = os.path.join(output_dir, "checkpoint-{}-epoch-{}".format(global_step, epoch_number))

            if args["save_model_every_epoch"] or args["evaluate_during_training"]:
                os.makedirs(output_dir_current, exist_ok=True)

            if args["save_model_every_epoch"]:
                self._save_model(output_dir_current, optimizer, scheduler, model=model)

            if args["evaluate_during_training"]:
                results = self.eval_model(
                    eval_file, verbose=verbose and args["evaluate_during_training_verbose"], silent=True, **kwargs
                )

                self._save_model(output_dir_current, optimizer, scheduler, results=results)

                training_progress_scores["global_step"].append(global_step)
                training_progress_scores["train_loss"].append(current_loss)
                for key in results:
                    training_progress_scores[key].append(results[key])
                report = pd.DataFrame(training_progress_scores)
                report.to_csv(os.path.join(args["output_dir"], "training_progress_scores.csv"), index=False)

                if args["wandb_project"]:
                    wandb.log(self._get_last_metrics(training_progress_scores))

                if not best_eval_metric:
                    best_eval_metric = results[args["early_stopping_metric"]]
                    self._save_model(args["best_model_dir"], optimizer, scheduler, model=model, results=results)
                if best_eval_metric and args["early_stopping_metric_minimize"]:
                    if results[args["early_stopping_metric"]] - best_eval_metric < args["early_stopping_delta"]:
                        best_eval_metric = results[args["early_stopping_metric"]]
                        self._save_model(args["best_model_dir"], optimizer, scheduler, model=model, results=results)
                        early_stopping_counter = 0
                    else:
                        if args["use_early_stopping"] and args["early_stopping_consider_epochs"]:
                            if early_stopping_counter < args["early_stopping_patience"]:
                                early_stopping_counter += 1
                                if verbose:
                                    logger.info(f" No improvement in {args['early_stopping_metric']}")
                                    logger.info(f" Current step: {early_stopping_counter}")
                                    logger.info(f" Early stopping patience: {args['early_stopping_patience']}")
                            else:
                                if verbose:
                                    logger.info(f" Patience of {args['early_stopping_patience']} steps reached")
                                    logger.info(" Training terminated.")
                                    train_iterator.close()
                                return global_step, tr_loss / global_step
                else:
                    if results[args["early_stopping_metric"]] - best_eval_metric > args["early_stopping_delta"]:
                        best_eval_metric = results[args["early_stopping_metric"]]
                        self._save_model(args["best_model_dir"], optimizer, scheduler, model=model, results=results)
                        early_stopping_counter = 0
                    else:
                        if args["use_early_stopping"] and args["early_stopping_consider_epochs"]:
                            if early_stopping_counter < args["early_stopping_patience"]:
                                early_stopping_counter += 1
                                if verbose:
                                    logger.info(f" No improvement in {args['early_stopping_metric']}")
                                    logger.info(f" Current step: {early_stopping_counter}")
                                    logger.info(f" Early stopping patience: {args['early_stopping_patience']}")
                            else:
                                if verbose:
                                    logger.info(f" Patience of {args['early_stopping_patience']} steps reached")
                                    logger.info(" Training terminated.")
                                    train_iterator.close()
                                return global_step, tr_loss / global_step

            if args["max_steps"] > 0 and global_step > args["max_steps"]:
                return global_step, tr_loss / global_step

        return global_step, tr_loss / global_step

    def eval_model(self, eval_file, output_dir=None, verbose=True, silent=False, **kwargs):
        """
        Evaluates the model on eval_df. Saves results to outpuargs['output_dir']
            result: Dictionary containing evaluation results.
        """  # noqa: ignore flake8"

        if not output_dir:
            output_dir = self.args["output_dir"]

        self._move_model_to_device()

        eval_dataset = self.load_and_cache_examples(eval_file, evaluate=True, verbose=verbose, silent=silent)
        os.makedirs(output_dir, exist_ok=True)

        result = self.evaluate(eval_dataset, output_dir, verbose=verbose, silent=silent, **kwargs)
        self.results.update(result)

        if verbose:
            logger.info(self.results)

        return result

    def evaluate(self, eval_dataset, output_dir, multi_label=False, prefix="", verbose=True, silent=False, **kwargs):
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
            if tokenizer._pad_token is None:
                return pad_sequence(examples, batch_first=True)
            return pad_sequence(examples, batch_first=True, padding_value=tokenizer.pad_token_id)

        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(
            eval_dataset, sampler=eval_sampler, batch_size=args["eval_batch_size"], collate_fn=collate
        )

        if args["n_gpu"] > 1:
            model = torch.nn.DataParallel(model)

        eval_loss = 0.0
        nb_eval_steps = 0
        model.eval()

        for batch in tqdm(eval_dataloader, disable=args["silent"] or silent):
            inputs, labels = mask_tokens(batch, tokenizer, args) if args["mlm"] else (batch, batch)
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            with torch.no_grad():
                outputs = model(inputs, masked_lm_labels=labels) if args["mlm"] else model(inputs, labels=labels)
                lm_loss = outputs[0]
                eval_loss += lm_loss.mean().item()
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

    def load_and_cache_examples(self, file_path, evaluate=False, no_cache=False, verbose=True, silent=False):
        """
        Reads a text file from file_path and creates training features.

        Utility function for train() and eval() methods. Not intended to be used directly.
        """

        tokenizer = self.tokenizer
        args = self.args

        if not no_cache:
            no_cache = args["no_cache"]

        os.makedirs(self.args["cache_dir"], exist_ok=True)

        mode = "dev" if evaluate else "train"

        if args["dataset_class"]:
            CustomDataset = args["dataset_class"]
            return CustomDataset(tokenizer, args, file_path, mode, args["block_size"])
        else:
            dataset_type = args["dataset_type"]
            if dataset_type == "text":
                return TextDataset(tokenizer, args, file_path, mode, args["block_size"])
            elif dataset_type == "line_by_line":
                return LineByLineTextDataset(tokenizer, args, file_path, args["block_size"])
            else:
                special_tokens_count = 3 if bool(args["model_type"] in ["roberta", "camembert", "xlmroberta"]) else 2
                if self.args["max_seq_length"] > 509:
                    self.args["max_seq_length"] = (
                        509 if bool(args["model_type"] in ["roberta", "camembert", "xlmroberta"]) else 510
                    )
                    self.args["block_size"] = (
                        509 if bool(args["model_type"] in ["roberta", "camembert", "xlmroberta"]) else 510
                    )
                return SimpleDataset(
                    tokenizer,
                    self.args,
                    file_path,
                    mode,
                    args["block_size"],
                    special_tokens_count,
                    sliding_window=args["sliding_window"],
                )

    # def predict(self, to_predict, multi_label=False):
    #     """
    #     Performs predictions on a list of text.

    #     Args:
    #         to_predict: A python list of text (str) to be sent to the model for prediction.

    #     Returns:
    #         preds: A python list of the predictions (0 or 1) for each text.
    #         model_outputs: A python list of the raw model outputs for each text.
    #     """

    #     device = self.device
    #     model = self.model
    #     args = self.args

    #     self._move_model_to_device()

    #     if multi_label:
    #         eval_examples = [
    #             InputExample(i, text, None, [0 for i in range(self.num_labels)]) for i, text in enumerate(to_predict)
    #         ]
    #     else:
    #         if isinstance(to_predict[0], list):
    #             eval_examples = [InputExample(i, text[0], text[1], 0) for i, text in enumerate(to_predict)]
    #         else:
    #             eval_examples = [InputExample(i, text, None, 0) for i, text in enumerate(to_predict)]
    #     if args["sliding_window"]:
    #         eval_dataset, window_counts = self.load_and_cache_examples(eval_examples, evaluate=True, no_cache=True)
    #     else:
    #         eval_dataset = self.load_and_cache_examples(
    #             eval_examples, evaluate=True, multi_label=multi_label, no_cache=True
    #         )

    #     eval_sampler = SequentialSampler(eval_dataset)
    #     eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args["eval_batch_size"])

    #     eval_loss = 0.0
    #     nb_eval_steps = 0
    #     preds = None
    #     out_label_ids = None

    #     for batch in tqdm(eval_dataloader, disable=args["silent"]):
    #         model.eval()
    #         batch = tuple(t.to(device) for t in batch)

    #         with torch.no_grad():
    #             inputs = self._get_inputs_dict(batch)
    #             outputs = model(**inputs)
    #             tmp_eval_loss, logits = outputs[:2]

    #             if multi_label:
    #                 logits = logits.sigmoid()

    #             eval_loss += tmp_eval_loss.mean().item()

    #         nb_eval_steps += 1

    #         if preds is None:
    #             preds = logits.detach().cpu().numpy()
    #             out_label_ids = inputs["labels"].detach().cpu().numpy()
    #         else:
    #             preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
    #             out_label_ids = np.append(out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0)

    #     eval_loss = eval_loss / nb_eval_steps

    #     if args["sliding_window"]:
    #         count = 0
    #         window_ranges = []
    #         for n_windows in window_counts:
    #             window_ranges.append([count, count + n_windows])
    #             count += n_windows

    #         preds = [preds[window_range[0] : window_range[1]] for window_range in window_ranges]

    #         model_outputs = preds

    #         preds = [np.argmax(pred, axis=1) for pred in preds]
    #         final_preds = []
    #         for pred_row in preds:
    #             mode_pred, counts = mode(pred_row)
    #             if len(counts) > 1 and counts[0] == counts[1]:
    #                 final_preds.append(args["tie_value"])
    #             else:
    #                 final_preds.append(mode_pred[0])
    #         preds = np.array(final_preds)
    #     elif not multi_label and args["regression"] is True:
    #         preds = np.squeeze(preds)
    #         model_outputs = preds
    #     else:
    #         model_outputs = preds
    #         if multi_label:
    #             if isinstance(args["threshold"], list):
    #                 threshold_values = args["threshold"]
    #                 preds = [
    #                     [self._threshold(pred, threshold_values[i]) for i, pred in enumerate(example)]
    #                     for example in preds
    #                 ]
    #             else:
    #                 preds = [[self._threshold(pred, args["threshold"]) for pred in example] for example in preds]
    #         else:
    #             preds = np.argmax(preds, axis=1)

    #     return preds, model_outputs

    def train_tokenizer(self, train_files, tokenizer_name=None, output_dir=None, use_trained_tokenizer=True):
        """
        Train a new tokenizer on `train_files`.

        Args:

        - train_files: List of files to be used when training the tokenizer.

        - tokenizer_name: Name of a pretrained tokenizer or a path to a directory containing a tokenizer.

        - output_dir (optional): The directory where model files will be saved. If not given, self.args['output_dir']
        will be used.

        - use_trained_tokenizer (optional): Load the trained tokenizer once training completes.

        Returns: None
        """

        if not self.args["vocab_size"]:
            raise AttributeError(
                "Cannot train a new tokenizer as vocab_size is not specified in args dict. "
                "Either provide a tokenizer or specify vocab_size."
            )

        if not isinstance(train_files, list):
            train_files = [train_files]

        if not output_dir:
            output_dir = self.args["output_dir"]

        if self.args["model_type"] in ["bert", "electra"]:
            tokenizer = BertWordPieceTokenizer()
            self.args["special_tokens"] = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
            self.args["wordpieces_prefix"] = "##"

            tokenizer.train(
                files=train_files,
                vocab_size=self.args["vocab_size"],
                min_frequency=self.args["min_frequency"],
                special_tokens=self.args["special_tokens"],
                wordpieces_prefix="##",
            )
        else:
            tokenizer = ByteLevelBPETokenizer()

            tokenizer.train(
                files=train_files,
                vocab_size=self.args["vocab_size"],
                min_frequency=self.args["min_frequency"],
                special_tokens=self.args["special_tokens"],
            )

        os.makedirs(output_dir, exist_ok=True)

        tokenizer.save(output_dir)
        logger.info(" Training of {} tokenizer complete. Saved to {}.".format(tokenizer_name, output_dir))

        _, _, tokenizer_class = MODEL_CLASSES[self.args["model_type"]]
        tokenizer = tokenizer_class.from_pretrained(output_dir)

        if use_trained_tokenizer:
            self.tokenizer = tokenizer
            self.args["tokenizer_name"] = output_dir
            try:
                if self.args["model_type"] == "electra":
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

                model_to_resize = self.model.module if hasattr(self.model, "module") else self.model
                model_to_resize.resize_token_embeddings(len(self.tokenizer))
            except AttributeError:
                pass

    def save_discriminator(self, output_dir=None):
        if self.args["model_type"] == "electra":
            if not output_dir:
                output_dir = os.path.join(self.args["output_dir"], "discriminator_model")
            os.makedirs(output_dir, exist_ok=True)
            model_to_save = (
                self.model.discriminator_model.module
                if hasattr(self.model.discriminator_model, "module")
                else self.model.discriminator_model
            )
            model_to_save.save_pretrained(output_dir)
            self.tokenizer.save_pretrained(output_dir)
        else:
            raise ValueError("Model must be of ElectraForLanguageModelingModel")

    def save_generator(self, output_dir=None):
        if self.args["model_type"] == "electra":
            if not output_dir:
                output_dir = os.path.join(self.args["output_dir"], "generator_model")
            os.makedirs(output_dir, exist_ok=True)
            model_to_save = (
                self.model.generator_model.module
                if hasattr(self.model.generator_model, "module")
                else self.model.generator_model
            )
            model_to_save.save_pretrained(output_dir)
            self.tokenizer.save_pretrained(output_dir)
        else:
            raise ValueError("Model must be of ElectraForLanguageModelingModel")

    def _threshold(self, x, threshold):
        if x >= threshold:
            return 1
        return 0

    def _move_model_to_device(self):
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

    def _save_model(self, output_dir, optimizer=None, scheduler=None, model=None, results=None):
        os.makedirs(output_dir, exist_ok=True)

        if model:
            # Take care of distributed/parallel training
            model_to_save = model.module if hasattr(model, "module") else model
            if self.args["model_type"] in "electra":
                os.makedirs(os.path.join(output_dir, "generator_config"), exist_ok=True)
                os.makedirs(os.path.join(output_dir, "discriminator_config"), exist_ok=True)
                self.generator_config.save_pretrained(os.path.join(output_dir, "generator_config"))
                self.discriminator_config.save_pretrained(os.path.join(output_dir, "discriminator_config"))
            model_to_save.save_pretrained(output_dir)
            self.tokenizer.save_pretrained(output_dir)
            torch.save(self.args, os.path.join(output_dir, "training_args.bin"))
            if optimizer:
                torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
            if scheduler:
                torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
            self._save_model_args(output_dir)

        if results:
            output_eval_file = os.path.join(output_dir, "eval_results.txt")
            with open(output_eval_file, "w") as writer:
                for key in sorted(results.keys()):
                    writer.write("{} = {}\n".format(key, str(results[key])))

    def _save_model_args(self, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "model_args.json"), "w") as f:
            json.dump(self.args, f)

    def _load_model_args(self, input_dir):
        if input_dir:
            input_dir, filename = os.path.split(input_dir)

            model_args_file = os.path.join(input_dir, "model_args.json")
            if os.path.isfile(model_args_file):
                with open(model_args_file, "r") as f:
                    model_args = json.load(f)
                return model_args
