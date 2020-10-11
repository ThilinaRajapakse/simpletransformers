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

import numpy as np
import pandas as pd
import torch
from scipy.stats import mode, pearsonr
from sklearn.metrics import (
    confusion_matrix,
    label_ranking_average_precision_score,
    matthews_corrcoef,
    mean_squared_error,
)
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from tqdm.auto import tqdm, trange
from transformers import (
    BERT_PRETRAINED_MODEL_ARCHIVE_LIST,
    WEIGHTS_NAME,
    AdamW,
    BertConfig,
    BertModel,
    BertTokenizer,
    get_linear_schedule_with_warmup,
)
from transformers.configuration_mmbt import MMBTConfig

from simpletransformers.classification.classification_utils import (
    ImageEncoder,
    InputExample,
    JsonlDataset,
    collate_fn,
    convert_examples_to_features,
    get_image_transforms,
)
from simpletransformers.classification.transformer_models.mmbt_model import MMBTForClassification
from simpletransformers.config.global_args import global_args
from simpletransformers.config.model_args import MultiModalClassificationArgs

try:
    import wandb

    wandb_available = True
except ImportError:
    wandb_available = False


logger = logging.getLogger(__name__)


class MultiModalClassificationModel:
    def __init__(
        self,
        model_type,
        model_name,
        multi_label=False,
        label_list=None,
        num_labels=None,
        pos_weight=None,
        args=None,
        use_cuda=True,
        cuda_device=-1,
        **kwargs,
    ):

        """
        Initializes a MultiModalClassificationModel model.

        Args:
            model_type: The type of model (bert, xlnet, xlm, roberta, distilbert, albert)
            model_name: Default Transformer model name or path to a directory containing Transformer model file (pytorch_nodel.bin).
            multi_label (optional): Set to True for multi label tasks.
            label_list (optional) : A list of all the labels (str) in the dataset.
            num_labels (optional): The number of labels or classes in the dataset.
            pos_weight (optional): A list of length num_labels containing the weights to assign to each label for loss calculation.
            args (optional): Default args will be used if this parameter is not provided. If provided, it should be a dict containing the args that should be changed in the default args.
            use_cuda (optional): Use GPU if available. Setting to False will force model to use CPU only.
            cuda_device (optional): Specific GPU that should be used. Will use the first available GPU by default.
            **kwargs (optional): For providing proxies, force_download, resume_download, cache_dir and other options specific to the 'from_pretrained' implementation where this will be supplied.
        """  # noqa: ignore flake8"

        MODEL_CLASSES = {
            "bert": (BertConfig, BertModel, BertTokenizer),
        }

        self.args = self._load_model_args(model_name)

        if isinstance(args, dict):
            self.args.update_from_dict(args)
        elif isinstance(args, MultiModalClassificationArgs):
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

        self.label_list = label_list
        if self.label_list and not num_labels:
            num_labels = len(self.label_list)
        elif self.label_list and num_labels:
            if len(self.label_list) != num_labels:
                raise ValueError(f"Mismatch in num_labels ({num_labels}) and length of label_list ({len(label_list)})")

        if num_labels and not self.label_list:
            self.label_list = [str(i) for i in range(num_labels)]

        if num_labels:
            self.transformer_config = config_class.from_pretrained(model_name, num_labels=num_labels, **kwargs)
            self.num_labels = num_labels
        else:
            self.transformer_config = config_class.from_pretrained(model_name, **kwargs)
            self.num_labels = self.transformer_config.num_labels

        self.multi_label = multi_label

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

        self.transformer = model_class.from_pretrained(model_name, config=self.transformer_config, **kwargs)
        self.config = MMBTConfig(self.transformer_config, num_labels=self.num_labels)
        self.results = {}

        self.img_encoder = ImageEncoder(self.args)
        self.model = MMBTForClassification(self.config, self.transformer, self.img_encoder)

        if model_name not in BERT_PRETRAINED_MODEL_ARCHIVE_LIST:
            try:
                self.model.load_state_dict(torch.load(os.path.join(model_name, "pytorch_model.bin")))
            except EnvironmentError:
                msg = (
                    "Model name '{}' was not found in model name list ({}). "
                    "We assumed '{}' was a path or url to model weight files named one of {} but "
                    "couldn't find any such file at this path or url.".format(
                        model_name, ", ".join(BERT_PRETRAINED_MODEL_ARCHIVE_LIST), model_name, "pytorch_model.bin",
                    )
                )
                raise EnvironmentError(msg)

        self.tokenizer = tokenizer_class.from_pretrained(model_name, do_lower_case=self.args.do_lower_case, **kwargs)

        self.args.model_name = model_name
        self.args.model_type = model_type

        if model_type in ["camembert", "xlmroberta"]:
            warnings.warn(
                f"use_multiprocessing automatically disabled as {model_type}"
                " fails when using multiprocessing for feature conversion."
            )
            self.args.use_multiprocessing = False

        self.args.model_name = model_name
        self.args.model_type = model_type

        if model_type in ["camembert", "xlmroberta"]:
            warnings.warn(
                f"use_multiprocessing automatically disabled as {model_type}"
                " fails when using multiprocessing for feature conversion."
            )
            self.args.use_multiprocessing = False

        if self.args.wandb_project and not wandb_available:
            warnings.warn("wandb_project specified but wandb is not available. Wandb disabled.")
            self.args.wandb_project = None

        if multi_label:
            self.criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        elif self.num_labels == 1:
            self.criterion = torch.nn.MSELoss()
        else:
            self.criterion = torch.nn.CrossEntropyLoss()

    def train_model(
        self,
        train_data,
        files_list=None,
        image_path=None,
        text_label=None,
        labels_label=None,
        images_label=None,
        image_type_extension=None,
        data_type_extension=None,
        auto_weights=False,
        output_dir=None,
        show_running_loss=True,
        args=None,
        eval_data=None,
        verbose=True,
        **kwargs,
    ):
        """
        Trains the model using 'train_df'

        Args:
            data: Path to data directory containing text files (JSON) and image files OR a Pandas DataFrame.
                If a DataFrame is given, it should contain the columns [text, labels, images]. When using a DataFrame,
                image_path MUST be specified. The image column of the DataFrame should contain the relative path from
                image_path to the image.
                E.g:
                    For an image file 1.jpeg located in "data/train/";
                        image_path = "data/train/"
                        images = "1.jpeg"
            files_list (optional): If given, only the files specified in this list will be taken from data directory.
                files_list can be a Python list or the path (str) to a JSON file containing a list of files.
            image_path (optional): Must be specified when using DataFrame as input. Path to the directory containing the
                images.
            text_label (optional): Column name to look for instead of the default "text"
            labels_label (optional): Column name to look for instead of the default "labels"
            images_label (optional): Column name to look for instead of the default "images"
            image_type_extension (optional): If given, this will be added to the end of each value in "images".
            data_type_extension (optional): If given, this will be added to the end of each value in "files_list".
            auto_weights (optional): If True, weights will be used to balance the classes. Only implemented for multi label tasks currently.
            output_dir: The directory where model files will be saved. If not given, self.args.output_dir will be used.
            show_running_loss (optional): Set to False to prevent running loss from being printed to console. Defaults to True.
            args (optional): Optional changes to the args dict of the model. Any changes made will persist for the model.
            eval_data (optional): A DataFrame against which evaluation will be performed when evaluate_during_training is enabled. Is required if evaluate_during_training is enabled.
            **kwargs: Additional metrics that should be used. Pass in the metrics as keyword arguments (name of metric: function to use). E.g. f1=sklearn.metrics.f1_score.
                        A metric function should take in two parameters. The first parameter will be the true labels, and the second parameter will be the predictions.

        Returns:
            global_step: Number of global steps trained
            training_details: Average training loss if evaluate_during_training is False or full training progress scores if evaluate_during_training is True
        """  # noqa: ignore flake8"

        if args:
            self.args.update_from_dict(args)

        if text_label:
            self.args.text_label = text_label

        if text_label:
            self.args.labels_label = labels_label

        if text_label:
            self.args.images_label = images_label

        if text_label:
            self.args.image_type_extension = image_type_extension

        if text_label:
            self.args.data_type_extension = data_type_extension

        if self.args.silent:
            show_running_loss = False

        if self.args.evaluate_during_training and eval_data is None:
            raise ValueError(
                "evaluate_during_training is enabled but eval_df is not specified."
                " Pass eval_df to model.train_model() if using evaluate_during_training."
            )

        if not output_dir:
            output_dir = self.args.output_dir

        if os.path.exists(output_dir) and os.listdir(output_dir) and not self.args.overwrite_output_dir:
            raise ValueError(
                "Output directory ({}) already exists and is not empty."
                " Set overwrite_output_dir to True to overcome.".format(output_dir)
            )

        self._move_model_to_device()

        train_dataset = self.load_and_cache_examples(
            train_data,
            files_list=files_list,
            image_path=image_path,
            text_label=self.args.text_label,
            labels_label=self.args.labels_label,
            images_label=self.args.images_label,
            image_type_extension=self.args.image_type_extension,
            data_type_extension=self.args.data_type_extension,
            verbose=verbose,
        )

        if auto_weights:
            if self.multi_label:
                self.criterion = torch.nn.BCEWithLogitsLoss(pos_weight=self.calculate_weights(train_dataset))

        os.makedirs(output_dir, exist_ok=True)

        global_step, training_details = self.train(
            train_dataset,
            output_dir,
            files_list=files_list,
            image_path=image_path,
            text_label=text_label,
            labels_label=labels_label,
            images_label=images_label,
            image_type_extension=image_type_extension,
            data_type_extension=data_type_extension,
            show_running_loss=show_running_loss,
            eval_data=eval_data,
            verbose=verbose,
            **kwargs,
        )

        self.save_model(output_dir, model=self.model)

        if verbose:
            logger.info(" Training of {} model complete. Saved to {}.".format(self.args.model_type, output_dir))

        return global_step, training_details

    def train(
        self,
        train_dataset,
        output_dir,
        files_list=None,
        image_path=None,
        text_label=None,
        labels_label=None,
        images_label=None,
        image_type_extension=None,
        data_type_extension=None,
        show_running_loss=True,
        eval_data=None,
        verbose=True,
        **kwargs,
    ):
        """
        Trains the model on train_dataset.

        Utility function to be used by the train_model() method. Not intended to be used directly.
        """

        device = self.device
        model = self.model
        args = self.args
        multi_label = self.multi_label

        tb_writer = SummaryWriter(logdir=args.tensorboard_dir)
        train_sampler = RandomSampler(train_dataset)
        train_dataloader = DataLoader(
            train_dataset,
            sampler=train_sampler,
            batch_size=args.train_batch_size,
            collate_fn=collate_fn,
            num_workers=self.args.dataloader_num_workers,
        )

        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

        no_decay = ["bias", "LayerNorm.weight"]

        optimizer_grouped_parameters = []
        custom_parameter_names = set()
        for group in self.args.custom_parameter_groups:
            params = group.pop("params")
            custom_parameter_names.update(params)
            param_group = {**group}
            param_group["params"] = [p for n, p in model.named_parameters() if n in params]
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
                            if n not in custom_parameter_names and not any(nd in n for nd in no_decay)
                        ],
                        "weight_decay": args.weight_decay,
                    },
                    {
                        "params": [
                            p
                            for n, p in model.named_parameters()
                            if n not in custom_parameter_names and any(nd in n for nd in no_decay)
                        ],
                        "weight_decay": 0.0,
                    },
                ]
            )

        warmup_steps = math.ceil(t_total * args.warmup_ratio)
        args.warmup_steps = warmup_steps if args.warmup_steps == 0 else args.warmup_steps

        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
        )

        if args.n_gpu > 1:
            model = torch.nn.DataParallel(model)

        global_step = 0
        training_progress_scores = None
        tr_loss, logging_loss = 0.0, 0.0
        model.zero_grad()
        train_iterator = trange(int(args.num_train_epochs), desc="Epoch", disable=args.silent)
        epoch_number = 0
        best_eval_metric = None
        early_stopping_counter = 0

        if args.evaluate_during_training:
            training_progress_scores = self._create_training_progress_scores(multi_label, **kwargs)

        if args.wandb_project:
            wandb.init(project=args.wandb_project, config={**asdict(args)}, **args.wandb_kwargs)
            wandb.watch(self.model)

        if args.fp16:
            from torch.cuda import amp

            scaler = amp.GradScaler()

        for _ in train_iterator:
            model.train()
            train_iterator.set_description(f"Epoch {epoch_number} of {args.num_train_epochs}")
            train_iterator.set_description(f"Epoch {epoch_number + 1} of {args.num_train_epochs}")
            batch_iterator = tqdm(
                train_dataloader,
                desc=f"Running Epoch {epoch_number} of {args.num_train_epochs}",
                disable=args.silent,
                mininterval=0,
            )
            for step, batch in enumerate(batch_iterator):
                batch = tuple(t.to(device) for t in batch)
                labels = batch[5]

                inputs = self._get_inputs_dict(batch)
                if args.fp16:
                    with amp.autocast():
                        outputs = model(**inputs)
                        # model outputs are always tuple in pytorch-transformers (see doc)
                        loss = outputs[0]
                else:
                    outputs = model(**inputs)
                    # model outputs are always tuple in pytorch-transformers (see doc)
                    logits = outputs[0]  # Different from default behaviour
                    loss = self.criterion(logits, labels)

                if args.n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu parallel training

                current_loss = loss.item()

                if show_running_loss:
                    batch_iterator.set_description(
                        f"Epochs {epoch_number}/{args.num_train_epochs}. Running Loss: {current_loss:9.4f}"
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
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

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
                        tb_writer.add_scalar("lr", scheduler.get_lr()[0], global_step)
                        tb_writer.add_scalar("loss", (tr_loss - logging_loss) / args.logging_steps, global_step)
                        logging_loss = tr_loss
                        if args.wandb_project:
                            wandb.log(
                                {
                                    "Training loss": current_loss,
                                    "lr": scheduler.get_lr()[0],
                                    "global_step": global_step,
                                }
                            )

                    if args.save_steps > 0 and global_step % args.save_steps == 0:
                        # Save model checkpoint
                        output_dir_current = os.path.join(output_dir, "checkpoint-{}".format(global_step))

                        self.save_model(output_dir_current, model=model)

                    if args.evaluate_during_training and (
                        args.evaluate_during_training_steps > 0
                        and global_step % args.evaluate_during_training_steps == 0
                    ):
                        # Only evaluate when single GPU otherwise metrics may not average well
                        results, _ = self.eval_model(
                            eval_data,
                            files_list=files_list,
                            image_path=image_path,
                            text_label=text_label,
                            labels_label=labels_label,
                            images_label=images_label,
                            image_type_extension=image_type_extension,
                            data_type_extension=data_type_extension,
                            verbose=verbose and args.evaluate_during_training_verbose,
                            silent=args.evaluate_during_training_silent,
                            **kwargs,
                        )
                        for key, value in results.items():
                            tb_writer.add_scalar("eval_{}".format(key), value, global_step)

                        output_dir_current = os.path.join(output_dir, "checkpoint-{}".format(global_step))

                        if args.save_eval_checkpoints:
                            self.save_model(output_dir_current, model=model, results=results)

                        training_progress_scores["global_step"].append(global_step)
                        training_progress_scores["train_loss"].append(current_loss)
                        for key in results:
                            training_progress_scores[key].append(results[key])
                        report = pd.DataFrame(training_progress_scores)
                        report.to_csv(
                            os.path.join(args.output_dir, "training_progress_scores.csv"), index=False,
                        )

                        if args.wandb_project:
                            wandb.log(self._get_last_metrics(training_progress_scores))

                        if not best_eval_metric:
                            best_eval_metric = results[args.early_stopping_metric]
                            self.save_model(args.best_model_dir, model=model, results=results)
                        if best_eval_metric and args.early_stopping_metric_minimize:
                            if results[args.early_stopping_metric] - best_eval_metric < args.early_stopping_delta:
                                best_eval_metric = results[args.early_stopping_metric]
                                self.save_model(args.best_model_dir, model=model, results=results)
                                early_stopping_counter = 0
                            else:
                                if args.use_early_stopping:
                                    if early_stopping_counter < args.early_stopping_patience:
                                        early_stopping_counter += 1
                                        if verbose:
                                            logger.info(f" No improvement in {args.early_stopping_metric}")
                                            logger.info(f" Current step: {early_stopping_counter}")
                                            logger.info(f" Early stopping patience: {args.early_stopping_patience}")
                                    else:
                                        if verbose:
                                            logger.info(f" Patience of {args.early_stopping_patience} steps reached")
                                            logger.info(" Training terminated.")
                                            train_iterator.close()
                                        return (
                                            global_step,
                                            tr_loss / global_step
                                            if not self.args.evaluate_during_training
                                            else training_progress_scores,
                                        )
                        else:
                            if results[args.early_stopping_metric] - best_eval_metric > args.early_stopping_delta:
                                best_eval_metric = results[args.early_stopping_metric]
                                self.save_model(args.best_model_dir, model=model, results=results)
                                early_stopping_counter = 0
                            else:
                                if args.use_early_stopping:
                                    if early_stopping_counter < args.early_stopping_patience:
                                        early_stopping_counter += 1
                                        if verbose:
                                            logger.info(f" No improvement in {args.early_stopping_metric}")
                                            logger.info(f" Current step: {early_stopping_counter}")
                                            logger.info(f" Early stopping patience: {args.early_stopping_patience}")
                                    else:
                                        if verbose:
                                            logger.info(f" Patience of {args.early_stopping_patience} steps reached")
                                            logger.info(" Training terminated.")
                                            train_iterator.close()
                                        return (
                                            global_step,
                                            tr_loss / global_step
                                            if not self.args.evaluate_during_training
                                            else training_progress_scores,
                                        )

            epoch_number += 1
            output_dir_current = os.path.join(output_dir, "checkpoint-{}-epoch-{}".format(global_step, epoch_number))

            if args.save_model_every_epoch or args.evaluate_during_training:
                os.makedirs(output_dir_current, exist_ok=True)

            if args.save_model_every_epoch:
                self.save_model(output_dir_current, model=model)

            if args.evaluate_during_training and args.evaluate_each_epoch:
                results, _ = self.eval_model(
                    eval_data,
                    files_list=files_list,
                    image_path=image_path,
                    text_label=text_label,
                    labels_label=labels_label,
                    images_label=images_label,
                    image_type_extension=image_type_extension,
                    data_type_extension=data_type_extension,
                    verbose=verbose and args.evaluate_during_training_verbose,
                    silent=args.evaluate_during_training_silent,
                    **kwargs,
                )

                self.save_model(output_dir_current, results=results)

                training_progress_scores["global_step"].append(global_step)
                training_progress_scores["train_loss"].append(current_loss)
                for key in results:
                    training_progress_scores[key].append(results[key])
                report = pd.DataFrame(training_progress_scores)
                report.to_csv(
                    os.path.join(args.output_dir, "training_progress_scores.csv"), index=False,
                )

                if not best_eval_metric:
                    best_eval_metric = results[args.early_stopping_metric]
                    self.save_model(args.best_model_dir, model=model, results=results)
                if best_eval_metric and args.early_stopping_metric_minimize:
                    if results[args.early_stopping_metric] - best_eval_metric < args.early_stopping_delta:
                        best_eval_metric = results[args.early_stopping_metric]
                        self.save_model(args.best_model_dir, model=model, results=results)
                        early_stopping_counter = 0
                    else:
                        if args.use_early_stopping and args.early_stopping_consider_epochs:
                            if early_stopping_counter < args.early_stopping_patience:
                                early_stopping_counter += 1
                                if verbose:
                                    logger.info(f" No improvement in {args.early_stopping_metric}")
                                    logger.info(f" Current step: {early_stopping_counter}")
                                    logger.info(f" Early stopping patience: {args.early_stopping_patience}")
                            else:
                                if verbose:
                                    logger.info(f" Patience of {args.early_stopping_patience} steps reached")
                                    logger.info(" Training terminated.")
                                    train_iterator.close()
                                return (
                                    global_step,
                                    tr_loss / global_step
                                    if not self.args.evaluate_during_training
                                    else training_progress_scores,
                                )
                else:
                    if results[args.early_stopping_metric] - best_eval_metric > args.early_stopping_delta:
                        best_eval_metric = results[args.early_stopping_metric]
                        self.save_model(args.best_model_dir, model=model, results=results)
                        early_stopping_counter = 0
                    else:
                        if args.use_early_stopping and args.early_stopping_consider_epochs:
                            if early_stopping_counter < args.early_stopping_patience:
                                early_stopping_counter += 1
                                if verbose:
                                    logger.info(f" No improvement in {args.early_stopping_metric}")
                                    logger.info(f" Current step: {early_stopping_counter}")
                                    logger.info(f" Early stopping patience: {args.early_stopping_patience}")
                            else:
                                if verbose:
                                    logger.info(f" Patience of {args.early_stopping_patience} steps reached")
                                    logger.info(" Training terminated.")
                                    train_iterator.close()
                                return (
                                    global_step,
                                    tr_loss / global_step
                                    if not self.args.evaluate_during_training
                                    else training_progress_scores,
                                )

        return (
            global_step,
            tr_loss / global_step if not self.args.evaluate_during_training else training_progress_scores,
        )

    def eval_model(
        self,
        data,
        files_list=None,
        image_path=None,
        text_label=None,
        labels_label=None,
        images_label=None,
        image_type_extension=None,
        data_type_extension=None,
        output_dir=None,
        verbose=True,
        silent=False,
        **kwargs,
    ):
        """
        Evaluates the model on eval_df. Saves results to output_dir.

        Args:
            data: Path to data directory containing text files (JSON) and image files OR a Pandas DataFrame.
                If a DataFrame is given, it should contain the columns [text, labels, images]. When using a DataFrame,
                image_path MUST be specified. The image column of the DataFrame should contain the relative path from
                image_path to the image.
                E.g:
                    For an image file 1.jpeg located in "data/train/";
                        image_path = "data/train/"
                        images = "1.jpeg"
            files_list (optional): If given, only the files specified in this list will be taken from data directory.
                files_list can be a Python list or the path (str) to a JSON file containing a list of files.
            image_path (optional): Must be specified when using DataFrame as input. Path to the directory containing the
                images.
            text_label (optional): Column name to look for instead of the default "text"
            labels_label (optional): Column name to look for instead of the default "labels"
            images_label (optional): Column name to look for instead of the default "images"
            image_type_extension (optional): If given, this will be added to the end of each value in "images".
            data_type_extension (optional): If given, this will be added to the end of each value in "files_list".
            output_dir: The directory where model files will be saved. If not given, self.args.output_dir will be used.
            verbose: If verbose, results will be printed to the console on completion of evaluation.
            silent: If silent, tqdm progress bars will be hidden.
            **kwargs: Additional metrics that should be used. Pass in the metrics as keyword arguments (name of metric: function to use). E.g. f1=sklearn.metrics.f1_score.
                        A metric function should take in two parameters. The first parameter will be the true labels, and the second parameter will be the predictions.

        Returns:
            result: Dictionary containing evaluation results. (Matthews correlation coefficient, tp, tn, fp, fn)
            model_outputs: List of model outputs for each row in eval_df
        """  # noqa: ignore flake8"

        if text_label:
            self.args.text_label = text_label

        if text_label:
            self.args.labels_label = labels_label

        if text_label:
            self.args.images_label = images_label

        if text_label:
            self.args.image_type_extension = image_type_extension

        if text_label:
            self.args.data_type_extension = data_type_extension

        if not output_dir:
            output_dir = self.args.output_dir

        self._move_model_to_device()

        # If data is a tuple,
        # this is for early stopping and first element is data_path and second element is files_list
        if isinstance(data, tuple):
            data, files_list = data

        eval_dataset = self.load_and_cache_examples(
            data,
            files_list=files_list,
            image_path=image_path,
            text_label=self.args.text_label,
            labels_label=self.args.labels_label,
            images_label=self.args.images_label,
            image_type_extension=self.args.image_type_extension,
            data_type_extension=self.args.data_type_extension,
            verbose=verbose,
            silent=silent,
        )
        os.makedirs(output_dir, exist_ok=True)

        result, model_outputs = self.evaluate(eval_dataset, output_dir, verbose=verbose, silent=silent, **kwargs)
        self.results.update(result)

        if verbose:
            logger.info(self.results)

        return result, model_outputs

    def evaluate(
        self, eval_dataset, output_dir, prefix="", verbose=True, silent=False, **kwargs,
    ):
        """
        Evaluates the model on eval_df.

        Utility function to be used by the eval_model() method. Not intended to be used directly.
        """

        device = self.device
        model = self.model
        args = self.args
        multi_label = self.multi_label
        eval_output_dir = output_dir

        results = {}

        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(
            eval_dataset,
            sampler=eval_sampler,
            batch_size=args.eval_batch_size,
            collate_fn=collate_fn,
            num_workers=self.args.dataloader_num_workers,
        )

        eval_loss = 0.0
        nb_eval_steps = 0
        preds = None
        out_label_ids = None
        model.eval()

        if args.n_gpu > 1:
            model = torch.nn.DataParallel(model)

        if args.fp16:
            from torch.cuda import amp

        for batch in tqdm(eval_dataloader, disable=args.silent or silent, desc="Running Evaluation"):
            batch = tuple(t.to(device) for t in batch)
            labels = batch[5]
            with torch.no_grad():
                inputs = self._get_inputs_dict(batch)

                outputs = model(**inputs)
                logits = outputs[0]  # Different from default behaviour
                tmp_eval_loss = self.criterion(logits, labels)

                if self.args.n_gpu > 1:
                    tmp_eval_loss = tmp_eval_loss.mean()
                eval_loss += tmp_eval_loss.item()

            nb_eval_steps += 1

            if preds is None:
                preds = torch.sigmoid(logits).detach().cpu().numpy()
                out_label_ids = labels.detach().cpu().numpy()
            else:
                preds = np.append(preds, torch.sigmoid(logits).detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(out_label_ids, labels.detach().cpu().numpy(), axis=0)

        eval_loss = eval_loss / nb_eval_steps
        model_outputs = preds

        if args.regression is True:
            preds = np.squeeze(preds)
            model_outputs = preds

        model_outputs = preds
        if multi_label:
            preds = (preds > 0.5).astype(int)
        else:
            preds = np.argmax(preds, axis=1)

        result = self.compute_metrics(preds, out_label_ids, **kwargs)
        result["eval_loss"] = eval_loss
        results.update(result)

        output_eval_file = os.path.join(eval_output_dir, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            for key in sorted(result.keys()):
                writer.write("{} = {}\n".format(key, str(result[key])))

        return results, model_outputs

    def load_and_cache_examples(
        self,
        data,
        files_list=None,
        image_path=None,
        text_label=None,
        labels_label=None,
        images_label=None,
        image_type_extension=None,
        data_type_extension=None,
        evaluate=False,
        no_cache=False,
        verbose=True,
        silent=False,
    ):
        """
        Converts a list of InputExample objects to a TensorDataset containing InputFeatures. Caches the InputFeatures.

        Args:
            data: Path to data directory containing text files (JSON) and image files OR a Pandas DataFrame.
                If a DataFrame is given, it should contain the columns [text, labels, images]. When using a DataFrame,
                image_path MUST be specified. The image column of the DataFrame should contain the relative path from
                image_path to the image.
                E.g:
                    For an image file 1.jpeg located in "data/train/";
                        image_path = "data/train/"
                        images = "1.jpeg"
            files_list (optional): If given, only the files specified in this list will be taken from data directory.
                files_list can be a Python list or the path (str) to a JSON file containing a list of files.
            image_path (optional): Must be specified when using DataFrame as input. Path to the directory containing the
                images.
            text_label (optional): Column name to look for instead of the default "text"
            labels_label (optional): Column name to look for instead of the default "labels"
            images_label (optional): Column name to look for instead of the default "images"
            image_type_extension (optional): If given, this will be added to the end of each value in "images".
            data_type_extension (optional): If given, this will be added to the end of each value in "files_list".

        Utility function for train() and eval() methods. Not intended to be used directly.
        """  # noqa: ignore flake8"

        tokenizer = self.tokenizer
        args = self.args

        if not isinstance(data, str):
            if not image_path:
                raise ValueError(
                    "data is not a str and image_path is not given. image_path must be specified when input is a DF"
                )
            else:
                data = data.rename(columns={text_label: "text", labels_label: "labels", images_label: "images"})

        transforms = get_image_transforms()

        if self.label_list:
            labels = self.label_list
        else:
            labels = [str(i) for i in range(self.num_labels)]

        dataset = JsonlDataset(
            data,
            tokenizer,
            transforms,
            labels,
            args.max_seq_length - args.num_image_embeds - 2,
            files_list=files_list,
            image_path=image_path,
            text_label=text_label,
            labels_label=labels_label,
            images_label=images_label,
            image_type_extension=image_type_extension,
            data_type_extension=data_type_extension,
            multi_label=self.multi_label,
        )
        return dataset

    def compute_metrics(self, preds, labels, **kwargs):
        """
        Computes the evaluation metrics for the model predictions.

        Args:
            preds: Model predictions
            labels: Ground truth labels
            **kwargs: Additional metrics that should be used. Pass in the metrics as keyword arguments (name of metric: function to use). E.g. f1=sklearn.metrics.f1_score.
                        A metric function should take in two parameters. The first parameter will be the true labels, and the second parameter will be the predictions.

        Returns:
            result: Dictionary containing evaluation results. (Matthews correlation coefficient, tp, tn, fp, fn)
            wrong: List of InputExample objects corresponding to each incorrect prediction by the model
        """  # noqa: ignore flake8"
        assert len(preds) == len(labels)

        multi_label = self.multi_label
        extra_metrics = {}
        for metric, func in kwargs.items():
            extra_metrics[metric] = func(labels, preds)

        if self.args.regression:
            return {**extra_metrics}

        if multi_label:
            label_ranking_score = label_ranking_average_precision_score(labels, preds)
            return {**{"LRAP": label_ranking_score}, **extra_metrics}

        mcc = matthews_corrcoef(labels, preds)

        if self.model.num_labels == 2:
            tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()
            return {**{"mcc": mcc, "tp": tp, "tn": tn, "fp": fp, "fn": fn}, **extra_metrics}
        else:
            return {**{"mcc": mcc}, **extra_metrics}

    def predict(self, to_predict, image_path, image_type_extension=None):
        """
        Performs predictions on a list of text.

        Args:
            to_predict: A python dictionary to be sent to the model for prediction.
                The dictionary should be of the form {"text": [<list of sentences>], "images": [<list of images>]}.
            image_path: Path to the directory containing the image/images.
            image_type_extension (optional): If given, this will be added to the end of each value in "images".

        Returns:
            preds: A python list of the predictions (0 or 1) for each text.
            model_outputs: A python list of the raw model outputs for each text.
        """

        device = self.device
        model = self.model
        args = self.args
        multi_label = self.multi_label

        self._move_model_to_device()

        to_predict.update({"labels": ["0" for i in range(len(to_predict["text"]))]})
        to_predict = pd.DataFrame.from_dict(to_predict)

        eval_dataset = self.load_and_cache_examples(
            to_predict, image_path=image_path, evaluate=True, image_type_extension=image_type_extension, no_cache=True,
        )

        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(
            eval_dataset,
            sampler=eval_sampler,
            batch_size=args.eval_batch_size,
            collate_fn=collate_fn,
            num_workers=self.args.dataloader_num_workers,
        )

        eval_loss = 0.0
        nb_eval_steps = 0
        preds = None
        out_label_ids = None

        if args.n_gpu > 1:
            model = torch.nn.DataParallel(model)

        if args.fp16:
            from torch.cuda import amp

        for batch in tqdm(eval_dataloader, disable=args.silent, desc="Running Prediction"):
            batch = tuple(t.to(device) for t in batch)
            labels = batch[5]
            with torch.no_grad():
                inputs = self._get_inputs_dict(batch)
                if self.args.fp16:
                    with amp.autocast():
                        outputs = model(**inputs)
                        logits = outputs[0]  # Different from default behaviour
                else:
                    outputs = model(**inputs)
                    logits = outputs[0]  # Different from default behaviour
                tmp_eval_loss = self.criterion(logits, labels)

                if self.args.n_gpu > 1:
                    tmp_eval_loss = tmp_eval_loss.mean()
                eval_loss += tmp_eval_loss.item()

            nb_eval_steps += 1

            if preds is None:
                preds = torch.sigmoid(logits).detach().cpu().numpy()
                out_label_ids = labels.detach().cpu().numpy()
            else:
                preds = np.append(preds, torch.sigmoid(logits).detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(out_label_ids, labels.detach().cpu().numpy(), axis=0)

        eval_loss = eval_loss / nb_eval_steps
        model_outputs = preds

        if multi_label:
            preds = (preds > 0.5).astype(int)
        else:
            preds = np.argmax(preds, axis=1)

        return preds, model_outputs

    def calculate_weights(self, train_dataset):
        label_frequences = train_dataset.get_label_frequencies()
        label_frequences = [label_frequences[label] if label_frequences[label] > 0 else 1 for label in self.label_list]
        label_weights = (
            torch.tensor(label_frequences, device=self.device, dtype=torch.float) / len(train_dataset)
        ) ** -1

        return label_weights

    def _threshold(self, x, threshold):
        if x >= threshold:
            return 1
        return 0

    def _move_model_to_device(self):
        self.model.to(self.device)

    def _get_inputs_dict(self, batch):
        inputs = {
            "input_ids": batch[0],
            "input_modal": batch[2],
            "attention_mask": batch[1],
            "modal_start_tokens": batch[3],
            "modal_end_tokens": batch[4],
        }

        return inputs

    def _get_last_metrics(self, metric_values):
        return {metric: values[-1] for metric, values in metric_values.items()}

    def _create_training_progress_scores(self, multi_label, **kwargs):
        extra_metrics = {key: [] for key in kwargs}

        if multi_label:
            training_progress_scores = {
                "global_step": [],
                "LRAP": [],
                "train_loss": [],
                "eval_loss": [],
                **extra_metrics,
            }
        else:
            if self.model.num_labels == 2:
                training_progress_scores = {
                    "global_step": [],
                    "tp": [],
                    "tn": [],
                    "fp": [],
                    "fn": [],
                    "mcc": [],
                    "train_loss": [],
                    "eval_loss": [],
                    **extra_metrics,
                }
            elif self.model.num_labels == 1:
                training_progress_scores = {
                    "global_step": [],
                    "train_loss": [],
                    "eval_loss": [],
                    **extra_metrics,
                }
            else:
                training_progress_scores = {
                    "global_step": [],
                    "mcc": [],
                    "train_loss": [],
                    "eval_loss": [],
                    **extra_metrics,
                }

        return training_progress_scores

    def save_model(self, output_dir, model=None, results=None):
        os.makedirs(output_dir, exist_ok=True)

        if model and not self.args.no_save:
            # Take care of distributed/parallel training
            model_to_save = model.module if hasattr(model, "module") else model
            torch.save(model_to_save.state_dict(), os.path.join(output_dir, "pytorch_model.bin"))
            self.tokenizer.save_pretrained(output_dir)
            torch.save(self.args, os.path.join(output_dir, "training_args.bin"))
            self.transformer_config.architectures = [model_to_save.__class__.__name__]
            self.transformer_config.save_pretrained(output_dir)
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
        args = MultiModalClassificationArgs()
        args.load(input_dir)
        return args

    def get_named_parameters(self):
        return [n for n, p in self.model.named_parameters()]
