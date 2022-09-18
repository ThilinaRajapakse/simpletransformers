import json
import logging
import math
import os
import random
import warnings
import string
from dataclasses import asdict
from multiprocessing import Pool, cpu_count
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import transformers
from tensorboardX import SummaryWriter
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
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
from transformers.models.dpr import (
    DPRConfig,
    DPRContextEncoder,
    DPRQuestionEncoder,
    DPRContextEncoderTokenizerFast,
    DPRQuestionEncoderTokenizerFast,
)
from transformers.models.auto import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
)
import datasets
from datasets import load_from_disk

from simpletransformers.config.global_args import global_args
from simpletransformers.config.model_args import RetrievalArgs
from simpletransformers.config.utils import sweep_config_to_sweep_values
from simpletransformers.retrieval.retrieval_utils import (
    get_prediction_passage_dataset,
    load_hf_dataset,
    get_evaluation_passage_dataset,
    mean_reciprocal_rank_at_k,
)

try:
    import wandb

    wandb_available = True
except ImportError:
    wandb_available = False

logger = logging.getLogger(__name__)

MODEL_CLASSES = {
    "dpr": (
        DPRConfig,
        DPRContextEncoder,
        DPRQuestionEncoder,
        DPRContextEncoderTokenizerFast,
        DPRQuestionEncoderTokenizerFast,
    ),
    "custom": (
        AutoConfig,
        AutoModel,
        AutoModel,
        AutoTokenizer,
        AutoTokenizer,
    ),
}


class RetrievalModel:
    def __init__(
        self,
        model_type=None,
        model_name=None,
        context_encoder_name=None,
        query_encoder_name=None,
        context_encoder_tokenizer=None,
        query_encoder_tokenizer=None,
        prediction_passages=None,
        args=None,
        use_cuda=True,
        cuda_device=-1,
        **kwargs,
    ):
        """
        Initializes a RetrievalModel model.

        Args:
            model_type (str, optional): The type of model architecture. Defaults to None.
            model_name (str, optional): The exact architecture and trained weights to use for the full model. This may be a Hugging Face Transformers compatible pre-trained model, a community model, or the path to a directory containing model files. Defaults to None.
            context_encoder_name (str, optional): The exact architecture and trained weights to use for the context encoder model. This may be a Hugging Face Transformers compatible pre-trained model, a community model, or the path to a directory containing model files. Defaults to None.
            query_encoder_name (str, optional): The exact architecture and trained weights to use for the query encoder model. This may be a Hugging Face Transformers compatible pre-trained model, a community model, or the path to a directory containing model files. Defaults to None.
            context_encoder_tokenizer (str, optional): The tokenizer to use for the context encoder. This may be a Hugging Face Transformers compatible pre-trained tokenizer, a community tokenizer, or the path to a directory containing tokenizer files. Defaults to None.
            query_encoder_tokenizer (str, optional): The tokenizer to use for the query encoder. This may be a Hugging Face Transformers compatible pre-trained tokenizer, a community tokenizer, or the path to a directory containing tokenizer files. Defaults to None.
            prediction_passages (str, optional): The passages to be used as the corpus for retrieval when making predictions. Provide this only when using the model for predictions. Defaults to None.
            args (dict or RetrievalArgs, optional):  Default args will be used if this parameter is not provided. If provided, it should be a dict containing the args that should be changed in the default args or an instance of RetrievalArgs.
            use_cuda (bool, optional): Use GPU if available. Setting to False will force model to use CPU only.. Defaults to True.
            cuda_device (int, optional): Specific GPU that should be used. Will use the first available GPU by default. Defaults to -1.
            **kwargs (optional): For providing proxies, force_download, resume_download, cache_dir and other options specific to the 'from_pretrained' implementation where this will be supplied.

        Raises:
            ValueError: [description]
        """  # noqa: ignore flake8"

        self.args = self._load_model_args(model_name)

        if isinstance(args, dict):
            self.args.update_from_dict(args)
        elif isinstance(args, RetrievalArgs):
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

        if not use_cuda:
            self.args.fp16 = False

        try:
            (
                config_class,
                context_encoder,
                query_encoder,
                context_tokenizer,
                query_tokenizer,
            ) = MODEL_CLASSES[model_type]
        except KeyError:
            raise ValueError(
                "Model type {} not found. Available options are {}".format(
                    model_type, list(MODEL_CLASSES.keys())
                )
            )

        if context_encoder_name:
            self.context_config = config_class.from_pretrained(
                context_encoder_name, **self.args.context_config
            )
            if self.args.context_config.get("projection_dim") is not None:
                context_encoder._keys_to_ignore_on_load_missing.append("encode_proj")
            self.context_encoder = context_encoder.from_pretrained(
                context_encoder_name, config=self.context_config
            )
            self.context_tokenizer = context_tokenizer.from_pretrained(
                context_encoder_name
            )
        elif model_name:
            self.context_config = config_class.from_pretrained(
                os.path.join(model_name, "context_encoder"), **self.args.context_config
            )
            self.context_encoder = context_encoder.from_pretrained(
                os.path.join(model_name, "context_encoder"), config=self.context_config
            )
            self.context_tokenizer = context_tokenizer.from_pretrained(
                os.path.join(model_name, "context_encoder")
            )
        else:
            self.context_config = config_class(**self.args.context_config)
            self.context_encoder = context_encoder(config=self.context_config)
            self.context_tokenizer = context_tokenizer.from_pretrained(
                context_encoder_tokenizer
            )

        if query_encoder_name:
            self.query_config = config_class.from_pretrained(
                query_encoder_name, **self.args.query_config
            )
            if self.args.query_config.get("projection_dim") is not None:
                query_encoder._keys_to_ignore_on_load_missing.append("encode_proj")
            self.query_encoder = query_encoder.from_pretrained(
                query_encoder_name, config=self.query_config
            )
            self.query_tokenizer = query_tokenizer.from_pretrained(query_encoder_name)
        elif model_name:
            self.query_config = config_class.from_pretrained(
                os.path.join(model_name, "query_encoder"), **self.args.query_config
            )
            self.query_encoder = query_encoder.from_pretrained(
                os.path.join(model_name, "query_encoder"), config=self.query_config
            )
            self.query_tokenizer = query_tokenizer.from_pretrained(
                os.path.join(model_name, "query_encoder")
            )
        else:
            self.query_config = config_class(**self.args.query_config)
            self.query_encoder = query_encoder(config=self.query_config)
            self.query_tokenizer = query_tokenizer.from_pretrained(
                query_encoder_tokenizer
            )

        # TODO: Add support for adding special tokens to the tokenizers

        self.args.model_type = model_type
        self.args.model_name = model_name

        if prediction_passages is not None:
            self.prediction_passages = self.get_updated_prediction_passages(
                prediction_passages
            )
        else:
            self.prediction_passages = None

    def train_model(
        self,
        train_data,
        output_dir=None,
        show_running_loss=True,
        args=None,
        eval_data=None,
        additional_eval_passages=None,
        verbose=True,
        **kwargs,
    ):
        """
        Trains the model using 'train_data'

        Args:
            train_data: Pandas DataFrame containing the 3 columns - `query_text`, `gold_passage`, and `title`. (Title is optional)
                        - `query_text`: The Query text sequence
                        - `gold_passage`: The gold passage text sequence
                        - `title`: The title of the gold passage
                        If `use_hf_datasets` is True, then this may also be the path to a TSV file with the same columns.
            output_dir: The directory where model files will be saved. If not given, self.args.output_dir will be used.
            show_running_loss (optional): Set to False to prevent running loss from being printed to console. Defaults to True.
            args (optional): Optional changes to the args dict of the model. Any changes made will persist for the model.
            additional_eval_passages: Additional passages to be used during evaluation.
                        This may be a list of passages, a pandas DataFrame with the column `passages`, or a TSV file with the column `passages`.
            eval_data (optional): A DataFrame against which evaluation will be performed when evaluate_during_training is enabled. Is required if evaluate_during_training is enabled.
            **kwargs: Additional metrics that should be used. Pass in the metrics as keyword arguments (name of metric: function to use).
                        A metric function should take in two parameters. The first parameter will be the true labels, and the second parameter will be the predictions. Both inputs
                        will be lists of strings. Note that this will slow down training significantly as the predicted sequences need to be generated.

        Returns:
            global_step: Number of global steps trained
            training_details: Average training loss if evaluate_during_training is False or full training progress scores if evaluate_during_training is True
        """  # noqa: ignore flake8"

        if args:
            self.args.update_from_dict(args)

        # if self.args.silent:
        #     show_running_loss = False

        if self.args.evaluate_during_training and eval_data is None:
            raise ValueError(
                "evaluate_during_training is enabled but eval_data is not specified."
                " Pass eval_data to model.train_model() if using evaluate_during_training."
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

        if self.args.ddp_training:
            self.context_encoder = self.context_encoder.to(kwargs["rank"])
            self.query_encoder = self.query_encoder.to(kwargs["rank"])
            self.context_encoder = DDP(
                self.context_encoder, device_ids=[kwargs["rank"]]
            )
            self.query_encoder = DDP(self.query_encoder, device_ids=[kwargs["rank"]])
            self.device = kwargs["rank"]
        else:
            self._move_model_to_device()

        train_dataset = self.load_and_cache_examples(train_data, verbose=verbose)

        os.makedirs(output_dir, exist_ok=True)

        global_step, training_details = self.train(
            train_dataset,
            output_dir,
            show_running_loss=show_running_loss,
            eval_data=eval_data,
            additional_eval_passages=additional_eval_passages,
            verbose=verbose,
            **kwargs,
        )

        self.save_model(
            self.args.output_dir,
            context_model=self.context_encoder,
            query_model=self.query_encoder,
        )

        if verbose:
            logger.info(
                " Training of {} model complete. Saved to {}.".format(
                    self.args.model_name, output_dir
                )
            )

        return global_step, training_details

    def train(
        self,
        train_dataset,
        output_dir,
        show_running_loss=True,
        eval_data=None,
        additional_eval_passages=None,
        verbose=True,
        **kwargs,
    ):
        """
        Trains the model on train_dataset.

        Utility function to be used by the train_model() method. Not intended to be used directly.
        """

        context_model = self.context_encoder
        query_model = self.query_encoder
        args = self.args

        tb_writer = SummaryWriter(logdir=args.tensorboard_dir)
        train_sampler = RandomSampler(train_dataset)
        train_dataloader = DataLoader(
            train_dataset,
            sampler=train_sampler,
            batch_size=args.train_batch_size,
            num_workers=self.args.dataloader_num_workers,
        )

        if args.max_steps > 0:
            t_total = args.max_steps
            args.num_train_epochs = (
                args.max_steps
                // (len(train_dataloader) // args.gradient_accumulation_steps)
                + 1
            )
        else:
            t_total = (
                len(train_dataloader)
                // args.gradient_accumulation_steps
                * args.num_train_epochs
            )

        optimizer_grouped_parameters = self.get_optimizer_parameters(
            context_model, query_model, args
        )

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
        else:
            raise ValueError(
                "{} is not a valid optimizer class. Please use one of ('AdamW', 'Adafactor') instead.".format(
                    args.optimizer
                )
            )

        scheduler = self.get_scheduler(optimizer, args, t_total)

        criterion = torch.nn.NLLLoss(reduction="mean")

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
            context_model = torch.nn.DataParallel(context_model)
            query_model = torch.nn.DataParallel(query_model)

        logger.info(" Training started")

        global_step = 0
        training_progress_scores = None
        tr_loss, logging_loss = 0.0, 0.0
        context_model.zero_grad()
        query_model.zero_grad()
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
                project=args.wandb_project,
                config={**asdict(args)},
                **args.wandb_kwargs,
            )
            wandb.run._label(repo="simpletransformers")
            wandb.watch(context_model)
            wandb.watch(query_model)

        if args.fp16:
            from torch.cuda import amp

            scaler = amp.GradScaler()

        for current_epoch in train_iterator:
            if args.train_context_encoder:
                context_model.train()
            else:
                context_model.eval()
            if args.train_query_encoder:
                query_model.train()
            else:
                query_model.eval()
            if epochs_trained > 0:
                epochs_trained -= 1
                continue
            train_iterator.set_description(
                f"Epoch {epoch_number + 1} of {args.num_train_epochs}"
            )
            batch_iterator = tqdm(
                train_dataloader,
                desc=f"Running Epoch {epoch_number} of {args.num_train_epochs}",
                disable=args.silent,
                mininterval=0,
            )
            for step, batch in enumerate(batch_iterator):
                if steps_trained_in_current_epoch > 0:
                    steps_trained_in_current_epoch -= 1
                    continue
                # batch = tuple(t.to(device) for t in batch)

                context_inputs, query_inputs, labels = self._get_inputs_dict(batch)
                if args.fp16:
                    with amp.autocast():
                        loss, *_, correct_count = self._calculate_loss(
                            context_model,
                            query_model,
                            context_inputs,
                            query_inputs,
                            labels,
                            criterion,
                        )
                else:
                    loss, *_, correct_count = self._calculate_loss(
                        context_model,
                        query_model,
                        context_inputs,
                        query_inputs,
                        labels,
                        criterion,
                    )

                if args.n_gpu > 1:
                    loss = loss.mean()

                current_loss = loss.item()

                if show_running_loss:
                    batch_iterator.set_description(
                        f"Epochs {epoch_number}/{args.num_train_epochs}. Running Loss: {current_loss:9.4f} Correct count: {correct_count}"
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
                            context_model.parameters(), args.max_grad_norm
                        )
                        torch.nn.utils.clip_grad_norm_(
                            query_model.parameters(), args.max_grad_norm
                        )

                    if args.fp16:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    scheduler.step()  # Update learning rate schedule
                    context_model.zero_grad()
                    query_model.zero_grad()
                    global_step += 1

                    if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                        # Log metrics
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
                            output_dir_current,
                            optimizer,
                            scheduler,
                            context_model=context_model,
                            query_model=query_model,
                        )

                    if args.evaluate_during_training and (
                        args.evaluate_during_training_steps > 0
                        and global_step % args.evaluate_during_training_steps == 0
                    ):
                        # Only evaluate when single GPU otherwise metrics may not average well
                        results, *_ = self.eval_model(
                            eval_data,
                            additional_passages=additional_eval_passages,
                            verbose=verbose and args.evaluate_during_training_verbose,
                            silent=args.evaluate_during_training_silent,
                            **kwargs,
                        )
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
                                context_model=context_model,
                                query_model=query_model,
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
                            if args.save_best_model:
                                self.save_model(
                                    args.best_model_dir,
                                    optimizer,
                                    scheduler,
                                    context_model=context_model,
                                    query_model=query_model,
                                    results=results,
                                )
                        if best_eval_metric and args.early_stopping_metric_minimize:
                            if (
                                results[args.early_stopping_metric] - best_eval_metric
                                < args.early_stopping_delta
                            ):
                                best_eval_metric = results[args.early_stopping_metric]
                                if args.save_best_model:
                                    self.save_model(
                                        args.best_model_dir,
                                        optimizer,
                                        scheduler,
                                        context_model=context_model,
                                        query_model=query_model,
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
                                                f" Patience of {args.early_stopping_patience} steps reached"
                                            )
                                            logger.info(" Training terminated.")
                                            train_iterator.close()
                                        return (
                                            global_step,
                                            tr_loss / global_step
                                            if not self.args.evaluate_during_training
                                            else training_progress_scores,
                                        )
                        else:
                            if (
                                results[args.early_stopping_metric] - best_eval_metric
                                > args.early_stopping_delta
                            ):
                                best_eval_metric = results[args.early_stopping_metric]
                                if args.save_best_model:
                                    self.save_model(
                                        args.best_model_dir,
                                        optimizer,
                                        scheduler,
                                        context_model=context_model,
                                        query_model=query_model,
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
                                                f" Patience of {args.early_stopping_patience} steps reached"
                                            )
                                            logger.info(" Training terminated.")
                                            train_iterator.close()
                                        return (
                                            global_step,
                                            tr_loss / global_step
                                            if not self.args.evaluate_during_training
                                            else training_progress_scores,
                                        )
                        context_model.train()
                        query_model.train()

            epoch_number += 1
            output_dir_current = os.path.join(
                output_dir, "checkpoint-{}-epoch-{}".format(global_step, epoch_number)
            )

            if args.save_model_every_epoch or args.evaluate_during_training:
                os.makedirs(output_dir_current, exist_ok=True)

            if args.save_model_every_epoch:
                self.save_model(
                    output_dir_current,
                    optimizer,
                    scheduler,
                    context_model=context_model,
                    query_model=query_model,
                )

            if args.evaluate_during_training and args.evaluate_each_epoch:
                results, *_ = self.eval_model(
                    eval_data,
                    additional_passages=additional_eval_passages,
                    verbose=verbose and args.evaluate_during_training_verbose,
                    silent=args.evaluate_during_training_silent,
                    **kwargs,
                )

                if args.save_eval_checkpoints:
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
                    if args.save_best_model:
                        self.save_model(
                            args.best_model_dir,
                            optimizer,
                            scheduler,
                            context_model=context_model,
                            query_model=query_model,
                            results=results,
                        )
                if best_eval_metric and args.early_stopping_metric_minimize:
                    if (
                        results[args.early_stopping_metric] - best_eval_metric
                        < args.early_stopping_delta
                    ):
                        best_eval_metric = results[args.early_stopping_metric]
                        if args.save_best_model:
                            self.save_model(
                                args.best_model_dir,
                                optimizer,
                                scheduler,
                                context_model=context_model,
                                query_model=query_model,
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
                                    tr_loss / global_step
                                    if not self.args.evaluate_during_training
                                    else training_progress_scores,
                                )
                else:
                    if (
                        results[args.early_stopping_metric] - best_eval_metric
                        > args.early_stopping_delta
                    ):
                        best_eval_metric = results[args.early_stopping_metric]
                        if args.save_best_model:
                            self.save_model(
                                args.best_model_dir,
                                optimizer,
                                scheduler,
                                context_model=context_model,
                                query_model=query_model,
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
                                    tr_loss / global_step
                                    if not self.args.evaluate_during_training
                                    else training_progress_scores,
                                )

        return (
            global_step,
            tr_loss / global_step
            if not self.args.evaluate_during_training
            else training_progress_scores,
        )

    def eval_model(
        self,
        eval_data,
        evaluate_with_all_passages=True,
        additional_passages=None,
        top_k_values=None,
        retrieve_n_docs=None,
        return_doc_dicts=True,
        passage_dataset=None,
        qa_evaluation=False,
        output_dir=None,
        verbose=True,
        silent=False,
        **kwargs,
    ):
        """
        Evaluates the model on eval_data. Saves results to output_dir.

        Args:
            eval_data: Pandas DataFrame containing the 2 columns - `query_text`, 'gold_passage'.
                        - `query_text`: The Query text sequence
                        - `gold_passage`: The gold passage text sequence
                        If `use_hf_datasets` is True, then this may also be the path to a TSV file with the same columns.
            evaluate_with_all_passages: If True, evaluate with all passages. If False, evaluate only with in-batch negatives.
            additional_passages: Additional passages to be used during evaluation.
                        This may be a list of passages, a pandas DataFrame with the column "passages", or a TSV file with the column "passages".
            top_k_values: List of top-k values to be used for evaluation.
            retrieve_n_docs: Number of documents to retrieve for each query. Overrides `args.retrieve_n_docs` for this evaluation.
            return_doc_dicts: If True, return the doc dicts for the retrieved passages. Setting this to False can speed up evaluation.
            passage_dataset: Path to a saved Huggingface dataset (containing generated embeddings) for both the eval_data and additional passages
            qa_evaluation: If True, evaluation is done by checking if the retrieved passages contain the gold passage.
            output_dir: The directory where model files will be saved. If not given, self.args.output_dir will be used.
            verbose: If verbose, results will be printed to the console on completion of evaluation.
            silent: If silent, tqdm progress bars will be hidden.
            **kwargs: Additional metrics that should be used. Pass in the metrics as keyword arguments (name of metric: function to use).
                        A metric function should take in two parameters. The first parameter will be the true labels, and the second parameter will be the predictions. Both inputs
                        will be lists of strings. Note that this will slow down evaluation significantly as the predicted sequences need to be generated.
        Returns:
            results: Dictionary containing evaluation results.
        """  # noqa: ignore flake8"

        if not output_dir:
            output_dir = self.args.output_dir

        self._move_model_to_device()

        if self.prediction_passages is None:
            passage_dataset = get_evaluation_passage_dataset(
                eval_data,
                additional_passages,
                self.context_encoder,
                self.context_tokenizer,
                self.context_config,
                self.args,
                self.device,
                passage_dataset=passage_dataset,
            )
        else:
            passage_dataset = self.prediction_passages

        eval_dataset, gold_passages = load_hf_dataset(
            eval_data,
            self.context_tokenizer,
            self.query_tokenizer,
            self.args,
            evaluate=True,
        )

        result, doc_ids, doc_vectors, doc_dicts = self.evaluate(
            eval_dataset,
            gold_passages,
            evaluate_with_all_passages,
            passage_dataset,
            qa_evaluation,
            top_k_values,
            return_doc_dicts,
            output_dir,
            verbose=verbose,
            silent=silent,
            retrieve_n_docs=retrieve_n_docs,
            **kwargs,
        )

        if verbose:
            logger.info(result)

        return result, doc_ids, doc_vectors, doc_dicts

    def evaluate(
        self,
        eval_dataset,
        gold_passages,
        evaluate_with_all_passages=True,
        passage_dataset=None,
        qa_evaluation=False,
        top_k_values=None,
        return_doc_dicts=True,
        output_dir=None,
        verbose=True,
        silent=False,
        retrieve_n_docs=None,
        **kwargs,
    ):
        """
        Evaluates the model on eval_dataset.

        Utility function to be used by the eval_model() method. Not intended to be used directly.
        """

        context_model = self.context_encoder
        query_model = self.query_encoder
        args = self.args
        eval_output_dir = output_dir

        results = {}

        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(
            eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size
        )

        if args.n_gpu > 1:
            context_model = torch.nn.DataParallel(context_model)
            query_model = torch.nn.DataParallel(query_model)

        nb_eval_steps = 0
        eval_loss = 0
        context_model.eval()
        query_model.eval()

        criterion = torch.nn.NLLLoss(reduction="mean")

        if self.args.fp16:
            from torch.cuda import amp

        all_query_embeddings = np.zeros(
            (
                len(eval_dataset),
                self.query_config.hidden_size
                if "projection_dim" not in self.query_config.to_dict()
                or not self.query_config.projection_dim
                else self.query_config.projection_dim,
            )
        )
        for i, batch in enumerate(
            tqdm(
                eval_dataloader,
                disable=args.silent or silent,
                desc="Running Evaluation",
            )
        ):
            # batch = tuple(t.to(device) for t in batch)

            context_inputs, query_inputs, labels = self._get_inputs_dict(
                batch, evaluate=True
            )
            with torch.no_grad():
                if self.args.fp16:
                    with amp.autocast():
                        (
                            tmp_eval_loss,
                            _,
                            query_outputs,
                            correct_count,
                        ) = self._calculate_loss(
                            context_model,
                            query_model,
                            context_inputs,
                            query_inputs,
                            labels,
                            criterion,
                        )
                else:
                    (
                        tmp_eval_loss,
                        _,
                        query_outputs,
                        correct_count,
                    ) = self._calculate_loss(
                        context_model,
                        query_model,
                        context_inputs,
                        query_inputs,
                        labels,
                        criterion,
                    )
                if self.args.n_gpu > 1:
                    tmp_eval_loss = tmp_eval_loss.mean()

                eval_loss += tmp_eval_loss.item()
                all_query_embeddings[
                    i * args.eval_batch_size : (i + 1) * args.eval_batch_size
                ] = (query_outputs.cpu().detach().numpy())
            nb_eval_steps += 1

        eval_loss = eval_loss / nb_eval_steps

        results["eval_loss"] = eval_loss

        if evaluate_with_all_passages:
            doc_ids, doc_vectors, doc_dicts = self.retrieve_docs_from_query_embeddings(
                all_query_embeddings,
                passage_dataset,
                retrieve_n_docs,
                return_doc_dicts=True,
            )

            doc_texts = [doc_dict["passages"] for doc_dict in doc_dicts]

            scores = self.compute_metrics(
                gold_passages,
                doc_texts,
                self.args,
                qa_evaluation,
                top_k_values,
                **kwargs,
            )

            results.update(scores)

        output_eval_file = os.path.join(eval_output_dir, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            for key in sorted(results.keys()):
                writer.write("{} = {}\n".format(key, str(results[key])))

        if args.wandb_project:
            if not wandb.setup().settings.sweep_id:
                logger.info(" Initializing WandB run for evaluation.")
                wandb.init(
                    project=args.wandb_project,
                    config={**asdict(args)},
                    **args.wandb_kwargs,
                )
                wandb.run._label(repo="simpletransformers")
                self.wandb_run_id = wandb.run.id
            wandb.log(results)

        return results, doc_ids, doc_vectors, doc_dicts

    def predict(
        self,
        to_predict,
        prediction_passages=None,
        retrieve_n_docs=None,
        passages_only=False,
    ):
        """
        Retrieve the relevant documents from the prediction passages for a list of queries.

        Args:
            to_predict (list): A list of strings containing the queries to be predicted.
            prediction_passages (Union[str, DataFrame], optional): Path to a directory containing a passage dataset, a JSON/TSV file containing the passages, or a Pandas DataFrame. Defaults to None.
            retrieve_n_docs (int, optional): Number of docs to retrieve per query. Defaults to None.

        Raises:
            ValueError: [description]

        Returns:
            passages: List of lists containing the retrieved passages per query. (Shape: `(len(to_predict), retrieve_n_docs)`)
            doc_ids: List of lists containing the retrieved doc ids per query. (Shape: `(len(to_predict), retrieve_n_docs)`)
            doc_vectors: List of lists containing the retrieved doc vectors per query. (Shape: `(len(to_predict), retrieve_n_docs)`)
            doc_dicts: List of dicts containing the retrieved doc dicts per query.
        """  # noqa: ignore flake8"
        if self.prediction_passages is None:
            if prediction_passages is None:
                raise ValueError(
                    "prediction_passages cannot be None if the model does not contain a predicition passage index."
                )
            else:
                self.context_encoder.to(self.device)
                self.context_encoder.eval()
                self.prediction_passages = self.get_updated_prediction_passages(
                    prediction_passages
                )
                self.context_encoder.to(self.device)

        all_query_embeddings = np.zeros(
            (
                len(to_predict),
                self.query_config.hidden_size
                if "projection_dim" not in self.query_config.to_dict()
                or not self.query_config.projection_dim
                else self.query_config.projection_dim,
            )
        )

        query_model = self.query_encoder
        query_model.to(self.device)

        if self.args.n_gpu > 1:
            query_model = torch.nn.DataParallel(query_model)

        if self.args.fp16:
            from torch.cuda import amp

        query_model.eval()

        # Batching
        for i, batch in tqdm(
            enumerate(
                [
                    to_predict[i : i + self.args.eval_batch_size]
                    for i in range(0, len(to_predict), self.args.eval_batch_size)
                ]
            ),
            desc="Generating query embeddings",
            disable=self.args.silent,
        ):
            query_batch = self.query_tokenizer(
                batch,
                max_length=self.args.max_seq_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )

            query_inputs = {
                "input_ids": query_batch["input_ids"].to(self.device),
                "attention_mask": query_batch["attention_mask"].to(self.device),
            }

            with torch.no_grad():
                if self.args.fp16:
                    with amp.autocast():
                        query_outputs = query_model(**query_inputs).pooler_output
                else:
                    query_outputs = query_model(**query_inputs).pooler_output

            all_query_embeddings[
                i * self.args.eval_batch_size : (i + 1) * self.args.eval_batch_size
            ] = (query_outputs.cpu().detach().numpy())

        if not passages_only:
            doc_ids, doc_vectors, doc_dicts = self.retrieve_docs_from_query_embeddings(
                all_query_embeddings, self.prediction_passages, retrieve_n_docs
            )
            passages = [d["passages"] for d in doc_dicts]

            return passages, doc_ids, doc_vectors, doc_dicts
        else:
            passages = self.retrieve_docs_from_query_embeddings(
                all_query_embeddings,
                self.prediction_passages,
                retrieve_n_docs,
                passages_only=True,
            )
            return passages

    def compute_metrics(
        self,
        gold_passages,
        doc_texts,
        args,
        qa_evaluation=False,
        top_k_values=None,
        **kwargs,
    ):
        """
        Computes the metrics for the evaluation data.
        """
        if top_k_values is None:
            top_k_values = [1, 2, 3, 5, 10]

        top_k_values = [k for k in top_k_values if k <= args.retrieve_n_docs]

        relevance_list = np.zeros((len(gold_passages), args.retrieve_n_docs))
        for i, (docs, truth) in enumerate(zip(doc_texts, gold_passages)):
            for j, d in enumerate(docs):
                if qa_evaluation:
                    if truth.strip().lower().replace(" ", "").translate(
                        str.maketrans("", "", string.punctuation)
                    ) in d.strip().lower().replace(" ", "").translate(
                        str.maketrans("", "", string.punctuation)
                    ):
                        relevance_list[i, j] = 1
                        break
                else:
                    if d.strip().lower().translate(
                        str.maketrans("", "", string.punctuation)
                    ) == truth.strip().lower().translate(
                        str.maketrans("", "", string.punctuation)
                    ):
                        relevance_list[i, j] = 1
                        break

        mrr = {
            f"mrr@{k}": mean_reciprocal_rank_at_k(relevance_list, k)
            for k in top_k_values
        }

        top_k_accuracy = {
            f"top_{k}_accuracy": np.mean(np.sum(relevance_list[:, :k], axis=-1))
            for k in top_k_values
        }

        extra_metrics = {}
        for metric, func in kwargs.items():
            extra_metrics[metric] = func(gold_passages, doc_texts)

        results = {**mrr, **top_k_accuracy, **extra_metrics}

        return results

    def retrieve_docs_from_query_embeddings(
        self,
        query_embeddings,
        passage_dataset,
        retrieve_n_docs=None,
        return_doc_dicts=True,
        passages_only=False,
    ):
        """
        Retrieves documents from the index using the given query embeddings.
        """
        args = self.args
        if retrieve_n_docs is None:
            retrieve_n_docs = args.retrieve_n_docs

        query_embeddings_batched = [
            query_embeddings[i : i + args.retrieval_batch_size]
            for i in range(0, len(query_embeddings), args.retrieval_batch_size)
        ]

        if passages_only:
            passages = []
            for i, query_embeddings in enumerate(
                tqdm(
                    query_embeddings_batched,
                    desc="Retrieving docs",
                    disable=args.silent,
                )
            ):
                _, _, doc_dicts_batch = passage_dataset.get_top_docs(
                    query_embeddings.astype(np.float32), retrieve_n_docs
                )

                passages.extend([d["passages"] for d in doc_dicts_batch])

            return passages
        else:
            ids_batched = np.zeros((len(query_embeddings), retrieve_n_docs))
            vectors_batched = np.zeros(
                (
                    len(query_embeddings),
                    retrieve_n_docs,
                    self.context_config.hidden_size
                    if "projection_dim" not in self.context_config.to_dict()
                    or not self.context_config.projection_dim
                    else self.context_config.projection_dim,
                )
            )
            doc_dicts = []

            for i, query_embeddings in enumerate(
                tqdm(
                    query_embeddings_batched,
                    desc="Retrieving docs",
                    disable=args.silent,
                )
            ):
                ids, vectors, doc_dicts_batch = passage_dataset.get_top_docs(
                    query_embeddings.astype(np.float32), retrieve_n_docs
                )
                ids_batched[
                    i * args.retrieval_batch_size : (i * args.retrieval_batch_size)
                    + len(ids)
                ] = ids
                vectors_batched[
                    i * args.retrieval_batch_size : (i * args.retrieval_batch_size)
                    + len(ids)
                ] = vectors

                if return_doc_dicts:
                    doc_dicts.extend(doc_dicts_batch)

            if not return_doc_dicts:
                doc_dicts = None

            return ids_batched, vectors_batched, doc_dicts

    def build_hard_negatives(
        self,
        queries,
        passage_dataset=None,
        retrieve_n_docs=None,
        write_to_disk=True,
        hard_negatives_save_file_path=None,
    ):
        hard_negatives, *_ = self.predict(
            to_predict=queries,
            prediction_passages=passage_dataset,
            retrieve_n_docs=retrieve_n_docs,
        )

        if retrieve_n_docs is None:
            retrieve_n_docs = self.args.retrieve_n_docs

        column_names = [f"hard_negatives_{i}" for i in range(retrieve_n_docs)]

        # Build hard negative df from list of lists
        hard_negative_df = pd.DataFrame(hard_negatives, columns=column_names)

        if write_to_disk:
            if hard_negatives_save_file_path is None:
                os.makedirs(self.args.output_dir, exist_ok=True)
                hard_negatives_save_file_path = os.path.join(
                    self.args.output_dir, "hard_negatives.tsv"
                )
            hard_negative_df.to_csv(
                hard_negatives_save_file_path,
                index=False,
                sep="\t",
            )

        return hard_negative_df

    def load_and_cache_examples(
        self, data, evaluate=False, no_cache=False, verbose=True, silent=False
    ):
        """
        Creates a IRDataset from data
        """

        if not no_cache:
            no_cache = self.args.no_cache

        if not no_cache:
            os.makedirs(self.args.cache_dir, exist_ok=True)

        if self.args.use_hf_datasets:
            dataset = load_hf_dataset(
                data, self.context_tokenizer, self.query_tokenizer, self.args
            )

            return dataset
        else:
            # Retrieval models can only be used with hf datasets
            raise ValueError("Retrieval models can only be used with hf datasets.")

    def get_optimizer_parameters(self, context_model, query_model, args):
        no_decay = ["bias", "LayerNorm.weight"]

        optimizer_grouped_parameters = []
        custom_parameter_names = set()
        for group in self.args.custom_parameter_groups:
            params = group.pop("params")
            custom_parameter_names.update(params)
            param_group = {**group}
            param_group["params"] = [
                p for n, p in context_model.named_parameters() if n in params
            ]
            param_group["params"].extend(
                [p for n, p in query_model.named_parameters() if n in params]
            )
            optimizer_grouped_parameters.append(param_group)

        for group in self.args.custom_layer_parameters:
            layer_number = group.pop("layer")
            layer = f"layer.{layer_number}."
            group_d = {**group}
            group_nd = {**group}
            group_nd["weight_decay"] = 0.0
            params_d = []
            params_nd = []
            for n, p in context_model.named_parameters():
                if n not in custom_parameter_names and layer in n:
                    if any(nd in n for nd in no_decay):
                        params_nd.append(p)
                    else:
                        params_d.append(p)
                    custom_parameter_names.add(n)
            for n, p in query_model.named_parameters():
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
            if self.args.train_context_encoder:
                optimizer_grouped_parameters.extend(
                    [
                        {
                            "params": [
                                p
                                for n, p in context_model.named_parameters()
                                if n not in custom_parameter_names
                                and not any(nd in n for nd in no_decay)
                            ],
                            "weight_decay": args.weight_decay,
                        },
                        {
                            "params": [
                                p
                                for n, p in context_model.named_parameters()
                                if n not in custom_parameter_names
                                and any(nd in n for nd in no_decay)
                            ],
                            "weight_decay": 0.0,
                        },
                    ]
                )
            if self.args.train_query_encoder:
                optimizer_grouped_parameters.extend(
                    [
                        {
                            "params": [
                                p
                                for n, p in query_model.named_parameters()
                                if n not in custom_parameter_names
                                and not any(nd in n for nd in no_decay)
                            ],
                            "weight_decay": args.weight_decay,
                        },
                        {
                            "params": [
                                p
                                for n, p in query_model.named_parameters()
                                if n not in custom_parameter_names
                                and any(nd in n for nd in no_decay)
                            ],
                            "weight_decay": 0.0,
                        },
                    ]
                )

        return optimizer_grouped_parameters

    def get_scheduler(self, optimizer, args, t_total):
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

        return scheduler

    def get_updated_prediction_passages(self, prediction_passages):
        """
        Update the model passage dataset with a new passage dataset.
        This is typycally only useful for prediction.

        Args:
            prediction_passages (str): Path to new passage dataset.
        """
        prediction_passages = get_prediction_passage_dataset(
            prediction_passages,
            self.context_encoder,
            self.context_tokenizer,
            self.context_config,
            self.args,
            self.device,
        )

        return prediction_passages

    def _calculate_loss(
        self,
        context_model,
        query_model,
        context_inputs,
        query_inputs,
        labels,
        criterion,
    ):
        context_outputs = context_model(**context_inputs).pooler_output
        query_outputs = query_model(**query_inputs).pooler_output

        context_outputs = torch.nn.functional.dropout(context_outputs, p=0.1)
        query_outputs = torch.nn.functional.dropout(query_outputs, p=0.1)

        similarity_score = torch.matmul(query_outputs, context_outputs.t())
        softmax_score = torch.nn.functional.log_softmax(similarity_score, dim=-1)

        criterion = torch.nn.NLLLoss(reduction="mean")

        loss = criterion(softmax_score, labels)

        max_score, max_idxs = torch.max(softmax_score, 1)
        correct_predictions_count = (
            (max_idxs == torch.tensor(labels)).sum().cpu().detach().numpy().item()
        )

        return loss, context_outputs, query_outputs, correct_predictions_count

    def _get_inputs_dict(self, batch, evaluate=False):
        device = self.device

        labels = [i for i in range(len(batch["context_ids"]))]
        labels = torch.tensor(labels, dtype=torch.long)

        if not evaluate:
            # Training
            labels = labels.to(device)
            if self.args.hard_negatives:
                shuffled_indices = torch.randperm(len(labels))
                context_ids = torch.cat(
                    [
                        batch["context_ids"],
                        batch["hard_negative_ids"][shuffled_indices],
                    ],
                    dim=0,
                )
                context_masks = torch.cat(
                    [
                        batch["context_mask"],
                        batch["hard_negatives_mask"][shuffled_indices],
                    ],
                    dim=0,
                )
            else:
                context_ids = batch["context_ids"]
                context_masks = batch["context_mask"]
            context_input = {
                "input_ids": context_ids.to(device),
                "attention_mask": context_masks.to(device),
            }
            query_input = {
                "input_ids": batch["query_ids"].to(device),
                "attention_mask": batch["query_mask"].to(device),
            }
        else:
            # Evaluation
            shuffled_indices = torch.randperm(len(labels))

            labels = labels[shuffled_indices].to(device)

            if self.args.hard_negatives:
                context_ids = torch.cat(
                    [
                        batch["context_ids"][shuffled_indices],
                        batch["hard_negative_ids"],
                    ],
                    dim=0,
                )
                context_masks = torch.cat(
                    [
                        batch["context_mask"][shuffled_indices],
                        batch["hard_negatives_mask"],
                    ],
                    dim=0,
                )
            else:
                context_ids = batch["context_ids"][shuffled_indices]
                context_masks = batch["context_mask"][shuffled_indices]

            context_input = {
                "input_ids": context_ids.to(device),
                "attention_mask": context_masks.to(device),
            }
            query_input = {
                "input_ids": batch["query_ids"].to(device),
                "attention_mask": batch["query_mask"].to(device),
            }

        return context_input, query_input, labels

    def _create_training_progress_scores(self, **kwargs):
        # TODO: top_k_values should be part of the model. Probably.
        top_k_values = [1, 2, 3, 5, 10]
        extra_metrics = {key: [] for key in kwargs}
        training_progress_scores = {
            "global_step": [],
            "eval_loss": [],
            "train_loss": [],
            **extra_metrics,
        }
        training_progress_scores = {
            **training_progress_scores,
            **{f"mrr@{k}": [] for k in top_k_values},
        }
        training_progress_scores = {
            **training_progress_scores,
            **{f"top_{k}_accuracy": [] for k in top_k_values},
        }

        return training_progress_scores

    def _get_last_metrics(self, metric_values):
        return {metric: values[-1] for metric, values in metric_values.items()}

    def save_model(
        self,
        output_dir=None,
        optimizer=None,
        scheduler=None,
        context_model=None,
        query_model=None,
        results=None,
    ):
        if not output_dir:
            output_dir = self.args.output_dir

        if context_model and query_model and not self.args.no_save:
            os.makedirs(output_dir, exist_ok=True)

            logger.info(f"Saving model into {output_dir}")
            # Take care of distributed/parallel training
            context_model_to_save = (
                context_model.module
                if hasattr(context_model, "module")
                else context_model
            )
            query_model_to_save = (
                query_model.module if hasattr(query_model, "module") else query_model
            )
            self.save_model_args(output_dir)

            os.makedirs(os.path.join(output_dir, "context_encoder"), exist_ok=True)
            os.makedirs(os.path.join(output_dir, "query_encoder"), exist_ok=True)
            self.context_config.save_pretrained(
                os.path.join(output_dir, "context_encoder")
            )
            self.query_config.save_pretrained(os.path.join(output_dir, "query_encoder"))

            context_model_to_save.save_pretrained(
                os.path.join(output_dir, "context_encoder")
            )
            query_model_to_save.save_pretrained(
                os.path.join(output_dir, "query_encoder")
            )

            self.context_tokenizer.save_pretrained(
                os.path.join(output_dir, "context_encoder")
            )
            self.query_tokenizer.save_pretrained(
                os.path.join(output_dir, "query_encoder")
            )

            torch.save(self.args, os.path.join(output_dir, "training_args.bin"))
            if optimizer and scheduler and self.args.save_optimizer_and_scheduler:
                torch.save(
                    optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt")
                )
                torch.save(
                    scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt")
                )

        if results:
            os.makedirs(output_dir, exist_ok=True)
            output_eval_file = os.path.join(output_dir, "eval_results.txt")
            with open(output_eval_file, "w") as writer:
                for key in sorted(results.keys()):
                    writer.write("{} = {}\n".format(key, str(results[key])))

    def _move_model_to_device(self):
        self.context_encoder.to(self.device)
        self.query_encoder.to(self.device)

    def save_model_args(self, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        self.args.save(output_dir)

    def _load_model_args(self, input_dir):
        args = RetrievalArgs()
        args.load(input_dir)
        return args

    def get_named_parameters(self):
        return [n for n, p in self.context_encoder.named_parameters()] + [
            n for n, p in self.query_encoder.named_parameters()
        ]
