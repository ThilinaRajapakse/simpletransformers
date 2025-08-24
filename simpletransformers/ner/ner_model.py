from __future__ import absolute_import, division, print_function
import collections
import logging
import math
import os
import random
import tempfile
import warnings
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from seqeval.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from simpletransformers.config.model_args import NERArgs
from simpletransformers.config.utils import sweep_config_to_sweep_values
from simpletransformers.losses.loss_utils import init_loss
from simpletransformers.ner.ner_utils import (
    InputExample,
    LazyNERDataset,
    convert_examples_to_features,
    get_examples_from_df,
    load_hf_dataset,
    read_examples_from_file,
    flatten_results,
)

from transformers import DummyObject, requires_backends


class NystromformerTokenizer(metaclass=DummyObject):
    _backends = ["sentencepiece"]

    def __init__(self, *args, **kwargs):
        requires_backends(self, ["sentencepiece"])


from torch.utils.tensorboard import SummaryWriter
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from tqdm.auto import tqdm, trange
from transformers import (
    AlbertConfig,
    AlbertForTokenClassification,
    AlbertTokenizer,
    AutoConfig,
    AutoModelForTokenClassification,
    AutoTokenizer,
    BertConfig,
    BertForTokenClassification,
    BertTokenizer,
    BertweetTokenizer,
    BigBirdConfig,
    BigBirdForTokenClassification,
    BigBirdTokenizer,
    CamembertConfig,
    CamembertForTokenClassification,
    CamembertTokenizer,
    DebertaConfig,
    DebertaForTokenClassification,
    DebertaTokenizer,
    DebertaV2Config,
    DebertaV2ForTokenClassification,
    DebertaV2Tokenizer,
    DistilBertConfig,
    DistilBertForTokenClassification,
    DistilBertTokenizer,
    ElectraConfig,
    ElectraForTokenClassification,
    ElectraTokenizer,
    HerbertTokenizerFast,
    LayoutLMConfig,
    LayoutLMForTokenClassification,
    LayoutLMTokenizer,
    LayoutLMv2Config,
    LayoutLMv2ForTokenClassification,
    LayoutLMv2Tokenizer,
    LongformerConfig,
    LongformerForTokenClassification,
    LongformerTokenizer,
    LukeConfig,
    LukeTokenizer,
    MLukeTokenizer,
    LukeForTokenClassification,
    MPNetConfig,
    MPNetForTokenClassification,
    MPNetTokenizer,
    MobileBertConfig,
    MobileBertForTokenClassification,
    MobileBertTokenizer,
    NystromformerConfig,
    NystromformerForTokenClassification,
    RemBertConfig,
    RemBertForTokenClassification,
    RemBertTokenizer,
    RemBertTokenizerFast,
    RobertaConfig,
    RobertaForTokenClassification,
    RobertaTokenizerFast,
    SqueezeBertConfig,
    SqueezeBertForTokenClassification,
    SqueezeBertTokenizer,
    XLMConfig,
    XLMForTokenClassification,
    XLMTokenizer,
    XLMRobertaConfig,
    XLMRobertaForTokenClassification,
    XLMRobertaTokenizer,
    XLNetConfig,
    XLNetForTokenClassification,
    XLNetTokenizerFast,
)
from transformers.convert_graph_to_onnx import convert, quantize
from torch.optim import AdamW
from transformers.optimization import Adafactor
from transformers.optimization import (
    get_constant_schedule,
    get_constant_schedule_with_warmup,
    get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    get_cosine_with_hard_restarts_schedule_with_warmup,
    get_polynomial_decay_schedule_with_warmup,
)

try:
    import wandb

    wandb_available = True
except ImportError:
    wandb_available = False

logger = logging.getLogger(__name__)

MODELS_WITHOUT_CLASS_WEIGHTS_SUPPORT = ["squeezebert", "deberta", "mpnet"]

MODELS_WITH_EXTRA_SEP_TOKEN = [
    "roberta",
    "camembert",
    "xlmroberta",
    "longformer",
    "mpnet",
]


class NERModel:
    def __init__(
        self,
        model_type,
        model_name,
        labels=None,
        weight=None,
        args=None,
        use_cuda=True,
        cuda_device=-1,
        onnx_execution_provider=None,
        **kwargs,
    ):
        """
        Initializes a NERModel

        Args:
            model_type: The type of model (bert, roberta)
            model_name: Default Transformer model name or path to a directory containing Transformer model file (pytorch_model.bin).
            labels (optional): A list of all Named Entity labels.  If not given, ["O", "B-MISC", "I-MISC",  "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"] will be used.
            weight (optional): A `torch.Tensor`, `numpy.ndarray` or list.  The weight to be applied to each class when computing the loss of the model.
            args (optional): Default args will be used if this parameter is not provided. If provided, it should be a dict containing the args that should be changed in the default args.
            use_cuda (optional): Use GPU if available. Setting to False will force model to use CPU only.
            cuda_device (optional): Specific GPU that should be used. Will use the first available GPU by default.
            onnx_execution_provider (optional): The execution provider to use for ONNX export.
            **kwargs (optional): For providing proxies, force_download, resume_download, cache_dir and other options specific to the 'from_pretrained' implementation where this will be supplied.
        """  # noqa: ignore flake8"

        MODEL_CLASSES = {
            "albert": (AlbertConfig, AlbertForTokenClassification, AlbertTokenizer),
            "auto": (AutoConfig, AutoModelForTokenClassification, AutoTokenizer),
            "bert": (BertConfig, BertForTokenClassification, BertTokenizer),
            "bertweet": (
                RobertaConfig,
                RobertaForTokenClassification,
                BertweetTokenizer,
            ),
            "bigbird": (BigBirdConfig, BigBirdForTokenClassification, BigBirdTokenizer),
            "camembert": (
                CamembertConfig,
                CamembertForTokenClassification,
                CamembertTokenizer,
            ),
            "deberta": (DebertaConfig, DebertaForTokenClassification, DebertaTokenizer),
            "deberta-v2": (
                DebertaV2Config,
                DebertaV2ForTokenClassification,
                DebertaV2Tokenizer,
            ),
            "distilbert": (
                DistilBertConfig,
                DistilBertForTokenClassification,
                DistilBertTokenizer,
            ),
            "electra": (ElectraConfig, ElectraForTokenClassification, ElectraTokenizer),
            "herbert": (BertConfig, BertForTokenClassification, HerbertTokenizerFast),
            "layoutlm": (
                LayoutLMConfig,
                LayoutLMForTokenClassification,
                LayoutLMTokenizer,
            ),
            "layoutlmv2": (
                LayoutLMv2Config,
                LayoutLMv2ForTokenClassification,
                LayoutLMv2Tokenizer,
            ),
            "longformer": (
                LongformerConfig,
                LongformerForTokenClassification,
                LongformerTokenizer,
            ),
            "luke": (
                LukeConfig,
                LukeForTokenClassification,
                LukeTokenizer,
            ),
            "mluke": (
                LukeConfig,
                LukeForTokenClassification,
                MLukeTokenizer,
            ),
            "mobilebert": (
                MobileBertConfig,
                MobileBertForTokenClassification,
                MobileBertTokenizer,
            ),
            "mpnet": (MPNetConfig, MPNetForTokenClassification, MPNetTokenizer),
            "nystromformer": (
                NystromformerConfig,
                NystromformerForTokenClassification,
                BigBirdTokenizer,
            ),
            "rembert": (
                RemBertConfig,
                RemBertForTokenClassification,
                RemBertTokenizerFast,
            ),
            "roberta": (
                RobertaConfig,
                RobertaForTokenClassification,
                RobertaTokenizerFast,
            ),
            "squeezebert": (
                SqueezeBertConfig,
                SqueezeBertForTokenClassification,
                SqueezeBertTokenizer,
            ),
            "xlm": (XLMConfig, XLMForTokenClassification, XLMTokenizer),
            "xlmroberta": (
                XLMRobertaConfig,
                XLMRobertaForTokenClassification,
                XLMRobertaTokenizer,
            ),
            "xlnet": (XLNetConfig, XLNetForTokenClassification, XLNetTokenizerFast),
        }

        self.args = self._load_model_args(model_name)

        if isinstance(args, dict):
            self.args.update_from_dict(args)
        elif isinstance(args, NERArgs):
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

        if not use_cuda:
            self.args.fp16 = False

        if labels and self.args.labels_list:
            assert labels == self.args.labels_list
            self.args.labels_list = labels
        elif labels:
            self.args.labels_list = labels
        elif self.args.labels_list:
            pass
        else:
            self.args.labels_list = [
                "O",
                "B-MISC",
                "I-MISC",
                "B-PER",
                "I-PER",
                "B-ORG",
                "I-ORG",
                "B-LOC",
                "I-LOC",
            ]
        self.num_labels = len(self.args.labels_list)
        self.id2label = {i: label for i, label in enumerate(self.args.labels_list)}
        self.label2id = {label: i for i, label in enumerate(self.args.labels_list)}

        config_class, model_class, tokenizer_class = MODEL_CLASSES[model_type]
        if self.num_labels:
            self.config = config_class.from_pretrained(
                model_name, num_labels=self.num_labels, **self.args.config
            )
            self.num_labels = self.num_labels
        else:
            self.config = config_class.from_pretrained(model_name, **self.args.config)
            self.num_labels = self.config.num_labels

        self.config.id2label = self.id2label
        self.config.label2id = self.label2id

        if model_type in MODELS_WITHOUT_CLASS_WEIGHTS_SUPPORT and weight is not None:
            raise ValueError(
                "{} does not currently support class weights".format(model_type)
            )
        else:
            self.weight = weight

        if use_cuda:
            if torch.cuda.is_available():
                if cuda_device == -1:
                    self.device = torch.device("cuda")
                else:
                    self.device = torch.device(f"cuda:{cuda_device}")
            else:
                raise ValueError(
                    "'use_cuda' set to True when cuda is unavailable."
                    "Make sure CUDA is available or set use_cuda=False."
                )
        else:
            self.device = "cpu"

        self.loss_fct = init_loss(
            weight=self.weight, device=self.device, args=self.args
        )

        if self.args.onnx:
            from onnxruntime import InferenceSession, SessionOptions

            if not onnx_execution_provider:
                onnx_execution_provider = (
                    ["CUDAExecutionProvider"] if use_cuda else ["CPUExecutionProvider"]
                )

            options = SessionOptions()

            if self.args.dynamic_quantize:
                model_path = quantize(Path(os.path.join(model_name, "onnx_model.onnx")))
                self.model = InferenceSession(
                    model_path.as_posix(), options, providers=onnx_execution_provider
                )
            else:
                model_path = os.path.join(model_name, "onnx_model.onnx")
                self.model = InferenceSession(
                    model_path, options, providers=onnx_execution_provider
                )
        else:
            if not self.args.quantized_model:
                self.model = model_class.from_pretrained(
                    model_name, config=self.config, **kwargs
                )
            else:
                quantized_weights = torch.load(
                    os.path.join(model_name, "pytorch_model.bin")
                )
                self.model = model_class.from_pretrained(
                    None, config=self.config, state_dict=quantized_weights
                )

            if self.args.dynamic_quantize:
                self.model = torch.quantization.quantize_dynamic(
                    self.model, {torch.nn.Linear}, dtype=torch.qint8
                )
            if self.args.quantized_model:
                self.model.load_state_dict(quantized_weights)
            if self.args.dynamic_quantize:
                self.args.quantized_model = True

        self.results = {}

        if self.args.fp16:
            try:
                from torch.cuda import amp
            except AttributeError:
                raise AttributeError(
                    "fp16 requires Pytorch >= 1.6. Please update Pytorch or turn off fp16."
                )

        if model_name in [
            "vinai/bertweet-base",
            "vinai/bertweet-covid19-base-cased",
            "vinai/bertweet-covid19-base-uncased",
        ]:
            self.tokenizer = tokenizer_class.from_pretrained(
                model_name,
                do_lower_case=self.args.do_lower_case,
                normalization=True,
                **kwargs,
            )
        else:
            self.tokenizer = tokenizer_class.from_pretrained(
                model_name, do_lower_case=self.args.do_lower_case, **kwargs
            )

        if self.args.special_tokens_list:
            self.tokenizer.add_tokens(
                self.args.special_tokens_list, special_tokens=True
            )
            self.model.resize_token_embeddings(len(self.tokenizer))

        self.args.model_name = model_name
        self.args.model_type = model_type

        self.pad_token_label_id = CrossEntropyLoss().ignore_index

        if model_type == "camembert":
            warnings.warn(
                "use_multiprocessing automatically disabled as CamemBERT"
                " fails when using multiprocessing for feature conversion."
            )
            self.args.use_multiprocessing = False

        if self.args.wandb_project and not wandb_available:
            warnings.warn(
                "wandb_project specified but wandb is not available. Wandb disabled."
            )
            self.args.wandb_project = None

    def train_model(
        self,
        train_data,
        output_dir=None,
        show_running_loss=True,
        args=None,
        eval_data=None,
        verbose=True,
        **kwargs,
    ):
        """
        Trains the model using 'train_data'

        Args:
            train_data: train_data should be the path to a .txt file containing the training data OR a pandas DataFrame with 3 columns.
                        If a text file is given the data should be in the CoNLL format. i.e. One word per line, with sentences seperated by an empty line.
                        The first word of the line should be a word, and the last should be a Name Entity Tag.
                        If a DataFrame is given, each sentence should be split into words, with each word assigned a tag, and with all words from the same sentence given the same sentence_id.
            eval_data: Evaluation data (same format as train_data) against which evaluation will be performed when evaluate_during_training is enabled. Is required if evaluate_during_training is enabled.
            output_dir: The directory where model files will be saved. If not given, self.args.output_dir will be used.
            show_running_loss (optional): Set to False to prevent running loss from being printed to console. Defaults to True.
            args (optional): Optional changes to the args dict of the model. Any changes made will persist for the model.
            **kwargs: Additional metrics that should be used. Pass in the metrics as keyword arguments (name of metric: function to use). E.g. f1=sklearn.metrics.f1_score.
                        A metric function should take in two parameters. The first parameter will be the true labels, and the second parameter will be the predictions.

        Returns:
            global_step: Number of global steps trained
            training_details: Average training loss if evaluate_during_training is False or full training progress scores if evaluate_during_training is True
        """  # noqa: ignore flake8"

        if args:
            self.args.update_from_dict(args)

        if self.args.silent:
            show_running_loss = False

        if self.args.evaluate_during_training and eval_data is None:
            if "eval_df" in kwargs:
                warnings.warn(
                    "The eval_df parameter has been renamed to eval_data."
                    " Using eval_df will raise an error in a future version."
                )
                eval_data = kwargs.pop("eval_df")
            else:
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
                " Use --overwrite_output_dir to overcome.".format(output_dir)
            )

        self._move_model_to_device()

        train_dataset = self.load_and_cache_examples(train_data)

        os.makedirs(output_dir, exist_ok=True)

        global_step, training_details = self.train(
            train_dataset,
            output_dir,
            show_running_loss=show_running_loss,
            eval_data=eval_data,
            **kwargs,
        )

        self.save_model(model=self.model)

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
        eval_data=None,
        test_data=None,
        verbose=True,
        **kwargs,
    ):
        """
        Trains the model on train_dataset.

        Utility function to be used by the train_model() method. Not intended to be used directly.
        """

        model = self.model
        args = self.args

        tb_writer = SummaryWriter(log_dir=args.tensorboard_dir)
        train_sampler = RandomSampler(train_dataset)
        train_dataloader = DataLoader(
            train_dataset,
            sampler=train_sampler,
            batch_size=args.train_batch_size,
            num_workers=self.args.dataloader_num_workers,
        )

        t_total = (
            len(train_dataloader)
            // args.gradient_accumulation_steps
            * args.num_train_epochs
        )

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

        if args.n_gpu > 1:
            model = torch.nn.DataParallel(model)

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
                project=args.wandb_project,
                config={**asdict(args)},
                **args.wandb_kwargs,
            )
            wandb.run._label(repo="simpletransformers")
            wandb.watch(self.model)
            self.wandb_run_id = wandb.run.id

        if self.args.fp16:
            from torch.cuda import amp

            scaler = amp.GradScaler()

        for _ in train_iterator:
            model.train()
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

                inputs = self._get_inputs_dict(batch)

                if self.args.fp16:
                    with amp.autocast():
                        loss, *_ = self._calculate_loss(
                            model,
                            inputs,
                            loss_fct=self.loss_fct,
                            num_labels=self.num_labels,
                            args=self.args,
                        )
                else:
                    loss, *_ = self._calculate_loss(
                        model,
                        inputs,
                        loss_fct=self.loss_fct,
                        num_labels=self.num_labels,
                        args=self.args,
                    )

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

                if self.args.fp16:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

                tr_loss += loss.item()
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    if self.args.fp16:
                        scaler.unscale_(optimizer)
                    if args.optimizer == "AdamW":
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(), args.max_grad_norm
                        )

                    if self.args.fp16:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    scheduler.step()  # Update learning rate schedule
                    model.zero_grad()
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
                            output_dir_current, optimizer, scheduler, model=model
                        )

                    if args.evaluate_during_training and (
                        args.evaluate_during_training_steps > 0
                        and global_step % args.evaluate_during_training_steps == 0
                    ):
                        output_dir_current = os.path.join(
                            output_dir, "checkpoint-{}".format(global_step)
                        )

                        os.makedirs(output_dir_current, exist_ok=True)

                        # Only evaluate when single GPU otherwise metrics may not average well
                        results, _, _ = self.eval_model(
                            eval_data,
                            verbose=verbose and args.evaluate_during_training_verbose,
                            wandb_log=False,
                            output_dir=output_dir_current,
                            **kwargs,
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

                        if test_data is not None:
                            test_results, _, _ = self.eval_model(
                                test_data,
                                verbose=verbose
                                and args.evaluate_during_training_verbose,
                                silent=args.evaluate_during_training_silent,
                                wandb_log=False,
                                **kwargs,
                            )
                            for key in test_results:
                                training_progress_scores["test_" + key].append(
                                    test_results[key]
                                )

                        report = pd.DataFrame(training_progress_scores)
                        report.to_csv(
                            os.path.join(
                                args.output_dir, "training_progress_scores.csv"
                            ),
                            index=False,
                        )

                        if args.wandb_project or self.is_sweeping:
                            wandb.log(self._get_last_metrics(training_progress_scores))

                        for key, value in flatten_results(
                            self._get_last_metrics(training_progress_scores)
                        ).items():
                            try:
                                tb_writer.add_scalar(key, value, global_step)
                            except (NotImplementedError, AssertionError):
                                if verbose:
                                    logger.warning(
                                        f"can't log value of type: {type(value)} to tensorboar"
                                    )
                        tb_writer.flush()

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
                                            (
                                                tr_loss / global_step
                                                if not self.args.evaluate_during_training
                                                else training_progress_scores
                                            ),
                                        )
                        model.train()

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
                results, _, _ = self.eval_model(
                    eval_data,
                    verbose=verbose and args.evaluate_during_training_verbose,
                    wandb_log=False,
                    **kwargs,
                )

                self.save_model(
                    output_dir_current, optimizer, scheduler, results=results
                )

                training_progress_scores["global_step"].append(global_step)
                training_progress_scores["train_loss"].append(current_loss)
                for key in results:
                    training_progress_scores[key].append(results[key])

                if test_data is not None:
                    test_results, _, _ = self.eval_model(
                        test_data,
                        verbose=verbose and args.evaluate_during_training_verbose,
                        silent=args.evaluate_during_training_silent,
                        wandb_log=False,
                        **kwargs,
                    )
                    for key in test_results:
                        training_progress_scores["test_" + key].append(
                            test_results[key]
                        )

                report = pd.DataFrame(training_progress_scores)
                report.to_csv(
                    os.path.join(args.output_dir, "training_progress_scores.csv"),
                    index=False,
                )

                if args.wandb_project or self.is_sweeping:
                    wandb.log(self._get_last_metrics(training_progress_scores))

                for key, value in flatten_results(
                    self._get_last_metrics(training_progress_scores)
                ).items():
                    try:
                        tb_writer.add_scalar(key, value, global_step)
                    except (NotImplementedError, AssertionError):
                        if verbose:
                            logger.warning(
                                f"can't log value of type: {type(value)} to tensorboar"
                            )
                tb_writer.flush()

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
        eval_data,
        output_dir=None,
        verbose=True,
        silent=False,
        wandb_log=True,
        **kwargs,
    ):
        """
        Evaluates the model on eval_data. Saves results to output_dir.

        Args:
            eval_data: eval_data should be the path to a .txt file containing the evaluation data or a pandas DataFrame.
                        If a text file is used the data should be in the CoNLL format. I.e. One word per line, with sentences seperated by an empty line.
                        The first word of the line should be a word, and the last should be a Name Entity Tag.
                        If a DataFrame is given, each sentence should be split into words, with each word assigned a tag, and with all words from the same sentence given the same sentence_id.

            output_dir: The directory where model files will be saved. If not given, self.args.output_dir will be used.
            verbose: If verbose, results will be printed to the console on completion of evaluation.
            silent: If silent, tqdm progress bars will be hidden.
            wandb_log: If True, evaluation results will be logged to wandb.
            **kwargs: Additional metrics that should be used. Pass in the metrics as keyword arguments (name of metric: function to use). E.g. f1=sklearn.metrics.f1_score.
                        A metric function should take in two parameters. The first parameter will be the true labels, and the second parameter will be the predictions.

        Returns:
            result: Dictionary containing evaluation results. (eval_loss, precision, recall, f1_score)
            model_outputs: List of raw model outputs
            preds_list: List of predicted tags
        """  # noqa: ignore flake8"
        if not output_dir:
            output_dir = self.args.output_dir

        self._move_model_to_device()

        eval_dataset = self.load_and_cache_examples(eval_data, evaluate=True)

        result, model_outputs, preds_list = self.evaluate(
            eval_dataset,
            output_dir,
            verbose=verbose,
            silent=silent,
            wandb_log=wandb_log,
            **kwargs,
        )
        self.results.update(result)

        if verbose:
            logger.info(self.results)

        return result, model_outputs, preds_list

    def evaluate(
        self,
        eval_dataset,
        output_dir,
        verbose=True,
        silent=False,
        wandb_log=True,
        **kwargs,
    ):
        """
        Evaluates the model on eval_dataset.

        Utility function to be used by the eval_model() method. Not intended to be used directly.
        """

        model = self.model
        args = self.args
        pad_token_label_id = self.pad_token_label_id
        eval_output_dir = output_dir

        results = {}

        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(
            eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size
        )

        eval_loss = 0.0
        nb_eval_steps = 0
        preds = None
        out_label_ids = None
        model.eval()

        if args.n_gpu > 1:
            model = torch.nn.DataParallel(model)

        if self.args.fp16:
            from torch.cuda import amp

        for batch in tqdm(
            eval_dataloader, disable=args.silent or silent, desc="Running Evaluation"
        ):
            with torch.no_grad():
                inputs = self._get_inputs_dict(batch)

                if self.args.fp16:
                    with amp.autocast():
                        outputs = self._calculate_loss(
                            model,
                            inputs,
                            loss_fct=self.loss_fct,
                            num_labels=self.num_labels,
                            args=self.args,
                        )
                        tmp_eval_loss, logits = outputs[:2]
                else:
                    outputs = self._calculate_loss(
                        model,
                        inputs,
                        loss_fct=self.loss_fct,
                        num_labels=self.num_labels,
                        args=self.args,
                    )
                    tmp_eval_loss, logits = outputs[:2]

                if self.args.n_gpu > 1:
                    tmp_eval_loss = tmp_eval_loss.mean()
                eval_loss += tmp_eval_loss.item()

            nb_eval_steps += 1

            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = inputs["labels"].detach().cpu().numpy()
                out_input_ids = inputs["input_ids"].detach().cpu().numpy()
                out_attention_mask = inputs["attention_mask"].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(
                    out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0
                )
                out_input_ids = np.append(
                    out_input_ids, inputs["input_ids"].detach().cpu().numpy(), axis=0
                )
                out_attention_mask = np.append(
                    out_attention_mask,
                    inputs["attention_mask"].detach().cpu().numpy(),
                    axis=0,
                )

        eval_loss = eval_loss / nb_eval_steps
        token_logits = preds
        preds = np.argmax(preds, axis=2)

        label_map = {i: label for i, label in enumerate(self.args.labels_list)}

        out_label_list = [[] for _ in range(out_label_ids.shape[0])]
        preds_list = [[] for _ in range(out_label_ids.shape[0])]

        for i in range(out_label_ids.shape[0]):
            for j in range(out_label_ids.shape[1]):
                if out_label_ids[i, j] != pad_token_label_id:
                    out_label_list[i].append(label_map[out_label_ids[i][j]])
                    preds_list[i].append(label_map[preds[i][j]])

        word_tokens = []
        for i in range(len(preds_list)):
            w_log = self._convert_tokens_to_word_logits(
                out_input_ids[i],
                out_label_ids[i],
                out_attention_mask[i],
                token_logits[i],
            )
            word_tokens.append(w_log)

        model_outputs = [
            [word_tokens[i][j] for j in range(len(preds_list[i]))]
            for i in range(len(preds_list))
        ]

        extra_metrics = {}
        for metric, func in kwargs.items():
            if metric.startswith("prob_"):
                extra_metrics[metric] = func(out_label_list, model_outputs)
            else:
                extra_metrics[metric] = func(out_label_list, preds_list)

        result = {
            "eval_loss": eval_loss,
            "precision": precision_score(out_label_list, preds_list),
            "recall": recall_score(out_label_list, preds_list),
            "f1_score": f1_score(out_label_list, preds_list),
            **extra_metrics,
        }

        results.update(result)

        os.makedirs(eval_output_dir, exist_ok=True)
        output_eval_file = os.path.join(eval_output_dir, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            if args.classification_report:
                cls_report = classification_report(out_label_list, preds_list, digits=4)
                writer.write("{}\n".format(cls_report))
            for key in sorted(result.keys()):
                writer.write("{} = {}\n".format(key, str(result[key])))

        if self.args.wandb_project and wandb_log:
            wandb.init(
                project=args.wandb_project,
                config={**asdict(args)},
                **args.wandb_kwargs,
            )
            wandb.run._label(repo="simpletransformers")

            labels_list = sorted(self.args.labels_list)

            truth = [tag for out in out_label_list for tag in out]
            preds = [tag for pred_out in preds_list for tag in pred_out]
            outputs = [
                np.mean(logits, axis=0) for output in model_outputs for logits in output
            ]

            # ROC
            wandb.log({"roc": wandb.plot.roc_curve(truth, outputs, labels_list)})

            # Precision Recall
            wandb.log({"pr": wandb.plot.pr_curve(truth, outputs, labels_list)})

            # Confusion Matrix
            wandb.sklearn.plot_confusion_matrix(
                truth,
                preds,
                labels=labels_list,
            )

        return results, model_outputs, preds_list

    def predict(self, to_predict, split_on_space=True):
        """
        Performs predictions on a list of text.

        Args:
            to_predict: A python list of text (str) to be sent to the model for prediction.
            split_on_space: If True, each sequence will be split by spaces for assigning labels.
                            If False, to_predict must be a a list of lists, with the inner list being a
                            list of strings consisting of the split sequences. The outer list is the list of sequences to
                            predict on.

        Returns:
            preds: A Python list of lists with dicts containing each word mapped to its NER tag.
            model_outputs: A Python list of lists with dicts containing each word mapped to its list with raw model output.
        """  # noqa: ignore flake8"

        device = self.device
        model = self.model
        args = self.args
        pad_token_label_id = self.pad_token_label_id
        preds = None

        if split_on_space:
            if self.args.model_type in ["layoutlm", "layoutlmv2"]:
                predict_examples = [
                    InputExample(
                        i,
                        sentence.split(),
                        [self.args.labels_list[0] for word in sentence.split()],
                        x0,
                        y0,
                        x1,
                        y1,
                    )
                    for i, (sentence, x0, y0, x1, y1) in enumerate(to_predict)
                ]
                to_predict = [sentence for sentence, *_ in to_predict]
            else:
                predict_examples = [
                    InputExample(
                        i,
                        sentence.split(),
                        [self.args.labels_list[0] for word in sentence.split()],
                    )
                    for i, sentence in enumerate(to_predict)
                ]
        else:
            if self.args.model_type in ["layoutlm", "layoutlmv2"]:
                predict_examples = [
                    InputExample(
                        i,
                        sentence,
                        [self.args.labels_list[0] for word in sentence],
                        x0,
                        y0,
                        x1,
                        y1,
                    )
                    for i, (sentence, x0, y0, x1, y1) in enumerate(to_predict)
                ]
                to_predict = [sentence for sentence, *_ in to_predict]
            else:
                predict_examples = [
                    InputExample(
                        i, sentence, [self.args.labels_list[0] for word in sentence]
                    )
                    for i, sentence in enumerate(to_predict)
                ]

        if self.args.onnx:
            # Encode
            model_inputs = self.tokenizer.batch_encode_plus(
                to_predict,
                return_tensors="pt",
                padding=True,
                truncation=True,
                is_split_into_words=(not split_on_space),
            )

            eval_dataset = self.load_and_cache_examples(
                None, evaluate=True, no_cache=True, to_predict=predict_examples
            )
            eval_sampler = SequentialSampler(eval_dataset)
            eval_dataloader = DataLoader(
                eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size
            )

            eval_loss = 0.0
            nb_eval_steps = 0
            preds = None
            out_label_ids = None

            for batch in tqdm(
                eval_dataloader, disable=args.silent, desc="Running Prediction"
            ):
                with torch.no_grad():
                    inputs = self._get_inputs_dict(batch)

                encoded_model_inputs = []
                if self.args.model_type in [
                    "bert",
                    "rembert",
                    "luke",
                    "mluke",
                    "xlnet",
                    "albert",
                    "layoutlm",
                    "layoutlmv2",
                ]:
                    inputs_onnx = {
                        "input_ids": inputs["input_ids"].detach().cpu().numpy(),
                        "attention_mask": inputs["attention_mask"]
                        .detach()
                        .cpu()
                        .numpy(),
                        "token_type_ids": inputs["token_type_ids"]
                        .detach()
                        .cpu()
                        .numpy(),
                    }
                else:
                    inputs_onnx = {
                        "input_ids": inputs["input_ids"].detach().cpu().numpy(),
                        "attention_mask": inputs["attention_mask"]
                        .detach()
                        .cpu()
                        .numpy(),
                    }

                # Run the model (None = get all the outputs)
                output = self.model.run(None, inputs_onnx)

                if preds is None:
                    preds = output[0]
                    out_input_ids = inputs_onnx["input_ids"]
                    out_attention_mask = inputs_onnx["attention_mask"]
                else:
                    preds = np.append(preds, output[0], axis=0)
                    out_input_ids = np.append(
                        out_input_ids, inputs_onnx["input_ids"], axis=0
                    )
                    out_attention_mask = np.append(
                        out_attention_mask, inputs_onnx["attention_mask"], axis=0
                    )

            pad_token_label_id = -100
            out_label_ids = [[] for _ in range(len(to_predict))]
            max_len = np.max([len(x) for x in out_input_ids])

            for index, sentence in enumerate(to_predict):
                if split_on_space:
                    for word in sentence.split():
                        word_tokens = self.tokenizer.tokenize(word)
                        out_label_ids[index].extend(
                            [0] + [pad_token_label_id] * (len(word_tokens) - 1)
                        )
                else:
                    for word in sentence:
                        word_tokens = self.tokenizer.tokenize(word)
                        out_label_ids[index].extend(
                            [0] + [pad_token_label_id] * (len(word_tokens) - 1)
                        )

                out_label_ids[index].insert(0, pad_token_label_id)
                out_label_ids[index].append(pad_token_label_id)

                if len(out_label_ids[index]) < max_len:
                    out_label_ids[index].extend(
                        [-100] * (max_len - len(out_label_ids[index]))
                    )
            xfer_label_ids = np.zeros((len(out_label_ids), max_len))
            for i, out_label_id in enumerate(out_label_ids):
                for j, label in enumerate(out_label_id):
                    xfer_label_ids[i][j] = np.int32(label)
            out_label_ids = np.array(
                [list(x) for x in out_label_ids], np.int32
            ).reshape(len(out_label_ids), max_len)
        else:
            eval_dataset = self.load_and_cache_examples(
                None, to_predict=predict_examples
            )
            eval_sampler = SequentialSampler(eval_dataset)
            eval_dataloader = DataLoader(
                eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size
            )

            self._move_model_to_device()

            eval_loss = 0.0
            nb_eval_steps = 0
            preds = None
            out_label_ids = None
            model.eval()

            if args.n_gpu > 1:
                model = torch.nn.DataParallel(model)

            if self.args.fp16:
                from torch.cuda import amp

            for batch in tqdm(
                eval_dataloader, disable=args.silent, desc="Running Prediction"
            ):
                batch = tuple(t.to(device) for t in batch)

                with torch.no_grad():
                    inputs = self._get_inputs_dict(batch)

                    if self.args.fp16:
                        with amp.autocast():
                            outputs = self._calculate_loss(
                                model,
                                inputs,
                                loss_fct=self.loss_fct,
                                num_labels=self.num_labels,
                                args=self.args,
                            )
                            tmp_eval_loss, logits = outputs[:2]
                    else:
                        outputs = self._calculate_loss(
                            model,
                            inputs,
                            loss_fct=self.loss_fct,
                            num_labels=self.num_labels,
                            args=self.args,
                        )
                        tmp_eval_loss, logits = outputs[:2]

                    if self.args.n_gpu > 1:
                        tmp_eval_loss = tmp_eval_loss.mean()
                    eval_loss += tmp_eval_loss.item()

                nb_eval_steps += 1

                if preds is None:
                    preds = logits.detach().cpu().numpy()
                    out_label_ids = inputs["labels"].detach().cpu().numpy()
                    out_input_ids = inputs["input_ids"].detach().cpu().numpy()
                    out_attention_mask = inputs["attention_mask"].detach().cpu().numpy()
                else:
                    preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                    out_label_ids = np.append(
                        out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0
                    )
                    out_input_ids = np.append(
                        out_input_ids,
                        inputs["input_ids"].detach().cpu().numpy(),
                        axis=0,
                    )
                    out_attention_mask = np.append(
                        out_attention_mask,
                        inputs["attention_mask"].detach().cpu().numpy(),
                        axis=0,
                    )

            eval_loss = eval_loss / nb_eval_steps
        token_logits = preds
        preds = np.argmax(preds, axis=2)

        label_map = {i: label for i, label in enumerate(self.args.labels_list)}

        out_label_list = [[] for _ in range(out_label_ids.shape[0])]
        preds_list = [[] for _ in range(out_label_ids.shape[0])]

        for i in range(out_label_ids.shape[0]):
            for j in range(out_label_ids.shape[1]):
                if out_label_ids[i, j] != pad_token_label_id:
                    out_label_list[i].append(label_map[out_label_ids[i][j]])
                    preds_list[i].append(label_map[preds[i][j]])

        if split_on_space:
            preds = [
                [
                    {word: preds_list[i][j]}
                    for j, word in enumerate(sentence.split()[: len(preds_list[i])])
                ]
                for i, sentence in enumerate(to_predict)
            ]
        else:
            preds = [
                [
                    {word: preds_list[i][j]}
                    for j, word in enumerate(sentence[: len(preds_list[i])])
                ]
                for i, sentence in enumerate(to_predict)
            ]

        word_tokens = []
        for n, sentence in enumerate(to_predict):
            w_log = self._convert_tokens_to_word_logits(
                out_input_ids[n],
                out_label_ids[n],
                out_attention_mask[n],
                token_logits[n],
            )
            word_tokens.append(w_log)

        if split_on_space:
            model_outputs = [
                [
                    {word: word_tokens[i][j]}
                    for j, word in enumerate(sentence.split()[: len(preds_list[i])])
                ]
                for i, sentence in enumerate(to_predict)
            ]
        else:
            model_outputs = [
                [
                    {word: word_tokens[i][j]}
                    for j, word in enumerate(sentence[: len(preds_list[i])])
                ]
                for i, sentence in enumerate(to_predict)
            ]

        return preds, model_outputs

    def _convert_tokens_to_word_logits(
        self, input_ids, label_ids, attention_mask, logits
    ):
        ignore_ids = [
            self.tokenizer.convert_tokens_to_ids(self.tokenizer.pad_token),
            self.tokenizer.convert_tokens_to_ids(self.tokenizer.sep_token),
            self.tokenizer.convert_tokens_to_ids(self.tokenizer.cls_token),
        ]

        # Remove unuseful positions
        masked_ids = input_ids[(1 == attention_mask)]
        masked_labels = label_ids[(1 == attention_mask)]
        masked_logits = logits[(1 == attention_mask)]
        for id in ignore_ids:
            masked_labels = masked_labels[(id != masked_ids)]
            masked_logits = masked_logits[(id != masked_ids)]
            masked_ids = masked_ids[(id != masked_ids)]

        # Map to word logits
        word_logits = []
        tmp = []
        for n, lab in enumerate(masked_labels):
            if lab != self.pad_token_label_id:
                if n != 0:
                    word_logits.append(tmp)
                tmp = [list(masked_logits[n])]
            else:
                tmp.append(list(masked_logits[n]))
        word_logits.append(tmp)

        return word_logits

    def load_and_cache_examples(
        self, data, evaluate=False, no_cache=False, to_predict=None
    ):
        """
        Reads data_file and generates a TensorDataset containing InputFeatures. Caches the InputFeatures.
        Utility function for train() and eval() methods. Not intended to be used directly.

        Args:
            data: Path to a .txt file containing training or evaluation data OR a pandas DataFrame containing 3 columns - sentence_id, words, labels.
                    If a DataFrame is given, each sentence should be split into words, with each word assigned a tag, and with all words from the same sentence given the same sentence_id.
            evaluate (optional): Indicates whether the examples are for evaluation or for training.
            no_cache (optional): Force feature conversion and prevent caching. I.e. Ignore cached features even if present.

        """  # noqa: ignore flake8"

        process_count = self.args.process_count

        tokenizer = self.tokenizer
        args = self.args

        if not no_cache:
            no_cache = args.no_cache

        mode = "dev" if evaluate else "train"
        if self.args.use_hf_datasets and data is not None:
            if self.args.model_type in ["layoutlm", "layoutlmv2"]:
                raise NotImplementedError(
                    "HuggingFace Datasets support is not implemented for LayoutLM models"
                )
            dataset = load_hf_dataset(
                data,
                self.args.labels_list,
                self.args.max_seq_length,
                self.tokenizer,
                # XLNet has a CLS token at the end
                cls_token_at_end=bool(args.model_type in ["xlnet"]),
                cls_token=tokenizer.cls_token_id,
                cls_token_segment_id=2 if args.model_type in ["xlnet"] else 0,
                sep_token=tokenizer.sep_token_id,
                # RoBERTa uses an extra separator b/w pairs of sentences,
                # cf. github.com/pytorch/fairseq/commit/1684e166e3da03f5b600dbb7855cb98ddfcd0805
                sep_token_extra=args.model_type in MODELS_WITH_EXTRA_SEP_TOKEN,
                # PAD on the left for XLNet
                pad_on_left=bool(args.model_type in ["xlnet"]),
                pad_token=tokenizer.pad_token_id,
                pad_token_segment_id=4 if args.model_type in ["xlnet"] else 0,
                pad_token_label_id=self.pad_token_label_id,
                silent=args.silent,
                args=self.args,
            )
        else:
            if not to_predict and isinstance(data, str) and self.args.lazy_loading:
                dataset = LazyNERDataset(data, tokenizer, self.args)
            else:
                if to_predict:
                    examples = to_predict
                    no_cache = True
                else:
                    if isinstance(data, str):
                        examples = read_examples_from_file(
                            data,
                            mode,
                            bbox=(
                                True
                                if self.args.model_type in ["layoutlm", "layoutlmv2"]
                                else False
                            ),
                        )
                    else:
                        if self.args.lazy_loading:
                            raise ValueError(
                                "Input must be given as a path to a file when using lazy loading"
                            )
                        examples = get_examples_from_df(
                            data,
                            bbox=(
                                True
                                if self.args.model_type in ["layoutlm", "layoutlmv2"]
                                else False
                            ),
                        )

                cached_features_file = os.path.join(
                    args.cache_dir,
                    "cached_{}_{}_{}_{}_{}".format(
                        mode,
                        args.model_type,
                        args.max_seq_length,
                        self.num_labels,
                        len(examples),
                    ),
                )
                if not no_cache:
                    os.makedirs(self.args.cache_dir, exist_ok=True)

                if os.path.exists(cached_features_file) and (
                    (not args.reprocess_input_data and not no_cache)
                    or (
                        mode == "dev" and args.use_cached_eval_features and not no_cache
                    )
                ):
                    features = torch.load(cached_features_file, weights_only=False)
                    logger.info(
                        f" Features loaded from cache at {cached_features_file}"
                    )
                else:
                    logger.info(" Converting to features started.")
                    features = convert_examples_to_features(
                        examples,
                        self.args.labels_list,
                        self.args.max_seq_length,
                        self.tokenizer,
                        # XLNet has a CLS token at the end
                        cls_token_at_end=bool(args.model_type in ["xlnet"]),
                        cls_token=tokenizer.cls_token,
                        cls_token_segment_id=2 if args.model_type in ["xlnet"] else 0,
                        sep_token=tokenizer.sep_token,
                        # RoBERTa uses an extra separator b/w pairs of sentences,
                        # cf. github.com/pytorch/fairseq/commit/1684e166e3da03f5b600dbb7855cb98ddfcd0805
                        sep_token_extra=args.model_type in MODELS_WITH_EXTRA_SEP_TOKEN,
                        # PAD on the left for XLNet
                        pad_on_left=bool(args.model_type in ["xlnet"]),
                        pad_token=tokenizer.pad_token_id,
                        pad_token_segment_id=4 if args.model_type in ["xlnet"] else 0,
                        pad_token_label_id=self.pad_token_label_id,
                        process_count=process_count,
                        silent=args.silent,
                        use_multiprocessing=args.use_multiprocessing,
                        chunksize=args.multiprocessing_chunksize,
                        mode=mode,
                        use_multiprocessing_for_evaluation=args.use_multiprocessing_for_evaluation,
                    )

                    if not no_cache:
                        torch.save(features, cached_features_file)

                all_input_ids = torch.tensor(
                    [f.input_ids for f in features], dtype=torch.long
                )
                all_input_mask = torch.tensor(
                    [f.input_mask for f in features], dtype=torch.long
                )
                all_segment_ids = torch.tensor(
                    [f.segment_ids for f in features], dtype=torch.long
                )
                all_label_ids = torch.tensor(
                    [f.label_ids for f in features], dtype=torch.long
                )

                if self.args.model_type in ["layoutlm", "layoutlmv2"]:
                    all_bboxes = torch.tensor(
                        [f.bboxes for f in features], dtype=torch.long
                    )

                if self.args.model_type in ["layoutlm", "layoutlmv2"]:
                    dataset = TensorDataset(
                        all_input_ids,
                        all_input_mask,
                        all_segment_ids,
                        all_label_ids,
                        all_bboxes,
                    )
                else:
                    dataset = TensorDataset(
                        all_input_ids, all_input_mask, all_segment_ids, all_label_ids
                    )

        return dataset

    def convert_to_onnx(self, output_dir=None, set_onnx_arg=True):
        """Convert the model to ONNX format and save to output_dir

        Args:
            output_dir (str, optional): If specified, ONNX model will be saved to output_dir (else args.output_dir will be used). Defaults to None.
            set_onnx_arg (bool, optional): Updates the model args to set onnx=True. Defaults to True.
        """  # noqa
        if not output_dir:
            output_dir = os.path.join(self.args.output_dir, "onnx")
        os.makedirs(output_dir, exist_ok=True)

        if os.listdir(output_dir):
            raise ValueError(
                "Output directory ({}) already exists and is not empty."
                " Output directory for onnx conversion must be empty.".format(
                    output_dir
                )
            )

        onnx_model_name = os.path.join(output_dir, "onnx_model.onnx")

        with tempfile.TemporaryDirectory() as temp_dir:
            self.save_model(output_dir=temp_dir, model=self.model)

            convert(
                framework="pt",
                model=temp_dir,
                tokenizer=self.tokenizer,
                output=Path(onnx_model_name),
                pipeline_name="ner",
                opset=11,
            )

        self.args.onnx = True
        self.tokenizer.save_pretrained(output_dir)
        self.config.save_pretrained(output_dir)
        self._save_model_args(output_dir)

    def _calculate_loss(self, model, inputs, loss_fct, num_labels, args):
        outputs = model(**inputs)
        # model outputs are always tuple in pytorch-transformers (see doc)
        loss = outputs[0]
        if loss_fct:
            logits = outputs[1]
            labels = inputs["labels"]
            attention_mask = inputs.get("attention_mask")
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, num_labels)
                active_labels = torch.where(
                    active_loss,
                    labels.view(-1),
                    torch.tensor(loss_fct.ignore_index).type_as(labels),
                )
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, num_labels), labels.view(-1))
        return (loss, *outputs[1:])

    def _move_model_to_device(self):
        self.model.to(self.device)

    def _get_last_metrics(self, metric_values):
        return {metric: values[-1] for metric, values in metric_values.items()}

    def _get_inputs_dict(self, batch):
        if self.args.use_hf_datasets and isinstance(batch, dict):
            return {key: value.to(self.device) for key, value in batch.items()}
        else:
            batch = tuple(t.to(self.device) for t in batch)
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "labels": batch[3],
            }
            # XLM, DistilBERT, RoBERTa, and XLM-RoBERTa don't use segment_ids
            if self.args.model_type in [
                "bert",
                "xlnet",
                "albert",
                "layoutlm",
                "layoutlmv2",
            ]:
                inputs["token_type_ids"] = batch[2]

            if self.args.model_type in ["layoutlm", "layoutlmv2"]:
                inputs["bbox"] = batch[4]

            return inputs

    def _create_training_progress_scores(self, **kwargs):
        return collections.defaultdict(list)
        """extra_metrics = {key: [] for key in kwargs}
        training_progress_scores = {
            "global_step": [],
            "precision": [],
            "recall": [],
            "f1_score": [],
            "train_loss": [],
            "eval_loss": [],
            **extra_metrics,
        }

        return training_progress_scores"""

    def save_model(
        self, output_dir=None, optimizer=None, scheduler=None, model=None, results=None
    ):
        if not output_dir:
            output_dir = self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)

        if model and not self.args.no_save:
            # Take care of distributed/parallel training
            model_to_save = model.module if hasattr(model, "module") else model
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
            self._save_model_args(output_dir)

        if results:
            output_eval_file = os.path.join(output_dir, "eval_results.txt")
            with open(output_eval_file, "w") as writer:
                for key in sorted(results.keys()):
                    writer.write("{} = {}\n".format(key, str(results[key])))

    def _save_model_args(self, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        self.args.save(output_dir)

    def _load_model_args(self, input_dir):
        args = NERArgs()
        args.load(input_dir)
        return args

    def get_named_parameters(self):
        return [n for n, p in self.model.named_parameters()]
