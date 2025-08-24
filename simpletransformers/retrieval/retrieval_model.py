from contextlib import nullcontext
import json
import logging
import math
import os
import random
import warnings
import string
import itertools
from dataclasses import asdict
from multiprocessing import Pool, cpu_count
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import transformers
from torch.utils.tensorboard import SummaryWriter
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
    AutoModelForSequenceClassification,
)
from datasets import load_from_disk
from datasets.arrow_dataset import Dataset as HFDataset
from datasets import disable_caching

from simpletransformers.config.global_args import global_args
from simpletransformers.config.model_args import RetrievalArgs
from simpletransformers.config.utils import sweep_config_to_sweep_values
from simpletransformers.custom_models.large_representation_retrieval_model import (
    DPRContextEncoderEnhanced,
    DPRQuestionEncoderEnhanced,
    DPRContextEncoderUnifiedRR,
    DPRQuestionEncoderUnifiedRR,
)
from simpletransformers.custom_models.reranking_model import RerankingModel
from simpletransformers.custom_models.retrieval_autoencoder import Autoencoder
from simpletransformers.retrieval.beir_evaluation import BeirRetrievalModel
from simpletransformers.retrieval.retrieval_utils import (
    KLDivLossForTriplets,
    MultiNegativesLoss,
    QuartetLossV1,
    QuartetLossV3,
    calculate_mrr,
    convert_beir_columns_to_trec_format,
    get_clustered_passage_dataset,
    get_output_embeddings,
    get_prediction_passage_dataset,
    get_tas_dataset,
    load_hf_dataset,
    get_evaluation_passage_dataset,
    mean_reciprocal_rank_at_k,
    get_recall_at_k,
    RetrievalOutput,
    load_trec_format,
    embed_passages_trec_format,
    compute_rerank_similarity,
    MarginMSELoss,
    colbert_score,
    MovingLossAverage,
)
from simpletransformers.retrieval.pytrec_eval_utils import (
    convert_predictions_to_pytrec_format,
    convert_qrels_dataset_to_pytrec_format,
    convert_metric_dict_to_scores_list,
)
from simpletransformers.utils.utils import ChunkSampler

from simpletransformers.custom_models.models import ColBERTModel


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
        reranking_model_name=None,
        prediction_passages=None,
        teacher_model_name=None,
        teacher_tokenizer_name=None,
        autoencoder_model=None,
        clustering_model=None,
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
        self.unified_rr = self.args.unified_rr

        if not use_cuda:
            self.args.fp16 = False

        if self.args.larger_representations:
            if self.args.unified_rr and not self.args.unified_cross_rr:
                MODEL_CLASSES["custom"] = (
                    DPRConfig,
                    DPRContextEncoderUnifiedRR,
                    DPRQuestionEncoderUnifiedRR,
                    AutoTokenizer,
                    AutoTokenizer,
                )
            else:
                MODEL_CLASSES["dpr"] = (
                    DPRConfig,
                    DPRContextEncoderEnhanced,
                    DPRQuestionEncoderEnhanced,
                    AutoTokenizer,
                    AutoTokenizer,
                )

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

        # Using autoencoder
        if self.args.use_autoencoder:
            self.autoencoder_model = Autoencoder()
            if autoencoder_model is not None:
                # Load with PyTorch
                self.autoencoder_model.load_state_dict(
                    torch.load(os.path.join(autoencoder_model, "pytorch_model.bin"))
                )
            elif model_name:
                # PyTorch model from a PyTorch checkpoint
                self.autoencoder_model.load_state_dict(
                    torch.load(
                        os.path.join(
                            model_name, "autoencoder_model", "pytorch_model.bin"
                        )
                    )
                )
        else:
            self.autoencoder_model = None

        # if self.args.unified_cross_rr:
        #     if reranking_model_name:
        #         self.reranking_config = config_class.from_pretrained(
        #             reranking_model_name, **self.args.reranking_config
        #         )
        #         self.reranking_model = RerankingModel.from_pretrained(
        #             reranking_model_name, config=self.reranking_config
        #         )
        #     elif model_name:
        #         self.reranking_config = config_class.from_pretrained(
        #             os.path.join(model_name, "reranking_model"),
        #             **self.args.reranking_config,
        #         )
        #         self.reranking_model = RerankingModel.from_pretrained(
        #             os.path.join(model_name, "reranking_model"),
        #             config=self.reranking_config,
        #         )
        #     else:
        #         self.reranking_config = config_class(**self.args.reranking_config)
        #         self.reranking_model = RerankingModel(config=self.reranking_config)
        # else:
        #     self.reranking_config = None
        #     self.reranking_model = None

        # Using standard DPR
        if context_encoder_name:
            self.context_config = config_class.from_pretrained(
                context_encoder_name, **self.args.context_config
            )
            if self.args.context_config.get("projection_dim") is not None:
                context_encoder._keys_to_ignore_on_load_missing.append("encode_proj")
            self.context_encoder = context_encoder.from_pretrained(
                context_encoder_name, config=self.context_config
            )
            try:
                self.context_tokenizer = context_tokenizer.from_pretrained(
                    context_encoder_name
                )
            except Exception:
                self.context_tokenizer = context_tokenizer.from_pretrained(
                    context_encoder_name, use_fast=False
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

        if self.args.tie_encoders:
            self.query_encoder = self.context_encoder
            self.query_tokenizer = self.context_tokenizer
            self.query_config = self.context_config
        else:
            if query_encoder_name:
                self.query_config = config_class.from_pretrained(
                    query_encoder_name, **self.args.query_config
                )
                # if self.args.query_config.get("projection_dim") is not None:
                # query_encoder._keys_to_ignore_on_load_missing.append("encode_proj")
                self.query_encoder = query_encoder.from_pretrained(
                    query_encoder_name, config=self.query_config
                )
                try:
                    self.query_tokenizer = query_tokenizer.from_pretrained(
                        query_encoder_name
                    )
                except Exception:
                    self.query_tokenizer = query_tokenizer.from_pretrained(
                        query_encoder_name, use_fast=False
                    )
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

        if args.unified_rr:
            self.teacher_model = AutoModelForSequenceClassification.from_pretrained(
                teacher_model_name
            )
            if teacher_tokenizer_name is None:
                self.teacher_tokenizer = AutoTokenizer.from_pretrained(
                    teacher_model_name, max_len=args.max_seq_length
                )
            elif isinstance(teacher_tokenizer_name, str):
                self.teacher_tokenizer = AutoTokenizer.from_pretrained(
                    teacher_tokenizer_name, max_len=args.max_seq_length
                )
            else:
                self.teacher_tokenizer = teacher_tokenizer_name
        else:
            self.teacher_model = None
            self.teacher_tokenizer = None

        if args.mse_loss or args.reranking_kl_div_loss:
            if args.teacher_type == "colbert":
                self.teacher_model = ColBERTModel.from_pretrained(
                    teacher_model_name,
                    query_maxlen=32,
                    doc_maxlen=180,
                    mask_punctuation=True,
                    device=self.device,
                )
            elif args.teacher_type == "cross_encoder":
                self.teacher_model = AutoModelForSequenceClassification.from_pretrained(
                    teacher_model_name
                )
                self.teacher_tokenizer = AutoTokenizer.from_pretrained(
                    teacher_model_name
                )
        else:
            self.teacher_model = None

        if clustering_model is not None:
            self.clustering_model = AutoModel.from_pretrained(clustering_model)
            try:
                self.clustering_tokenizer = AutoTokenizer.from_pretrained(
                    clustering_model, max_len=args.max_seq_length
                )
            except Exception:
                self.clustering_tokenizer = AutoTokenizer.from_pretrained(
                    "bert-base-multilingual-cased",
                    max_len=args.max_seq_length,
                    # clustering_model
                )
        else:
            self.clustering_model = None
            self.clustering_tokenizer = None

        # TODO: Add support for adding special tokens to the tokenizers

        if self.args.larger_representations or self.args.include_bce_loss:
            from tokenizers.processors import TemplateProcessing

            if self.args.extra_cls_token_count > 0:
                cls_substring = (
                    " ".join(["[CLS]"] * self.args.extra_cls_token_count) + " "
                )

                cls_substring = (
                    " ".join(
                        [f"[unused{i}]" for i in range(self.args.extra_cls_token_count)]
                    )
                    + " "
                )

                special_tokens = [
                    ("[CLS]", self.context_tokenizer.cls_token_id),
                    ("[UNK]", self.context_tokenizer.unk_token_id),
                    ("[SEP]", self.context_tokenizer.sep_token_id),
                    ("[PAD]", self.context_tokenizer.pad_token_id),
                    ("[MASK]", self.context_tokenizer.mask_token_id),
                ]

                for i in range(self.args.extra_cls_token_count):
                    special_tokens.append(
                        (
                            f"[unused{i}]",
                            self.context_tokenizer.convert_tokens_to_ids(
                                f"[unused{i}]"
                            ),
                        )
                    )

                context_post_processor = TemplateProcessing(
                    single=f"[CLS] {cls_substring}$A [SEP]",
                    pair="[CLS] $A [SEP] $B:1 [SEP]:1",
                    special_tokens=special_tokens,
                )

                self.context_tokenizer._tokenizer.post_processor = (
                    context_post_processor
                )
            else:
                cls_substring = ""

            if self.args.extra_mask_token_count > 0:
                mask_substring = (
                    " ".join(["[MASK]"] * self.args.extra_mask_token_count) + " "
                )
            else:
                mask_substring = ""

            if (
                self.args.extra_cls_token_count > 0
                or self.args.extra_mask_token_count > 0
            ):
                query_post_processor = TemplateProcessing(
                    single=f"[CLS] {cls_substring}$A {mask_substring}[SEP]",
                    pair="[CLS] $A [SEP] $B:1 [SEP]:1",
                    special_tokens=special_tokens,
                )

                self.query_tokenizer._tokenizer.post_processor = query_post_processor

        self.args.model_type = model_type
        self.args.model_name = model_name
        self.context_encoder_name = context_encoder_name
        self.query_encoder_name = query_encoder_name

        if args.disable_datasets_caching:
            disable_caching()

        if prediction_passages is not None:
            if self.args.multi_head_vectors:
                (
                    self.prediction_passages,
                    self.id_to_vec_dataset,
                ) = self.get_updated_prediction_passages(prediction_passages)
            else:
                self.prediction_passages = self.get_updated_prediction_passages(
                    prediction_passages
                )
        else:
            self.prediction_passages = None
        self.id_to_vec_dataset = None

        if self.args.ance_training and not self.args.hard_negatives:
            self.args.hard_negatives = True
            warnings.warn(
                "Setting hard_negatives to True since ANCE training is enabled."
            )

        self.eval_dataset_names = None

    def train_model(
        self,
        train_data,
        output_dir=None,
        show_running_loss=True,
        args=None,
        eval_data=None,
        additional_eval_passages=None,
        relevant_docs=None,
        clustered_training=False,
        top_k_values=None,
        verbose=True,
        eval_set="dev",
        eval_dataset_names=None,
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
            relevant_docs: A list of lists or path to a JSON file of relevant documents for each query.
            eval_data (optional): A DataFrame against which evaluation will be performed when evaluate_during_training is enabled. Is required if evaluate_during_training is enabled.
            clustered_training (optional): Whether to use clustered training. Defaults to False.
            top_k_values (optional): Cutoff values for metrics. Defaults to [1, 2, 3, 5, 10].
            verbose (optional): If verbose, the training progress will be displayed. Defaults to True.
            eval_set (optional): The set of evaluation dataset to use. Defaults to 'dev'. Typically 'dev' or 'test'. Should be a list if multiple datasets are used for evaluation.
            eval_dataset_names (optional): A list of names for the evaluation datasets. This is used to identify the datasets in the evaluation results. Defaults to None.
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

        if self.args.evaluate_during_training:
            if not eval_data:
                raise ValueError(
                    "evaluate_during_training is enabled but eval_data is not specified."
                    " Pass eval_data to model.train_model() if using evaluate_during_training."
                )
            elif isinstance(eval_data, list):
                # Checks if everything is in order for multiple evaluation datasets
                if not eval_dataset_names:
                    raise ValueError(
                        "eval_dataset_names is required when multiple evaluation datasets are used."
                    )
                if len(eval_data) != len(eval_dataset_names):
                    raise ValueError(
                        "Length of eval_data and eval_dataset_names should be the same."
                    )
                if not isinstance(eval_set, list) and (
                    self.args.data_format == "beir"
                    or self.args.data_format == "msmarco"
                    or self.args.evaluate_with_beir
                ):
                    eval_set = [eval_set] * len(eval_data)
                    warnings.warn(
                        f"eval_set is not a list. Setting eval_set to {eval_set}."
                    )
                elif isinstance(eval_set, list) and len(eval_set) != len(eval_data):
                    raise ValueError(
                        "Length of eval_set should be the same as the number of evaluation datasets."
                    )
                self.eval_dataset_names = eval_dataset_names
            else:
                self.eval_dataset_names = None

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
            if self.unified_rr:
                self.teacher_model = self.teacher_model.to(kwargs["rank"])
                self.teacher_model = DDP(
                    self.teacher_model, device_ids=[kwargs["rank"]]
                )
        else:
            self._move_model_to_device()

        train_dataset = self.load_and_cache_examples(
            train_data,
            verbose=verbose,
            clustered_training=clustered_training,
            evaluate=False,
            additional_eval_passages=additional_eval_passages,
        )

        os.makedirs(output_dir, exist_ok=True)

        global_step, training_details = self.train(
            train_dataset,
            output_dir,
            show_running_loss=show_running_loss,
            eval_data=eval_data,
            additional_eval_passages=additional_eval_passages,
            relevant_docs=relevant_docs,
            clustered_training=clustered_training,
            train_data=train_data,
            top_k_values=top_k_values,
            verbose=verbose,
            eval_set=eval_set,
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
        relevant_docs=None,
        clustered_training=False,
        train_data=None,
        top_k_values=None,
        verbose=True,
        eval_set="dev",
        **kwargs,
    ):
        """
        Trains the model on train_dataset.

        Utility function to be used by the train_model() method. Not intended to be used directly.
        """

        context_model = self.context_encoder
        query_model = self.query_encoder
        args = self.args

        tb_writer = SummaryWriter(log_dir=args.tensorboard_dir)
        if self.args.batch_chunk_size is not None:
            if clustered_training or args.tas_clustering:
                raise ValueError(
                    "Chunked training is not supported with clustered training or TAS clustering."
                )
            train_sampler = ChunkSampler(
                train_dataset, self.args.batch_chunk_size, self.args.train_batch_size
            )
        else:
            train_sampler = RandomSampler(train_dataset)

        if clustered_training or args.tas_clustering:
            train_dataloader = train_dataset
            if args.curriculum_clustering:
                # Number of epochs where randomizing happens is 75% of total epochs
                randomize_epochs = int(0.75 * args.num_train_epochs)

                # Linearly decrease randomize percentage for 1 to 0 over the randomize epochs
                randomize_percentage = np.linspace(1, 0, randomize_epochs)

                # Add 0 to the end of the randomize percentage for the remaining epochs
                randomize_percentage = np.append(
                    randomize_percentage,
                    np.zeros(args.num_train_epochs - randomize_epochs),
                )
            else:
                randomize_percentage = np.zeros(args.num_train_epochs)
        else:
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
        self.t_total = t_total
        if self.args.nll_lambda_start_decay is not None:
            total_decay_steps = self.t_total - self.args.nll_lambda_start_decay
            self.nll_lambda_decay_per_step = (
                self.args.nll_lambda - self.args.nll_lambda_min
            ) / total_decay_steps

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
            self.teacher_model = torch.nn.DataParallel(self.teacher_model)

        logger.info(" Training started")

        self.global_step = 0
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
        moving_loss = MovingLossAverage(args.moving_average_loss_count)

        if args.model_name and os.path.exists(args.model_name):
            try:
                # set global_step to gobal_step of last saved checkpoint from model path
                checkpoint_suffix = args.model_name.split("/")[-1].split("-")
                if len(checkpoint_suffix) > 2:
                    checkpoint_suffix = checkpoint_suffix[1]
                else:
                    checkpoint_suffix = checkpoint_suffix[-1]
                self.global_step = int(checkpoint_suffix)
                epochs_trained = (
                    self.global_step
                    // (len(train_dataloader) // args.gradient_accumulation_steps)
                    + 1
                )
                steps_trained_in_current_epoch = self.global_step % (
                    len(train_dataloader) // args.gradient_accumulation_steps
                )

                logger.info(
                    "   Continuing training from checkpoint, will skip to saved global_step"
                )
                logger.info("   Continuing training from epoch %d", epochs_trained)
                logger.info(
                    "   Continuing training from global step %d", self.global_step
                )
                logger.info(
                    "   Will skip the first %d steps in the current epoch",
                    steps_trained_in_current_epoch,
                )
                train_iterator = trange(
                    epochs_trained,
                    int(args.num_train_epochs),
                    desc="Epoch",
                    disable=args.silent,
                    mininterval=0,
                )
            except ValueError:
                logger.info("   Starting fine-tuning.")

        if args.evaluate_during_training:
            training_progress_scores = self._create_training_progress_scores(
                calculate_recall=relevant_docs is not None,
                top_k_values=top_k_values,
                **kwargs,
            )
            if self.args.early_stopping_metric not in training_progress_scores:
                raise ValueError(
                    "early_stopping_metric should be one of the metrics used for evaluation. \
                        Available metrics are {} and the early_stopping_metric is {}".format(
                        list(training_progress_scores.keys()),
                        self.args.early_stopping_metric,
                    )
                )

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

        if args.external_embeddings:
            args.train_context_encoder = False

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
                epoch_number = current_epoch - 1
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
                # batch = tuple(t.to(device) for t in batch)
                (
                    context_inputs,
                    query_inputs,
                    labels,
                    teacher_scores,
                ) = self._get_inputs_dict(batch)

                high_loss_repeats = 0

                while True:
                    if args.fp16:
                        with amp.autocast():
                            retrieval_output = self._calculate_loss(
                                context_model,
                                query_model,
                                context_inputs,
                                query_inputs,
                                labels,
                                teacher_scores=teacher_scores,
                            )
                            loss = retrieval_output.loss
                            correct_predictions_percentage = (
                                retrieval_output.correct_predictions_percentage
                            )
                    else:
                        retrieval_output = self._calculate_loss(
                            context_model,
                            query_model,
                            context_inputs,
                            query_inputs,
                            labels,
                            teacher_scores=teacher_scores,
                        )
                        loss = retrieval_output.loss
                        correct_predictions_percentage = (
                            retrieval_output.correct_predictions_percentage
                        )
                    colbert_percentage = (
                        retrieval_output.teacher_correct_predictions_percentage
                    )

                    if args.n_gpu > 1:
                        loss = loss.mean()

                    # Compare the current loss to the moving average loss
                    current_loss = loss.item()

                    if (
                        args.repeat_high_loss_n == 0
                        or moving_loss.size() < args.moving_average_loss_count
                    ):
                        break

                    if current_loss > moving_loss.get_average_loss():
                        # Increment the high loss repeats counter
                        high_loss_repeats += 1

                        if high_loss_repeats > args.repeat_high_loss_n:
                            # Exit the loop if the high loss repeats counter exceeds the threshold
                            break
                        if args.fp16:
                            scaler.scale(loss).backward()
                        else:
                            loss.backward()

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
                            context_model.zero_grad()
                            query_model.zero_grad()
                    else:
                        # Exit the loop if the current loss is lower than the moving average loss
                        break

                if args.repeat_high_loss_n > 0:
                    moving_loss.add_loss(current_loss)

                if show_running_loss and (args.reranking_kl_div_loss or args.mse_loss):
                    batch_iterator.set_description(
                        f"Epochs {epoch_number + 1}/{args.num_train_epochs}. Running Loss: {current_loss:9.4f} Correct percentage: {correct_predictions_percentage:4.1f} Teacher correct percentage: {colbert_percentage:4.1f}"
                    )
                elif show_running_loss:
                    if args.repeat_high_loss_n > 0:
                        batch_iterator.set_description(
                            f"Epochs {epoch_number + 1}/{args.num_train_epochs}. Running Loss: {current_loss:9.4f} Correct percentage: {correct_predictions_percentage:4.1f} High loss repeats: {high_loss_repeats}"
                        )
                    else:
                        if args.include_quartet_loss:
                            batch_iterator.set_description(
                                f"Epochs {epoch_number + 1}/{args.num_train_epochs}. Running Loss: {current_loss:9.4f} Quartet loss: {retrieval_output.quartet_loss:9.4f} Correct percentage: {correct_predictions_percentage:4.1f}"
                            )
                        else:
                            batch_iterator.set_description(
                                f"Epochs {epoch_number + 1}/{args.num_train_epochs}. Running Loss: {current_loss:9.4f} Correct percentage: {correct_predictions_percentage:4.1f}"
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
                    self.global_step += 1

                    if (
                        args.logging_steps > 0
                        and self.global_step % args.logging_steps == 0
                    ):
                        # Log metrics
                        tb_writer.add_scalar(
                            "lr", scheduler.get_last_lr()[0], self.global_step
                        )
                        tb_writer.add_scalar(
                            "loss",
                            (tr_loss - logging_loss) / args.logging_steps,
                            self.global_step,
                        )
                        logging_loss = tr_loss
                        if args.wandb_project or self.is_sweeping:
                            # if args.reranking_kl_div_loss or args.mse_loss:
                            #     # Deprecated
                            #     logging_dict = {
                            #         "Training loss": current_loss,
                            #         "lr": scheduler.get_last_lr()[0],
                            #         "global_step": self.global_step,
                            #         "correct_predictions_percentage": correct_predictions_percentage,
                            #         "teacher_correct_predictions_percentage": colbert_percentage,
                            #     }
                            #     if args.include_nll_loss:
                            #         logging_dict[
                            #             "nll_loss"
                            #         ] = retrieval_output.nll_loss
                            #         if args.reranking_kl_div_loss:
                            #             logging_dict["kl_div_loss"] = (
                            #                 current_loss - retrieval_output.nll_loss
                            #             )
                            #         elif args.mse_loss:
                            #             logging_dict["mse_loss"] = (
                            #                 current_loss - retrieval_output.nll_loss
                            #             )

                            logging_dict = {
                                "Training loss": current_loss,
                                "lr": scheduler.get_last_lr()[0],
                                "global_step": self.global_step,
                                "correct_predictions_percentage": correct_predictions_percentage,
                            }

                            if retrieval_output.nll_loss is not None:
                                logging_dict["nll_loss"] = retrieval_output.nll_loss
                            if retrieval_output.kl_div_loss is not None:
                                logging_dict["kl_div_loss"] = (
                                    retrieval_output.kl_div_loss
                                )
                            if retrieval_output.margin_mse_loss is not None:
                                logging_dict["margin_mse_loss"] = (
                                    retrieval_output.margin_mse_loss
                                )
                            if retrieval_output.quartet_loss is not None:
                                logging_dict["quartet_loss"] = (
                                    retrieval_output.quartet_loss
                                )

                            wandb.log(logging_dict)

                    if args.save_steps > 0 and self.global_step % args.save_steps == 0:
                        # Save model checkpoint
                        output_dir_current = os.path.join(
                            output_dir, "checkpoint-{}".format(self.global_step)
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
                        and self.global_step % args.evaluate_during_training_steps == 0
                    ):
                        # Only evaluate when single GPU otherwise metrics may not average well
                        if args.data_format == "beir" or args.data_format == "msmarco":
                            if isinstance(eval_data, list):
                                results = {}
                                for eval_data_item, eval_name, current_set in zip(
                                    eval_data, self.eval_dataset_names, eval_set
                                ):
                                    # We need to keep adding the results to the same dictionary
                                    # as we are evaluating on multiple datasets
                                    results_default_names = self.eval_model(
                                        eval_data_item,
                                        verbose=verbose
                                        and args.evaluate_during_training_verbose,
                                        silent=args.evaluate_during_training_silent,
                                        evaluating_during_training=True,
                                        eval_set=current_set,
                                        **kwargs,
                                    )

                                    for key in results_default_names:
                                        results[f"{eval_name}_{key}"] = (
                                            results_default_names[key]
                                        )
                            else:
                                results = self.eval_model(
                                    eval_data,
                                    verbose=verbose
                                    and args.evaluate_during_training_verbose,
                                    silent=args.evaluate_during_training_silent,
                                    evaluating_during_training=True,
                                    eval_set=eval_set,
                                    **kwargs,
                                )
                        else:
                            results, *_ = self.eval_model(
                                eval_data,
                                additional_passages=additional_eval_passages,
                                relevant_docs=relevant_docs,
                                verbose=verbose
                                and args.evaluate_during_training_verbose,
                                silent=args.evaluate_during_training_silent,
                                top_k_values=top_k_values,
                                evaluating_during_training=True,
                                **kwargs,
                            )
                        for key, value in results.items():
                            try:
                                tb_writer.add_scalar(
                                    "eval_{}".format(key), value, self.global_step
                                )
                            except (NotImplementedError, AssertionError):
                                pass

                        output_dir_current = os.path.join(
                            output_dir, "checkpoint-{}".format(self.global_step)
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

                        training_progress_scores["global_step"].append(self.global_step)
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
                                            self.global_step,
                                            (
                                                tr_loss / self.global_step
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
                                            self.global_step,
                                            (
                                                tr_loss / self.global_step
                                                if not self.args.evaluate_during_training
                                                else training_progress_scores
                                            ),
                                        )
                        context_model.train()
                        query_model.train()

            epoch_number += 1
            output_dir_current = os.path.join(
                output_dir,
                "checkpoint-{}-epoch-{}".format(self.global_step, epoch_number),
            )

            if (
                clustered_training or self.args.ance_training
            ) and epoch_number != args.num_train_epochs:
                train_dataset = self.load_and_cache_examples(
                    train_data,
                    verbose=verbose,
                    clustered_training=clustered_training,
                    evaluate=False,
                    additional_eval_passages=additional_eval_passages,
                    clustered_batch_randomize_percentage=(
                        randomize_percentage[epoch_number]
                        if clustered_training
                        else None
                    ),
                    epoch_number=epoch_number,
                    dataset=train_dataset,
                )
                train_sampler = RandomSampler(train_dataset)
                if clustered_training:
                    train_dataloader = train_dataset
                else:
                    train_dataloader = DataLoader(
                        train_dataset,
                        sampler=train_sampler,
                        batch_size=args.train_batch_size,
                        num_workers=self.args.dataloader_num_workers,
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
                if args.data_format == "beir" or args.data_format == "msmarco":
                    if isinstance(eval_data, list):
                        results = {}
                        for eval_data_item, eval_name, current_set in zip(
                            eval_data, self.eval_dataset_names, eval_set
                        ):
                            # We need to keep adding the results to the same dictionary
                            # as we are evaluating on multiple datasets
                            results_default_names = self.eval_model(
                                eval_data_item,
                                verbose=verbose
                                and args.evaluate_during_training_verbose,
                                silent=args.evaluate_during_training_silent,
                                evaluating_during_training=True,
                                eval_set=current_set,
                                **kwargs,
                            )

                            for key in results_default_names:
                                results[f"{eval_name}_{key}"] = results_default_names[
                                    key
                                ]
                    else:
                        results = self.eval_model(
                            eval_data,
                            verbose=verbose and args.evaluate_during_training_verbose,
                            silent=args.evaluate_during_training_silent,
                            evaluating_during_training=True,
                            eval_set=eval_set,
                            **kwargs,
                        )
                else:
                    results, *_ = self.eval_model(
                        eval_data,
                        additional_passages=additional_eval_passages,
                        relevant_docs=relevant_docs,
                        verbose=verbose and args.evaluate_during_training_verbose,
                        silent=args.evaluate_during_training_silent,
                        top_k_values=top_k_values,
                        evaluating_during_training=True,
                        **kwargs,
                    )

                if args.save_eval_checkpoints:
                    self.save_model(
                        output_dir_current, optimizer, scheduler, results=results
                    )

                training_progress_scores["global_step"].append(self.global_step)
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
                                    self.global_step,
                                    (
                                        tr_loss / self.global_step
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
                                    self.global_step,
                                    (
                                        tr_loss / self.global_step
                                        if not self.args.evaluate_during_training
                                        else training_progress_scores
                                    ),
                                )

        return (
            self.global_step,
            (
                tr_loss / self.global_step
                if not self.args.evaluate_during_training
                else training_progress_scores
            ),
        )

    def eval_model(
        self,
        eval_data,
        evaluate_with_all_passages=True,
        additional_passages=None,
        relevant_docs=None,
        top_k_values=None,
        retrieve_n_docs=None,
        return_doc_dicts=True,
        passage_dataset=None,
        qa_evaluation=False,
        output_dir=None,
        verbose=True,
        silent=False,
        evaluating_during_training=False,
        pytrec_eval_metrics=None,
        save_as_experiment=False,
        experiment_name=None,
        dataset_name=None,
        model_name=None,
        eval_set="dev",
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
            relevant_docs: A list of lists or path to a JSON file of relevant documents for each query.
            top_k_values: List of top-k values to be used for evaluation.
            retrieve_n_docs: Number of documents to retrieve for each query. Overrides `args.retrieve_n_docs` for this evaluation.
            return_doc_dicts: If True, return the doc dicts for the retrieved passages. Setting this to False can speed up evaluation.
            passage_dataset: Path to a saved Huggingface dataset (containing generated embeddings) for both the eval_data and additional passages
            qa_evaluation: If True, evaluation is done by checking if the retrieved passages contain the gold passage.
            output_dir: The directory where model files will be saved. If not given, self.args.output_dir will be used.
            verbose: If verbose, results will be printed to the console on completion of evaluation.
            silent: If silent, tqdm progress bars will be hidden.
            evaluating_during_training: Set to True to perform evaluation during training.
            pytrec_eval_metrics: A list of pytrec_eval metrics to use. Only valid if `data_format == "beir"`.
            save_as_experiment: If True, `experiment_name`, `dataset_name`, and `model_name` must be provided. Pytrec_eval output will be saved to `experiment_name`.
            experiment_name: Name of the experiment. Only valid if `save_as_experiment == True`.
            dataset_name: Name of the dataset. Only valid if `save_as_experiment == True`.
            model_name: Name of the model. Only valid if `save_as_experiment == True`.
            eval_set: "dev" or "test".
            **kwargs: Additional metrics that should be used. Pass in the metrics as keyword arguments (name of metric: function to use).
                        A metric function should take in two parameters. The first parameter will be the true labels, and the second parameter will be the predictions. Both inputs
                        will be lists of strings. Note that this will slow down evaluation significantly as the predicted sequences need to be generated.
        Returns:
            result: Dictionary containing evaluation results.
            doc_ids: List of retrieved document IDs.
            doc_dicts: List of retrieved document dictionaries.
            top_k_accuracy_each_query: List of top-k accuracy for each query.
            recall_at_k_each_query: List of recall at k for each query.
            relevance_list: Array of relevance hits for each query.
        """  # noqa: ignore flake8"

        if save_as_experiment:
            if not experiment_name or not dataset_name or not model_name:
                raise ValueError(
                    "experiment_name, dataset_name, and model_name must be provided if save_as_experiment is True"
                )
        elif (experiment_name or dataset_name or model_name) and not save_as_experiment:
            raise ValueError(
                "experiment_name, dataset_name, and model_name provided but save_as_experiment is False. Please set save_as_experiment to True."
            )

        if not output_dir:
            output_dir = self.args.output_dir

        self._move_model_to_device(is_evaluating=True)

        context_encoder_was_training = self.context_encoder.training
        query_encoder_was_training = self.query_encoder.training
        self.context_encoder.eval()
        self.query_encoder.eval()

        if self.args.evaluate_with_beir:
            results = self.evaluate_beir(
                eval_data,
                eval_set=eval_set,
            )

            return results, None, None, None, None, None

        if self.args.data_format == "beir" or self.args.data_format == "msmarco":
            try:
                import pytrec_eval
            except ImportError:
                logger.error(
                    "pytrec_eval not installed. Please install with `pip install pytrec_eval`. (See https://github.com/cvangysel/pytrec_eval)"
                )
                return

            if passage_dataset != "preloaded":
                passage_dataset, query_dataset, qrels_dataset = load_trec_format(
                    eval_data, qrels_name=eval_set, data_format=self.args.data_format
                )
            else:
                _, query_dataset, qrels_dataset = load_trec_format(
                    eval_data,
                    qrels_name=eval_set,
                    data_format=self.args.data_format,
                    skip_passages=True,
                )

            if self.args.data_format == "beir" and passage_dataset != "preloaded":
                (
                    passage_dataset,
                    query_dataset,
                    qrels_dataset,
                ) = convert_beir_columns_to_trec_format(
                    passage_dataset,
                    query_dataset,
                    qrels_dataset,
                    include_titles=self.args.include_title_in_corpus,
                )

            if passage_dataset != "preloaded":
                passage_index = embed_passages_trec_format(
                    passage_dataset,
                    self.context_encoder,
                    self.context_tokenizer,
                    args=self.args,
                    context_config=self.context_config,
                    device=self.device,
                    autoencoder=self.autoencoder_model,
                )
                if self.args.multi_head_vectors:
                    passage_index, id_to_vec_dataset = passage_index
                else:
                    id_to_vec_dataset = None
                self.prediction_passages = passage_index
            else:
                id_to_vec_dataset = None

            query_text_column = (
                "text" if self.args.data_format == "beir" else "query_text"
            )

            predicted_doc_ids, scores = self.predict(
                to_predict=query_dataset[query_text_column],
                doc_ids_only=True,
                id_to_vec_dataset=id_to_vec_dataset,
            )

            run_dict = convert_predictions_to_pytrec_format(
                predicted_doc_ids,
                query_dataset,
                id_column="_id" if self.args.data_format == "beir" else "query_id",
                predicted_scores=scores,
            )

            qrels_dict = convert_qrels_dataset_to_pytrec_format(qrels_dataset)

            if pytrec_eval_metrics is None:
                pytrec_eval_metrics = self.args.pytrec_eval_metrics

            if "mrr" in pytrec_eval_metrics:
                custom_mrr = True
                pytrec_eval_metrics.remove("mrr")
            else:
                custom_mrr = False

            evaluator = pytrec_eval.RelevanceEvaluator(
                qrels_dict,
                pytrec_eval_metrics,
                relevance_level=self.args.relevance_level,
            )

            try:
                results = evaluator.evaluate(run_dict)
            except:
                # Convert run_dict keys to strings
                run_dict = {
                    str(key): {str(k): v for k, v in value.items()}
                    for key, value in run_dict.items()
                }
                results = evaluator.evaluate(run_dict)

            if save_as_experiment:
                os.makedirs(
                    os.path.join(experiment_name, dataset_name, model_name),
                    exist_ok=True,
                )
            result_report = {}

            for metric in pytrec_eval_metrics:
                per_metric_dict = {
                    query_id: value[metric] for query_id, value in results.items()
                }
                mean_metric = np.mean(list(per_metric_dict.values()))
                result_report[metric] = mean_metric

                if save_as_experiment:
                    with open(
                        os.path.join(
                            experiment_name, dataset_name, model_name, f"{metric}.json"
                        ),
                        "w",
                    ) as f:
                        json.dump(per_metric_dict, f)

            if custom_mrr:
                # TODO: Implement custom MRR
                pass

            if save_as_experiment:
                with open(
                    os.path.join(
                        experiment_name, dataset_name, model_name, "results.json"
                    ),
                    "w",
                ) as f:
                    json.dump(result_report, f)

                # Save run_dict
                with open(
                    os.path.join(
                        experiment_name, dataset_name, model_name, "run_dict.json"
                    ),
                    "w",
                ) as f:
                    json.dump(run_dict, f)

            if context_encoder_was_training:
                self.context_encoder.train()
            if query_encoder_was_training:
                self.query_encoder.train()

            return result_report

        if self.prediction_passages is None or evaluating_during_training:
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

        if relevant_docs is not None:
            if isinstance(relevant_docs, str):
                relevant_docs = json.load(open(relevant_docs, "r"))
            elif isinstance(relevant_docs[0], list):
                pass
            else:
                raise ValueError(
                    "relevant_docs must be a list of lists or a path to a JSON file"
                )

        (
            result,
            doc_ids,
            doc_vectors,
            doc_dicts,
            top_k_accuracy_each_query,
            recall_at_k_each_query,
            mrr_each_query_dict,
            relevance_list,
        ) = self.evaluate(
            eval_dataset,
            gold_passages,
            evaluate_with_all_passages,
            passage_dataset,
            relevant_docs,
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

        if context_encoder_was_training:
            self.context_encoder.train()
        if query_encoder_was_training:
            self.query_encoder.train()

        return (
            result,
            doc_ids,
            doc_vectors,
            doc_dicts,
            top_k_accuracy_each_query,
            recall_at_k_each_query,
            mrr_each_query_dict,
            relevance_list,
        )

    def evaluate(
        self,
        eval_dataset,
        gold_passages,
        evaluate_with_all_passages=True,
        passage_dataset=None,
        relevant_docs=None,
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
            self.teacher_model = torch.nn.DataParallel(self.teacher_model)

        nb_eval_steps = 0
        eval_loss = 0
        context_model.eval()
        query_model.eval()

        criterion = torch.nn.NLLLoss(reduction="mean")

        if self.args.fp16:
            from torch.cuda import amp

        if args.larger_representations:
            all_query_embeddings = np.zeros(
                (
                    len(eval_dataset),
                    self.query_config.hidden_size * (1 + args.extra_cls_token_count),
                )
            )
        else:
            all_query_embeddings = np.zeros(
                (
                    len(eval_dataset),
                    (
                        self.query_config.hidden_size
                        if "projection_dim" not in self.query_config.to_dict()
                        or not self.query_config.projection_dim
                        else self.query_config.projection_dim
                    ),
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

            context_inputs, query_inputs, labels, _ = self._get_inputs_dict(
                batch, evaluate=True
            )
            with torch.no_grad():
                if self.args.fp16:
                    with amp.autocast():
                        retrieval_outputs = self._calculate_loss(
                            context_model,
                            query_model,
                            context_inputs,
                            query_inputs,
                            labels,
                        )
                else:
                    retrieval_outputs = self._calculate_loss(
                        context_model,
                        query_model,
                        context_inputs,
                        query_inputs,
                        labels,
                    )

                tmp_eval_loss = retrieval_outputs.loss
                query_outputs = retrieval_outputs.query_outputs
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

            top_k_accuracy_each_query = None
            recall_at_k_each_query = None
            if relevant_docs is not None:
                (
                    scores,
                    top_k_accuracy_each_query,
                    recall_at_k_each_query,
                    mrr_each_query_dict,
                    relevance_list,
                ) = self.compute_metrics(
                    gold_passages,
                    doc_texts,
                    relevant_docs,
                    self.args,
                    qa_evaluation,
                    top_k_values,
                    **kwargs,
                )
            else:
                (
                    scores,
                    top_k_accuracy_each_query,
                    mrr_each_query_dict,
                    relevance_list,
                ) = self.compute_metrics(
                    gold_passages,
                    doc_texts,
                    relevant_docs,
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

        return (
            results,
            doc_ids,
            doc_vectors,
            doc_dicts,
            top_k_accuracy_each_query,
            recall_at_k_each_query,
            mrr_each_query_dict,
            relevance_list,
        )

    def evaluate_beir(
        self,
        eval_data,
        eval_set="dev",
    ):
        from beir.datasets.data_loader import GenericDataLoader
        from beir.retrieval.evaluation import EvaluateRetrieval
        from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES

        corpus, queries, qrels = GenericDataLoader(data_folder=eval_data).load(
            split=eval_set
        )

        beir_model = DRES(
            BeirRetrievalModel(
                self.context_encoder,
                self.query_encoder,
                self.context_tokenizer,
                self.query_tokenizer,
                self.context_config,
                self.query_config,
                self.args,
            )
        )

        retriever = EvaluateRetrieval(
            beir_model,
            score_function="dot",
        )

        results = retriever.retrieve(corpus, queries)

        ndcg_b, _map_b, recall_b, precision_b = retriever.evaluate(
            qrels, results, retriever.k_values
        )
        mrr_b = retriever.evaluate_custom(
            qrels, results, retriever.k_values, metric="mrr"
        )

        ndcg = {}
        _map = {}
        recall = {}
        precision = {}
        mrr = {}

        for i, k in enumerate(retriever.k_values):
            ndcg["ndcg_at_" + str(k)] = ndcg_b["NDCG@" + str(k)]
            _map["map_at_" + str(k)] = _map_b["MAP@" + str(k)]
            recall["recall_at_" + str(k)] = recall_b["Recall@" + str(k)]
            precision["precision_at_" + str(k)] = precision_b["P@" + str(k)]
            mrr["mrr_at_" + str(k)] = mrr_b["MRR@" + str(k)]

        return {
            **ndcg,
            **_map,
            **recall,
            **precision,
            **mrr,
            "eval_loss": -1,
        }

    def predict(
        self,
        to_predict,
        prediction_passages=None,
        retrieve_n_docs=None,
        passages_only=False,
        doc_ids_only=False,
        is_training=False,
        id_to_vec_dataset=None,
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
        if self.prediction_passages is None or (
            self.args.ance_training and is_training
        ):
            if prediction_passages is None:
                raise ValueError(
                    "prediction_passages cannot be None if the model does not contain a predicition passage index."
                )
            else:
                if self.args.ance_training and is_training:
                    logger.info(
                        "Updating corpus embeddings for ANCE training. This may take a while."
                    )
                self.context_encoder.to(self.device)
                self.context_encoder.eval()
                self.prediction_passages = self.get_updated_prediction_passages(
                    prediction_passages
                )
                self.context_encoder.to(self.device)
                if self.args.ance_training and is_training:
                    logger.info("Done updating corpus embeddings.")

        all_reranking_query_embeddings = None

        if self.args.larger_representations:
            if self.unified_rr:
                all_query_embeddings = np.zeros(
                    (len(to_predict), self.query_config.hidden_size)
                )
                all_reranking_query_embeddings = np.zeros(
                    (len(to_predict), self.query_config.hidden_size)
                )
            else:
                all_query_embeddings = np.zeros(
                    (
                        len(to_predict),
                        self.query_config.hidden_size
                        * (1 + self.args.extra_cls_token_count),
                    )
                )
        elif self.args.multi_head_vectors:
            num_heads = 12
            head_dim = 64
            all_query_embeddings = torch.zeros((len(to_predict), num_heads, head_dim))
        else:
            all_query_embeddings = np.zeros(
                (
                    len(to_predict),
                    (
                        self.query_config.hidden_size
                        if "projection_dim" not in self.query_config.to_dict()
                        or not self.query_config.projection_dim
                        else self.query_config.projection_dim
                    ),
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
            total=math.ceil(len(to_predict) / self.args.eval_batch_size),
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
                        query_outputs = query_model(**query_inputs)
                        query_outputs = get_output_embeddings(
                            query_outputs,
                            concatenate_embeddings=self.args.larger_representations
                            and self.args.model_type == "custom",
                            n_cls_tokens=(1 + self.args.extra_cls_token_count),
                            use_pooler_output=self.args.use_pooler_output,
                            args=self.args,
                            return_all_embeddings=self.args.use_autoencoder,
                            input_mask=query_inputs["attention_mask"],
                        )
                        if self.args.use_autoencoder:
                            query_outputs = self.autoencoder_model.encode(query_outputs)
                else:
                    query_outputs = query_model(**query_inputs)
                    query_outputs = get_output_embeddings(
                        query_outputs,
                        concatenate_embeddings=self.args.larger_representations
                        and self.args.model_type == "custom",
                        n_cls_tokens=(1 + self.args.extra_cls_token_count),
                        use_pooler_output=self.args.use_pooler_output,
                        args=self.args,
                        query_embeddings=True,
                        input_mask=query_inputs["attention_mask"],
                    )

            if self.unified_rr:
                reranking_query_outputs = query_outputs[
                    :, query_outputs.size(1) // 2 :
                ].cpu()
                query_outputs = query_outputs[:, : query_outputs.size(1) // 2]
            else:
                reranking_query_outputs = None

            all_query_embeddings[
                i * self.args.eval_batch_size : (i + 1) * self.args.eval_batch_size
            ] = (
                (query_outputs.cpu().detach().numpy())
                if not self.args.multi_head_vectors
                else query_outputs.cpu().detach()
            )

            if self.unified_rr:
                all_reranking_query_embeddings[
                    i * self.args.eval_batch_size : (i + 1) * self.args.eval_batch_size
                ] = (reranking_query_outputs.cpu().detach().numpy())

        if self.args.multi_head_vectors:
            # Right now, shape is (n_queries, 12, 64)
            # Need to convert to (n_queries * 12, 64)
            all_query_embeddings = all_query_embeddings.reshape(-1, head_dim)

        if passages_only:
            passages = self.retrieve_docs_from_query_embeddings(
                all_query_embeddings,
                self.prediction_passages,
                retrieve_n_docs,
                passages_only=True,
            )
            return passages
        elif doc_ids_only:
            doc_ids, scores = self.retrieve_docs_from_query_embeddings(
                all_query_embeddings,
                self.prediction_passages,
                retrieve_n_docs,
                doc_ids_only=True,
                reranking_query_outputs=all_reranking_query_embeddings,
            )
            return doc_ids, scores
        else:
            retrieval_outputs = self.retrieve_docs_from_query_embeddings(
                all_query_embeddings, self.prediction_passages, retrieve_n_docs
            )
            doc_ids, doc_vectors, doc_dicts = retrieval_outputs

            try:
                passages = [d["passages"] for d in doc_dicts]
            except KeyError:
                passages = [d["passage_text"] for d in doc_dicts]

            if self.args.unified_rr:
                rerank_similarity = compute_rerank_similarity(
                    reranking_query_outputs, doc_dicts
                )

                # Get indices of rerank_similarity sorted by descending order
                rerank_indices = np.argsort(rerank_similarity, axis=1)[:, ::-1]

                # Sort passages, doc_ids, doc_vectors, doc_dicts by rerank_indices
                for i, doc_dict in enumerate(doc_dicts):
                    doc_dict["passages"] = [
                        doc_dict["passages"][j] for j in rerank_indices[i]
                    ]
                    doc_dict["embeddings"] = [
                        doc_dict["embeddings"][j] for j in rerank_indices[i]
                    ]
                    doc_dict["rerank_embeddings"] = [
                        doc_dict["rerank_embeddings"][j] for j in rerank_indices[i]
                    ]

                passages = [
                    [passages[i][j] for j in rerank_indices[i]]
                    for i in range(len(passages))
                ]
                doc_ids = [
                    [doc_ids[i][j] for j in rerank_indices[i]]
                    for i in range(len(doc_ids))
                ]
                doc_vectors = [
                    [doc_vectors[i][j] for j in rerank_indices[i]]
                    for i in range(len(doc_vectors))
                ]

            return passages, doc_ids, doc_vectors, doc_dicts

    def compute_metrics(
        self,
        gold_passages,
        doc_texts,
        relevant_docs,
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

        if max(top_k_values) > args.retrieve_n_docs:
            raise ValueError(
                "retrieve_n_docs must be >= max(top_k_values). top_k_values: {}, retrieve_n_docs: {}".format(
                    top_k_values, args.retrieve_n_docs
                )
            )

        top_k_values = [k for k in top_k_values if k <= args.retrieve_n_docs]

        relevance_list_first_hit = np.zeros((len(gold_passages), args.retrieve_n_docs))
        relevance_list_all_hits = np.zeros((len(gold_passages), args.retrieve_n_docs))

        if relevant_docs is None:
            for i, (docs, truth) in enumerate(zip(doc_texts, gold_passages)):
                for j, d in enumerate(docs):
                    if qa_evaluation:
                        if truth.strip().lower().replace(" ", "").translate(
                            str.maketrans("", "", string.punctuation)
                        ) in d.strip().lower().replace(" ", "").translate(
                            str.maketrans("", "", string.punctuation)
                        ):
                            relevance_list_first_hit[i, j] = 1
                            break
                    else:
                        if d.strip().lower().translate(
                            str.maketrans("", "", string.punctuation)
                        ) == truth.strip().lower().translate(
                            str.maketrans("", "", string.punctuation)
                        ):
                            relevance_list_first_hit[i, j] = 1
                            break
            relevance_list_all_hits = relevance_list_first_hit
        else:
            total_relevant = [
                len(relevant_doc_set) for relevant_doc_set in relevant_docs
            ]
            for i, (docs, relevant_doc_set) in enumerate(zip(doc_texts, relevant_docs)):
                for j, d in enumerate(docs):
                    for relevant in relevant_doc_set:
                        if qa_evaluation:
                            if relevant.strip().lower().replace(" ", "").translate(
                                str.maketrans("", "", string.punctuation)
                            ) in d.strip().lower().replace(" ", "").translate(
                                str.maketrans("", "", string.punctuation)
                            ):
                                relevance_list_all_hits[i, j] = 1
                                if sum(relevance_list_first_hit[i]) == 0:
                                    relevance_list_first_hit[i, j] = 1
                                break
                        else:
                            if d.strip().lower().translate(
                                str.maketrans("", "", string.punctuation)
                            ) == relevant.strip().lower().translate(
                                str.maketrans("", "", string.punctuation)
                            ):
                                relevance_list_all_hits[i, j] = 1
                                if sum(relevance_list_first_hit[i]) == 0:
                                    relevance_list_first_hit[i, j] = 1
                                break

        mrr_each_query_dict = {}
        mrr = {}
        for k in top_k_values:
            mrr_at_k, mrr_matrix_k = mean_reciprocal_rank_at_k(
                relevance_list_first_hit, k, return_individual_scores=True
            )
            mrr[f"mrr_at_{k}"] = mrr_at_k
            mrr_each_query_dict[f"mrr_at_{k}"] = mrr_matrix_k

        top_k_accuracy_dict = {}
        top_k_accuracy_each_query_dict = {}
        recall_at_k_dict = {}
        recall_at_k_each_query_dict = {}

        for k in top_k_values:
            top_k_accuracy_each_query = np.sum(relevance_list_first_hit[:, :k], axis=1)
            top_k_accuracy_dict[f"top_{k}_accuracy"] = np.mean(
                top_k_accuracy_each_query
            )
            top_k_accuracy_each_query_dict[f"top_{k}_accuracy"] = (
                top_k_accuracy_each_query.tolist()
            )

            if relevant_docs is not None:
                recall_at_k, recall_at_k_each_query = get_recall_at_k(
                    relevance_list_all_hits, total_relevant, k
                )
                recall_at_k_dict[f"recall_at_{k}"] = recall_at_k
                recall_at_k_each_query_dict[f"recall_at_{k}"] = recall_at_k_each_query

        extra_metrics = {}
        for metric, func in kwargs.items():
            extra_metrics[metric] = func(gold_passages, doc_texts)

        if relevant_docs is not None:
            return (
                {**mrr, **top_k_accuracy_dict, **recall_at_k_dict, **extra_metrics},
                top_k_accuracy_each_query_dict,
                recall_at_k_each_query_dict,
                mrr_each_query_dict,
                relevance_list_all_hits,
            )
        else:
            return (
                {**mrr, **top_k_accuracy_dict, **extra_metrics},
                top_k_accuracy_each_query_dict,
                mrr_each_query_dict,
                relevance_list_all_hits,
            )

    def retrieve_docs_from_query_embeddings(
        self,
        query_embeddings,
        passage_dataset,
        retrieve_n_docs=None,
        return_doc_dicts=True,
        passages_only=False,
        doc_ids_only=False,
        reranking_query_outputs=None,
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

        if reranking_query_outputs is not None:
            reranking_query_outputs_batched = [
                reranking_query_outputs[i : i + args.retrieval_batch_size]
                for i in range(
                    0, len(reranking_query_outputs), args.retrieval_batch_size
                )
            ]

        if passages_only:
            passages = []
            for i, query_embeddings_retr in enumerate(
                tqdm(
                    query_embeddings_batched,
                    desc="Retrieving docs",
                    disable=args.silent,
                )
            ):
                doc_dicts_batch = passage_dataset.get_top_docs(
                    query_embeddings_retr.astype(np.float32),
                    retrieve_n_docs,
                    passages_only=True,
                )

                try:
                    passages.extend([d["passages"] for d in doc_dicts_batch])
                except KeyError:
                    passages.extend([d["passage_text"] for d in doc_dicts_batch])

            return passages
        elif doc_ids_only:
            doc_ids_batched = []
            scores_batched = []

            for i, query_embeddings_retr in enumerate(
                tqdm(
                    query_embeddings_batched,
                    desc="Retrieving docs",
                    disable=args.silent,
                )
            ):
                ids, scores = passage_dataset.get_top_doc_ids(
                    query_embeddings_retr.astype(np.float32), retrieve_n_docs
                )
                doc_ids_batched.extend(ids)
                scores_batched.extend(scores)

            reranking_scores = scores_batched

            return doc_ids_batched, reranking_scores
        else:
            ids_batched = np.zeros((len(query_embeddings), retrieve_n_docs))
            if self.args.larger_representations:
                vectors_batched = np.zeros(
                    (
                        len(query_embeddings),
                        retrieve_n_docs,
                        self.query_config.hidden_size
                        * (1 + args.extra_cls_token_count),
                    )
                )
            else:
                vectors_batched = np.zeros(
                    (
                        len(query_embeddings),
                        retrieve_n_docs,
                        (
                            self.context_config.hidden_size
                            if "projection_dim" not in self.context_config.to_dict()
                            or not self.context_config.projection_dim
                            else self.context_config.projection_dim
                        ),
                    )
                )
            doc_dicts = []

            for i, query_embeddings_retr in enumerate(
                tqdm(
                    query_embeddings_batched,
                    desc="Retrieving docs",
                    disable=args.silent,
                )
            ):
                ids, vectors, doc_dicts_batch = passage_dataset.get_top_docs(
                    query_embeddings_retr.astype(np.float32), retrieve_n_docs
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

    def get_hard_negatives(
        self,
        queries,
        passage_dataset=None,
        retrieve_n_docs=None,
        passages_only=False,
    ):
        if passages_only:
            hard_negatives = self.predict(
                to_predict=queries,
                prediction_passages=passage_dataset,
                retrieve_n_docs=retrieve_n_docs,
                passages_only=True,
                is_training=True,
            )
        else:
            hard_negatives, *_ = self.predict(
                to_predict=queries,
                prediction_passages=passage_dataset,
                retrieve_n_docs=retrieve_n_docs,
                is_training=True,
            )

        if retrieve_n_docs is None:
            retrieve_n_docs = self.args.retrieve_n_docs

        return hard_negatives

    def build_hard_negatives(
        self,
        queries,
        passage_dataset=None,
        retrieve_n_docs=None,
        write_to_disk=True,
        hard_negatives_save_file_path=None,
    ):
        hard_negatives = self.get_hard_negatives(
            queries,
            passage_dataset=passage_dataset,
            retrieve_n_docs=retrieve_n_docs,
        )

        # Build hard negative df from list of lists
        column_names = [f"hard_negatives_{i}" for i in range(retrieve_n_docs)]
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

    def add_hard_negatives_for_ance(
        self,
        train_data,
        passage_dataset=None,
    ):
        hard_negatives = self.get_hard_negatives(
            train_data["query_text"].tolist(),
            passage_dataset=passage_dataset,
            retrieve_n_docs=self.args.retrieve_n_docs,
            passages_only=True,
        )

        hard_negative_list = []
        hn_found = False
        for hns, gold in zip(hard_negatives, train_data["gold_passage"]):
            # Randomly sample a hn from hns and check if equal to gold
            for _ in range(len(hns)):
                hn = random.choice(hns)
                if hn != gold:
                    hard_negative_list.append(hn)
                    hn_found = True
                    break

            if not hn_found:
                hard_negative_list.append(hns[-1])
                hn_found = False

        train_data["hard_negative"] = hard_negative_list

        return train_data

    def load_and_cache_examples(
        self,
        data,
        evaluate=False,
        no_cache=False,
        verbose=True,
        silent=False,
        clustered_training=False,
        additional_eval_passages=None,
        clustered_batch_randomize_percentage=0.0,
        epoch_number=None,
        dataset=None,
    ):
        """
        Creates a IRDataset from data
        """

        if not no_cache:
            no_cache = self.args.no_cache

        if not no_cache:
            os.makedirs(self.args.cache_dir, exist_ok=True)

        if self.args.use_hf_datasets:
            if self.args.ance_training and not evaluate:
                if (
                    epoch_number is None
                    or epoch_number % self.args.ance_refresh_n_epochs == 0
                ):
                    logger.info("Adding hard negatives for ANCE training.")
                    data = self.add_hard_negatives_for_ance(
                        data,
                        passage_dataset=additional_eval_passages,
                    )
                    logger.info("Finished adding hard negatives for ANCE training.")
                else:
                    logger.info(
                        "Not updating hard negatives for ANCE training. {} epochs until next refresh.".format(
                            self.args.ance_refresh_n_epochs
                            - (epoch_number % self.args.ance_refresh_n_epochs)
                        )
                    )
            if dataset is None or epoch_number % self.args.cluster_every_n_epochs == 0:
                dataset = load_hf_dataset(
                    data,
                    self.context_tokenizer,
                    self.query_tokenizer,
                    self.args,
                    teacher_tokenizer=(
                        None
                        if self.args.teacher_type == "colbert"
                        else self.teacher_tokenizer
                    ),
                    clustered_training=clustered_training,
                )

            if clustered_training:
                return get_clustered_passage_dataset(
                    passage_dataset=dataset,
                    train_batch_size=self.args.train_batch_size,
                    encoder=self.context_encoder,
                    tokenizer=self.context_tokenizer,
                    args=self.args,
                    device=self.device,
                    teacher_model=self.teacher_model,
                    teacher_tokenizer=(
                        None
                        if self.args.teacher_type == "colbert"
                        else self.teacher_tokenizer
                    ),
                    clustered_batch_randomize_percentage=clustered_batch_randomize_percentage,
                    epoch_number=epoch_number,
                )
            elif self.args.tas_clustering:
                return get_tas_dataset(
                    passage_dataset=dataset,
                    train_batch_size=self.args.train_batch_size,
                    encoder=self.clustering_model,
                    tokenizer=self.clustering_tokenizer,
                    args=self.args,
                    device=self.device,
                )
            return dataset
        else:
            # Retrieval models can only be used with hf datasets
            raise ValueError("Retrieval models can only be used with hf datasets.")

    def get_optimizer_parameters(
        self, context_model, query_model, args, reranking_model=None
    ):
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
            if not args.tie_encoders:
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
            if not args.tie_encoders:
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
            if self.args.train_query_encoder and not self.args.tie_encoders:
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

        if self.args.multi_head_vectors:
            prediction_passages, id_to_vec_dataset = prediction_passages
            return prediction_passages, id_to_vec_dataset

        return prediction_passages

    def _calculate_loss(
        self,
        context_model,
        query_model,
        context_inputs,
        query_inputs,
        labels,
        teacher_scores=None,
    ):
        # if self.args.larger_representations:
        #     context_outputs_all = context_model(**context_inputs)
        #     query_outputs_all = query_model(**query_inputs)

        #     # context_outputs_all.sequence_output: (batch_size, context_length, hidden_size)
        #     # We have 3 CLS tokens in the beginning of each sequence.
        #     # The embeddings of these 3 CLS tokens are concatenated to be the representation of the sequence.
        #     context_cls_embeddings = context_outputs_all.sequence_output[:, :3, :]
        #     context_outputs = context_cls_embeddings.view(
        #         context_cls_embeddings.size(0), -1
        #     )

        #     query_cls_embeddings = query_outputs_all.sequence_output[:, :3, :]
        #     query_outputs = query_cls_embeddings.view(query_cls_embeddings.size(0), -1)
        # else:

        with (
            torch.no_grad()
            if not (self.args.train_context_encoder or self.args.train_query_encoder)
            else nullcontext()
        ):
            if self.args.external_embeddings:
                context_outputs = context_inputs["external_embeddings"]
            else:
                context_outputs = context_model(**context_inputs)
            query_outputs = query_model(**query_inputs)

            context_outputs = get_output_embeddings(
                context_outputs,
                concatenate_embeddings=self.args.larger_representations
                and self.args.model_type == "custom",
                n_cls_tokens=(1 + self.args.extra_cls_token_count),
                use_pooler_output=self.args.use_pooler_output,
                args=self.args,
                return_all_embeddings=self.args.use_autoencoder,
            )
            query_outputs = get_output_embeddings(
                query_outputs,
                concatenate_embeddings=self.args.larger_representations
                and self.args.model_type == "custom",
                n_cls_tokens=(1 + self.args.extra_cls_token_count),
                use_pooler_output=self.args.use_pooler_output,
                args=self.args,
                query_embeddings=True,
                return_all_embeddings=self.args.use_autoencoder,
            )
            if self.args.multi_vector_query:
                query_outputs = query_outputs[0]

            # context_outputs = torch.nn.functional.dropout(
            #     context_outputs, p=self.args.output_dropout
            # )
            # query_outputs = torch.nn.functional.dropout(
            #     query_outputs, p=self.args.output_dropout
            # )

        if self.args.include_triplet_loss:
            nll_criterion = torch.nn.NLLLoss(reduction="mean")
            triplet_criterion = torch.nn.TripletMarginLoss(
                margin=self.args.triplet_margin, reduction="mean"
            )
            positive_context_outputs = context_outputs[: query_outputs.size(0)]
            negative_context_outputs = context_outputs[query_outputs.size(0) :]
            nll_labels = labels

            if self.args.include_hard_negatives_for_triplets_only:
                similarity_score = torch.matmul(
                    query_outputs, positive_context_outputs.t()
                )
                softmax_score = torch.nn.functional.log_softmax(
                    similarity_score, dim=-1
                )
                nll_loss = nll_criterion(softmax_score, nll_labels)
            else:
                similarity_score = torch.matmul(query_outputs, context_outputs.t())
                softmax_score = torch.nn.functional.log_softmax(
                    similarity_score, dim=-1
                )
                nll_loss = nll_criterion(softmax_score, nll_labels)

            if self.context_encoder.training:
                triplet_loss = triplet_criterion(
                    query_outputs,
                    positive_context_outputs,
                    negative_context_outputs,
                )
                loss = (
                    self.args.nll_lambda * nll_loss
                    + self.args.triplet_lambda * triplet_loss
                )
            else:
                loss = nll_loss
        else:
            if self.args.skip_hard_negatives_for_nll:
                similarity_score = torch.matmul(
                    query_outputs,
                    context_outputs[: context_outputs.size(0) // 2, :].t(),
                )
            else:
                if self.args.similarity_function == "cosine":
                    similarity_score = torch.nn.functional.cosine_similarity(
                        query_outputs.unsqueeze(1),
                        context_outputs.unsqueeze(0),
                        dim=-1,
                    )
                elif self.args.similarity_function == "dot_product":
                    if self.args.multi_head_vectors:
                        # context_outputs: (batch_size, num_heads, hidden_size)
                        # query_outputs: (batch_size, num_heads, hidden_size)
                        # similarity_score: (batch_size, num_heads, num_heads)

                        similarity_score = compute_multi_head_similarity(
                            query_outputs,
                            context_outputs,
                            strategy=self.args.multi_head_vector_strategy,
                        )
                    else:
                        similarity_score = torch.matmul(
                            query_outputs,
                            context_outputs.t(),
                        )
            softmax_score = torch.nn.functional.log_softmax(similarity_score, dim=-1)

            if self.args.include_bce_loss and self.context_encoder.training:
                bce_criterion = torch.nn.BCEWithLogitsLoss()
                nll_criterion = torch.nn.NLLLoss(reduction="mean")
                bce_labels, nll_labels = labels

                bce_loss = bce_criterion(similarity_score, bce_labels)

                if self.args.include_nll_loss:
                    nll_loss = nll_criterion(softmax_score, nll_labels)
                    loss = bce_loss + nll_loss
                else:
                    loss = bce_loss
                    nll_loss = None
            else:
                if self.args.mse_loss:
                    raise ValueError(
                        "Use of MSE loss is not supported. Were you looking for margin_mse?"
                    )
                    with torch.no_grad():
                        label_scores = self._get_teacher_scores(
                            None, query_inputs, context_inputs
                        )
                (
                    loss,
                    nll_loss,
                    nll_labels,
                    label_scores,
                    margin_mse_loss,
                    kl_div_loss,
                    quartet_loss,
                ) = self._get_loss(
                    similarity_score,
                    softmax_score,
                    labels,
                    query_outputs=query_outputs,
                    context_outputs=context_outputs,
                    teacher_scores=teacher_scores,
                )

        (
            correct_predictions_count,
            correct_predictions_percentage,
            teacher_correct_predictions_percentage,
        ) = self._get_running_stats(
            softmax_score,
            nll_labels,
            label_scores,
        )

        retrieval_output = RetrievalOutput(
            loss=loss,
            context_outputs=context_outputs,
            query_outputs=query_outputs,
            correct_predictions_count=correct_predictions_count,
            correct_predictions_percentage=correct_predictions_percentage,
            nll_loss=nll_loss,
            teacher_correct_predictions_percentage=teacher_correct_predictions_percentage,
            margine_mse_loss=margin_mse_loss,
            kl_div_loss=kl_div_loss,
            quartet_loss=quartet_loss,
        )

        return retrieval_output

    # def _rerank_passages(
    #     self, query_outputs, context_outputs, is_evaluating=False, unified_cross_rr=True
    # ):
    #     """
    #     Unified cross reranking

    #     query_outputs: (batch_size, hidden_size)
    #     context_outputs: (batch_size, hidden_size)

    #     reranking_model_input_embeds: (batch_size, max_seq_length, hidden_size)
    #     Here, a single row of reranking_model_inputs is the concatenation of a query output (hidden_size) and all context outputs padded to max_seq_length.
    #     reranking_model_attention_mask: (batch_size, max_seq_length)
    #     reranking_model_token_type_ids: (batch_size, max_seq_length) - 0 for the query, 1 for the context
    #     """
    #     if unified_cross_rr:
    #         if is_evaluating:
    #             query_outputs = query_outputs.unsqueeze(1)
    #             reranking_model_inputs_embeds = torch.cat(
    #                 [query_outputs, context_outputs], dim=1
    #             )

    #             reranking_model_attention_mask = torch.ones_like(
    #                 reranking_model_inputs_embeds[:, :, 0]
    #             )

    #             reranking_model_token_type_ids = torch.zeros_like(
    #                 reranking_model_inputs_embeds[:, :, 0]
    #             )
    #             reranking_model_token_type_ids[:, 0] = torch.ones(
    #                 (query_outputs.size(0))
    #             )
    #         else:
    #             reranking_model_inputs_embeds = torch.zeros(
    #                 (
    #                     query_outputs.size(0),
    #                     self.args.max_seq_length,
    #                     query_outputs.size(1),
    #                 )
    #             )

    #             reranking_model_inputs_embeds[:, 0, :] = query_outputs
    #             reranking_model_inputs_embeds[
    #                 :, 1 : context_outputs.size(0) + 1, :
    #             ] = context_outputs

    #             reranking_model_attention_mask = torch.zeros(
    #                 (query_outputs.size(0), self.args.max_seq_length)
    #             )
    #             reranking_model_attention_mask[
    #                 :, 0 : context_outputs.size(0) + 1
    #             ] = torch.ones((query_outputs.size(0), context_outputs.size(0) + 1)).to(
    #                 self.device
    #             )

    #             reranking_model_token_type_ids = torch.zeros(
    #                 (query_outputs.size(0), self.args.max_seq_length)
    #             )
    #             reranking_model_token_type_ids[:, 0] = torch.ones(
    #                 (query_outputs.size(0))
    #             )

    #         reranking_model_inputs = {
    #             "inputs_embeds": reranking_model_inputs_embeds.to(self.device),
    #             "attention_mask": reranking_model_attention_mask.long().to(self.device),
    #             "token_type_ids": reranking_model_token_type_ids.long().to(self.device),
    #         }

    #         reranking_outputs = self.reranking_model(**reranking_model_inputs)

    #         if True:
    #             reranking_softmax_score = torch.nn.functional.log_softmax(
    #                 reranking_outputs, dim=-1
    #             )
    #             return reranking_outputs, reranking_softmax_score
    #         else:
    #             reranking_query_outputs = reranking_outputs[0][:, 0, :]
    #             reranking_context_outputs = reranking_outputs[0][
    #                 :, 1 : context_outputs.size(1 if is_evaluating else 0) + 1, :
    #             ]

    #             reranking_dot_score = torch.bmm(
    #                 reranking_query_outputs.unsqueeze(1),pairwise_dot_score(query_outputs, context_outputs)
    #                 reranking_context_outputs.transpose(-1, -2),
    #             )
    #             reranking_softmax_score = torch.nn.functional.log_softmax(
    #                 reranking_dot_score.squeeze(1), dim=-1
    #             )

    #             return reranking_dot_score, reranking_softmax_score
    #     else:
    #         # Autoencoder
    #         pass
    #         # reranking_dot_score = torch.matmul(
    #         #     query_outputs, context_outputs.t()
    #         # )
    #         # reranking_softmax_score = torch.nn.functional.log_softmax(
    #         #     reranking_dot_score, dim=-1
    #         # )

    #         # return reranking_dot_score, reranking_softmax_score

    def _get_teacher_scores(
        self, reranking_input=None, query_inputs=None, context_inputs=None
    ):
        """Get teacher scores for reranking_input

        Args:
            reranking_input (dict): Reranking input dict
        """
        if self.args.teacher_type == "colbert":
            label_scores = colbert_score(
                self.teacher_model,
                query_inputs,
                context_inputs,
                device=self.device,
            )

            return label_scores
        else:
            reranking_target_tensor = []
            for (
                reranking_input_ids,
                reranking_input_mask,
                reranking_token_type_ids,
            ) in zip(
                reranking_input["input_ids"],
                reranking_input["attention_mask"],
                reranking_input["token_type_ids"],
            ):
                reranking_target_tensor.extend(
                    self.teacher_model(
                        input_ids=reranking_input_ids,
                        attention_mask=reranking_input_mask,
                        token_type_ids=reranking_token_type_ids,
                    ).logits
                )

        # Stack and back to float32
        reranking_target_tensor = torch.stack(reranking_target_tensor).float()

        return reranking_target_tensor

    def _get_loss(
        self,
        similarity_score,
        softmax_score,
        labels,
        label_scores=None,
        query_outputs=None,
        context_outputs=None,
        teacher_scores=None,
    ):
        margin_mse_loss = None
        kl_div_loss = None
        nll_loss = None
        quartet_loss = None

        if self.args.mse_loss:
            mse_criterion = torch.nn.MSELoss()

            label_scores = label_scores.reshape(similarity_score.shape)
            mse_loss = mse_criterion(
                similarity_score,
                label_scores,
            )
        if self.args.include_nll_loss:
            criterion = torch.nn.NLLLoss(reduction="mean")
            nll_loss = criterion(softmax_score, labels)
            nll_labels = labels
        if self.args.include_margin_mse_loss:
            mmse_criterion = MarginMSELoss()
            half = context_outputs.size(0) // 2
            hn_outputs = context_outputs[half:, :]
            positive_outputs = context_outputs[:half, :]

            mmse_loss = mmse_criterion(
                query_outputs, positive_outputs, hn_outputs, teacher_scores["margins"]
            )
        if self.args.include_kl_div_loss:
            kl_criterion = KLDivLossForTriplets()
            half = context_outputs.size(0) // 2
            hn_outputs = context_outputs[half:, :]
            positive_outputs = context_outputs[:half, :]

            kl_div_loss = kl_criterion(
                query_outputs,
                positive_outputs,
                hn_outputs,
                teacher_scores["true_p_scores"],
                teacher_scores["true_n_scores"],
            )

        # This is where the quartet loss will go
        if self.args.include_quartet_loss:
            half = context_outputs.size(0) // 2
            context_outputs_a = context_outputs[:half, :]
            context_outputs_b = context_outputs[half:, :]
            query_outputs_a = query_outputs[:half, :]
            query_outputs_b = query_outputs[half:, :]

            quartet_criterion = QuartetLossV3()

            quartet_loss = quartet_criterion(
                query_outputs_a,
                query_outputs_b,
                context_outputs_a,
                context_outputs_b,
                similarity_score,
                softmax_score,
                teacher_scores,
                similarity_function=self.args.similarity_function,
            )

        if self.args.include_multi_negatives_loss:
            multi_negatives_criterion = MultiNegativesLoss()
            multi_negatives_loss = multi_negatives_criterion(
                similarity_score,
                labels,
            )

        if not (
            self.args.include_nll_loss
            or self.args.mse_loss
            or self.args.reranking_kl_div_loss
            or self.args.include_margin_mse_loss
            or self.args.include_quartet_loss
            or self.args.include_multi_negatives_loss
        ):
            raise ValueError(
                """One of the following must be True:
                - include_nll_loss
                - include_margin_mse_loss
                - include_kl_div_loss
                - include_quartet_loss
                - include_multi_negatives_loss
                """
            )

        nll_labels = labels

        if self.args.include_nll_loss:
            loss = nll_loss
            if self.args.include_margin_mse_loss or self.args.include_kl_div_loss:
                if (
                    self.args.nll_lambda_start_decay is not None
                    and self.global_step > self.args.nll_lambda_start_decay
                ):
                    nll_lambda = self.args.nll_lambda - (
                        self.nll_lambda_decay_per_step
                        * (self.global_step - self.args.nll_lambda_start_decay)
                    )
                else:
                    nll_lambda = self.args.nll_lambda

            if self.args.include_margin_mse_loss:
                loss = loss + self.args.margin_mse_lambda * mmse_loss
            if self.args.include_kl_div_loss:
                loss = loss + self.args.kl_div_lambda * kl_div_loss
            if self.args.include_quartet_loss:
                loss = loss + self.args.quartet_lambda * quartet_loss
        elif self.args.include_margin_mse_loss:
            loss = mmse_loss
        elif self.args.include_kl_div_loss:
            loss = kl_div_loss
        elif self.args.include_quartet_loss:
            loss = quartet_loss
        elif self.args.include_multi_negatives_loss:
            loss = multi_negatives_loss
        else:
            loss = nll_loss

        nll_loss = nll_loss.item() if self.args.include_nll_loss else None
        return (
            loss,
            nll_loss,
            nll_labels,
            label_scores,
            margin_mse_loss,
            kl_div_loss,
            quartet_loss,
        )

    def _get_running_stats(
        self,
        softmax_score,
        nll_labels,
        label_scores,
    ):
        max_score, max_idxs = torch.max(softmax_score, 1)
        correct_predictions_count = (
            (max_idxs == nll_labels.clone().detach()).sum().cpu().numpy().item()
        )
        correct_predictions_percentage = (
            correct_predictions_count / len(nll_labels)
        ) * 100

        if self.args.reranking_kl_div_loss or self.args.mse_loss:
            teacher_softmax_score = torch.nn.functional.softmax(label_scores, dim=-1)
            teacher_max_score, teacher_max_idxs = torch.max(teacher_softmax_score, 1)
            teacher_correct_predictions_count = (
                (teacher_max_idxs == nll_labels.clone().detach())
                .sum()
                .cpu()
                .numpy()
                .item()
            )
            teacher_correct_predictions_percentage = (
                teacher_correct_predictions_count / len(nll_labels)
            ) * 100
        else:
            teacher_correct_predictions_percentage = None

        return (
            correct_predictions_count,
            correct_predictions_percentage,
            teacher_correct_predictions_percentage,
        )

    def _get_inputs_dict(self, batch, evaluate=False):
        device = self.device

        if "labels" in batch:
            labels = batch["labels"].to(device)
        else:
            if self.args.include_quartet_loss or self.args.quartet_training_format:
                labels = [
                    i if a > b else i + batch["context_ids_a"].size(0)
                    for i, (a, b) in enumerate(
                        zip(batch["aa_score"], batch["ab_score"])
                    )
                ]
                labels += [
                    i + batch["context_ids_a"].size(0) if a > b else i
                    for i, (a, b) in enumerate(
                        zip(batch["bb_score"], batch["ba_score"])
                    )
                ]
            else:
                labels = [i for i in range(len(batch["context_ids"]))]
            labels = torch.tensor(labels, dtype=torch.long).to(device)
        margins = None
        true_p_scores = None
        true_n_scores = None
        aa_scores = None
        ab_scores = None
        ba_scores = None
        bb_scores = None
        teacher_scores = {}

        if not evaluate:
            # Training
            # labels = labels.to(device)
            if self.args.include_quartet_loss or self.args.quartet_training_format:
                context_ids = torch.cat(
                    [
                        batch["context_ids_a"],
                        batch["context_ids_b"],
                    ],
                    dim=0,
                )
                context_masks = torch.cat(
                    [
                        batch["context_mask_a"],
                        batch["context_mask_b"],
                    ],
                    dim=0,
                )

                query_ids = torch.cat(
                    [
                        batch["query_ids_a"],
                        batch["query_ids_b"],
                    ],
                    dim=0,
                )
                query_masks = torch.cat(
                    [
                        batch["query_mask_a"],
                        batch["query_mask_b"],
                    ],
                    dim=0,
                )
                if self.args.include_quartet_loss:
                    aa_scores = batch["aa_score"].to(device)
                    ab_scores = batch["ab_score"].to(device)
                    ba_scores = batch["ba_score"].to(device)
                    bb_scores = batch["bb_score"].to(device)

                teacher_scores["aa_scores"] = aa_scores
                teacher_scores["ab_scores"] = ab_scores
                teacher_scores["ba_scores"] = ba_scores
                teacher_scores["bb_scores"] = bb_scores
            else:
                if self.args.hard_negatives:
                    if self.args.n_hard_negatives == 1:
                        context_ids = torch.cat(
                            [
                                batch["context_ids"],
                                batch["hard_negative_ids"],
                            ],
                            dim=0,
                        )
                        context_masks = torch.cat(
                            [
                                batch["context_mask"],
                                batch["hard_negatives_mask"],
                            ],
                            dim=0,
                        )
                    else:
                        context_ids = torch.cat(
                            [
                                batch["context_ids"],
                            ]
                            + [
                                batch[f"hard_negative_{i}_ids"]
                                for i in range(self.args.n_hard_negatives)
                            ],
                            dim=0,
                        )

                        context_masks = torch.cat(
                            [
                                batch["context_mask"],
                            ]
                            + [
                                batch[f"hard_negative_{i}_mask"]
                                for i in range(self.args.n_hard_negatives)
                            ],
                            dim=0,
                        )
                    if self.args.include_margin_mse_loss:
                        margins = batch["margin"].to(device)

                    if self.args.include_kl_div_loss:
                        true_p_scores = batch["true_p_scores"].to(device)
                        true_n_scores = batch["true_n_scores"].to(device)
                    if (
                        self.args.include_quartet_loss
                        or self.args.quartet_training_format
                    ):
                        aa_scores = batch["aa_scores"].to(device)
                        ab_scores = batch["ab_scores"].to(device)
                        ba_scores = batch["ba_scores"].to(device)
                        bb_scores = batch["bb_scores"].to(device)

                    teacher_scores["true_p_scores"] = true_p_scores
                    teacher_scores["true_n_scores"] = true_n_scores
                    teacher_scores["margins"] = margins
                else:
                    context_ids = batch["context_ids"]
                    context_masks = batch["context_mask"]

                query_ids = batch["query_ids"]
                query_masks = batch["query_mask"]

            if self.args.external_embeddings:
                external_embeddings = batch["embeddings"].to(device)
                if self.args.hard_negatives:
                    hard_negative_embeddings = batch["hard_negative_embeddings"].to(
                        device
                    )
                    external_embeddings = torch.cat(
                        [external_embeddings, hard_negative_embeddings], dim=0
                    )
                context_input = {
                    "external_embeddings": external_embeddings,
                }

            context_input = {
                "input_ids": context_ids.to(device),
                "attention_mask": context_masks.to(device),
            }
            query_input = {
                "input_ids": query_ids.to(device),
                "attention_mask": query_masks.to(device),
            }
        else:
            # Evaluation
            shuffled_indices = torch.randperm(len(labels))

            labels = labels[shuffled_indices].to(device)

            if self.args.hard_negatives and self.args.hard_negatives_in_eval:
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

        if (
            isinstance(batch, HFDataset)
            and "labels" in batch.column_names
            and self.args.include_bce_loss
        ):
            labels = batch["labels"].to(device), labels  # BCELabels, NLLLabels

        return context_input, query_input, labels, teacher_scores

    def _create_training_progress_scores(
        self, calculate_recall=False, top_k_values=None, **kwargs
    ):
        # TODO: top_k_values should be part of the model. Probably.
        if top_k_values is None:
            top_k_values = [1, 2, 3, 5, 10]
        extra_metrics = {key: [] for key in kwargs}
        training_progress_scores = {
            "global_step": [],
            "eval_loss": [],
            "train_loss": [],
            **extra_metrics,
        }

        if self.args.evaluate_with_beir:
            beir_top_ks = [1, 3, 5, 10, 100, 1000]
            training_progress_scores = {
                **training_progress_scores,
                **{f"ndcg_at_{k}": [] for k in beir_top_ks},
                **{f"map_at_{k}": [] for k in beir_top_ks},
                **{f"recall_at_{k}": [] for k in beir_top_ks},
                **{f"precision_at_{k}": [] for k in beir_top_ks},
                **{f"mrr_at_{k}": [] for k in beir_top_ks},
            }
        elif self.args.data_format == "beir":
            if self.eval_dataset_names:
                for dataset_name in self.eval_dataset_names:
                    training_progress_scores = {
                        **training_progress_scores,
                        **{f"{dataset_name}_ndcg": []},
                        **{f"{dataset_name}_recip_rank": []},
                        **{f"{dataset_name}_recall_100": []},
                        **{f"{dataset_name}_ndcg_cut_10": []},
                    }

            else:
                training_progress_scores = {
                    **training_progress_scores,
                    **{"ndcg": []},
                    **{"recip_rank": []},
                    **{"recall_100": []},
                    **{"ndcg_cut_10": []},
                }
            # Remove eval_loss from training_progress_scores
            training_progress_scores.pop("eval_loss")
        else:
            training_progress_scores = {
                **training_progress_scores,
                **{f"mrr_at_{k}": [] for k in top_k_values},
            }
            training_progress_scores = {
                **training_progress_scores,
                **{f"top_{k}_accuracy": [] for k in top_k_values},
            }

            if calculate_recall:
                training_progress_scores = {
                    **training_progress_scores,
                    **{f"recall_at_{k}": [] for k in top_k_values},
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

    def _move_model_to_device(self, is_evaluating=False):
        self.context_encoder.to(self.device)
        self.query_encoder.to(self.device)

        if (
            self.args.mse_loss or self.args.reranking_kl_div_loss
        ) and not is_evaluating:
            self.teacher_model.to(self.device)

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

    def get_multi_head_predictions(
        self,
        doc_ids,
        scores,
        all_query_embeddings,
        id_to_vec_dataset,
        num_heads=12,
        head_dim=64,
    ):
        logger.info("Reranking documents using multi-head similarity scores.")
        all_query_embeddings = all_query_embeddings.reshape(-1, num_heads, head_dim)

        # Build id_to_vec_dict
        # all_doc_ids = set(itertools.chain(*doc_ids))
        # id_to_vec_dict = id_to_vec_dataset.filter(
        #     lambda x: x["passage_id"] in all_doc_ids
        # ).to_dict()
        # id_to_vec_dict = {
        #     pid: vector
        #     for pid, vector in zip(
        #         id_to_vec_dict["passage_id"], id_to_vec_dict["embeddings"]
        #     )
        # }

        ids_to_idx_map = {
            id: idx
            for idx, id in zip(
                range(len(id_to_vec_dataset)), id_to_vec_dataset["passage_id"]
            )
        }

        reranked_doc_ids = np.zeros(
            (len(all_query_embeddings), self.args.retrieve_n_docs), dtype=int
        ).tolist()
        reranked_scores = np.zeros(
            (len(all_query_embeddings), self.args.retrieve_n_docs)
        )

        with torch.no_grad():
            # Now we need to combine doc_ids into groups of num_heads and dedupe in the process
            query_i = 0
            for i in tqdm(
                range(0, len(doc_ids), num_heads), desc="Reranking documents"
            ):
                doc_group = doc_ids[i : i + num_heads]
                doc_group = list(set(itertools.chain(*doc_group)))

                query_outputs = (
                    all_query_embeddings[query_i].unsqueeze(0).to(self.device)
                )

                # Get doc head vectors
                doc_group_vectors = []
                for doc_id in doc_group:
                    # doc_group_vectors.append(torch.tensor(id_to_vec_dict[doc_id]))
                    doc_group_vectors.append(
                        torch.tensor(
                            id_to_vec_dataset[ids_to_idx_map[doc_id]]["embeddings"]
                        )
                    )

                doc_group_vectors = torch.stack(doc_group_vectors).to(self.device)

                similarity_scores = compute_multi_head_similarity(
                    query_outputs,
                    doc_group_vectors,
                    strategy=self.args.multi_head_vector_strategy,
                )
                ranked_values, ranked_ids = torch.topk(
                    similarity_scores, self.args.retrieve_n_docs, sorted=True
                )

                reranked_doc_ids[query_i] = [
                    doc_group[ranked_id]
                    for ranked_id in ranked_ids.squeeze().cpu().numpy()
                ]
                reranked_scores[query_i] = ranked_values.cpu().numpy()

                query_i += 1

        logger.info("Finished reranking documents.")
        return reranked_doc_ids, reranked_scores


def compute_multi_head_similarity(
    query_outputs,
    context_outputs,
    strategy="maxsim",
):
    """
    Compute multi-head similarity scores for each query vs. every document.
    Each query and document has multiple vectors (heads).
    The similarity score between a query  and a doc is computed as follows:
        1. For each query head, compute the similarity score with each document head for that document.
        2. Take the max sim for each query head.
        3. Sum the max sims for all query heads.

    Parameters:
    - query_outputs: Tensor of shape (num_queries, num_heads, hidden_dim), query embeddings.
    - context_outputs: Tensor of shape (num_docs, num_heads, hidden_dim), document embeddings.

    Returns:
    - similarity_scores: Tensor of shape (num_queries, num_docs), similarity scores for each query-doc pair.
    """
    num_queries, num_heads, hidden_dim = query_outputs.shape
    num_docs = context_outputs.shape[0]

    # Compute similarity scores for each query head with each document head.
    # similarity_scores: Tensor of shape (num_queries, num_docs, num_heads, num_heads)

    similarity_scores = torch.zeros(
        (num_queries, num_docs, num_heads, num_heads), device=query_outputs.device
    )

    for i in range(num_heads):
        for j in range(num_heads):
            similarity_scores[:, :, i, j] = torch.matmul(
                query_outputs[:, :, i].unsqueeze(1),
                context_outputs[:, :, j].transpose(-1, -2),
            ).squeeze(1)

    # Take the max similarity score for each query head.
    # max_similarities: Tensor of shape (num_queries, num_docs, num_heads)
    max_similarities = torch.max(similarity_scores, dim=3).values

    if strategy == "maxsim":
        # Sum the max similarities for all query heads.
        # similarity_scores: Tensor of shape (num_queries, num_docs)
        similarity_scores = torch.sum(max_similarities, dim=2)
    elif strategy == "minmax":
        min_similarities = torch.min(similarity_scores, dim=3).values
        similarity_scores = torch.sum(max_similarities, dim=2) + torch.sum(
            min_similarities, dim=2
        )
    elif strategy == "max":
        # Take the max similarity of all query heads.
        # similarity_scores: Tensor of shape (num_queries, num_docs)
        similarity_scores = torch.max(max_similarities, dim=2).values
    elif strategy == "mean":
        # Take the mean similarity of all query heads versus all doc heads.
        # similarity_scores: Tensor of shape (num_queries, num_docs)
        pass

    return similarity_scores
