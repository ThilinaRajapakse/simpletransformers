import logging
import os
import random
import warnings
from multiprocessing import cpu_count

import numpy as np
import torch
from transformers import (
    WEIGHTS_NAME,
    AlbertConfig,
    AlbertTokenizer,
    BertConfig,
    BertTokenizer,
    BertweetTokenizer,
    BigBirdConfig,
    BigBirdTokenizer,
    CamembertConfig,
    CamembertTokenizer,
    DistilBertConfig,
    DistilBertTokenizer,
    ElectraConfig,
    ElectraTokenizer,
    HerbertTokenizer,
    FlaubertConfig,
    FlaubertTokenizer,
    LayoutLMConfig,
    LayoutLMTokenizerFast,
    LongformerConfig,
    LongformerTokenizer,
    NystromformerConfig,
    # NystromformerTokenizer,
    NystromformerForSequenceClassification,
    RemBertConfig,
    RemBertForSequenceClassification,
    RemBertTokenizer,
    RobertaConfig,
    RobertaTokenizer,
    XLMConfig,
    XLMRobertaConfig,
    XLMRobertaTokenizer,
    XLMTokenizer,
    XLNetConfig,
    XLNetTokenizer,
)

from simpletransformers.classification import ClassificationModel
from simpletransformers.config.global_args import global_args
from simpletransformers.config.model_args import MultiLabelClassificationArgs
from simpletransformers.config.utils import sweep_config_to_sweep_values
from simpletransformers.custom_models.models import (
    AlbertForMultiLabelSequenceClassification,
    BertForMultiLabelSequenceClassification,
    BertweetForMultiLabelSequenceClassification,
    BigBirdForMultiLabelSequenceClassification,
    CamembertForMultiLabelSequenceClassification,
    DistilBertForMultiLabelSequenceClassification,
    ElectraForMultiLabelSequenceClassification,
    FlaubertForMultiLabelSequenceClassification,
    LayoutLMForMultiLabelSequenceClassification,
    LongformerForMultiLabelSequenceClassification,
    NystromformerForMultiLabelSequenceClassification,
    RemBertForMultiLabelSequenceClassification,
    RobertaForMultiLabelSequenceClassification,
    XLMForMultiLabelSequenceClassification,
    XLMRobertaForMultiLabelSequenceClassification,
    XLNetForMultiLabelSequenceClassification,
)

try:
    import wandb

    wandb_available = True
except ImportError:
    wandb_available = False

logger = logging.getLogger(__name__)


class MultiLabelClassificationModel(ClassificationModel):
    def __init__(
        self,
        model_type,
        model_name,
        num_labels=None,
        pos_weight=None,
        args=None,
        use_cuda=True,
        cuda_device=-1,
        global_attention_fn=None,
        **kwargs,
    ):
        """
        Initializes a MultiLabelClassification model.

        Args:
            model_type: The type of model (bert, roberta)
            model_name: Default Transformer model name or path to a directory containing Transformer model file (pytorch_nodel.bin).
            num_labels (optional): The number of labels or classes in the dataset.
            pos_weight (optional): A list of length num_labels containing the weights to assign to each label for loss calculation.
            args (optional): Default args will be used if this parameter is not provided. If provided, it should be a dict containing the args that should be changed in the default args.
            use_cuda (optional): Use GPU if available. Setting to False will force model to use CPU only.
            cuda_device (optional): Specific GPU that should be used. Will use the first available GPU by default.
            **kwargs (optional): For providing proxies, force_download, resume_download, cache_dir and other options specific to the 'from_pretrained' implementation where this will be supplied.
        """  # noqa: ignore flake8"

        MODEL_CLASSES = {
            "albert": (
                AlbertConfig,
                AlbertForMultiLabelSequenceClassification,
                AlbertTokenizer,
            ),
            "bert": (
                BertConfig,
                BertForMultiLabelSequenceClassification,
                BertTokenizer,
            ),
            "bertweet": (
                RobertaConfig,
                BertweetForMultiLabelSequenceClassification,
                BertweetTokenizer,
            ),
            "bigbird": (
                BigBirdConfig,
                BigBirdForMultiLabelSequenceClassification,
                BigBirdTokenizer,
            ),
            "camembert": (
                CamembertConfig,
                CamembertForMultiLabelSequenceClassification,
                CamembertTokenizer,
            ),
            "distilbert": (
                DistilBertConfig,
                DistilBertForMultiLabelSequenceClassification,
                DistilBertTokenizer,
            ),
            "electra": (
                ElectraConfig,
                ElectraForMultiLabelSequenceClassification,
                ElectraTokenizer,
            ),
            "herbert": (
                BertConfig,
                BertForMultiLabelSequenceClassification,
                HerbertTokenizer,
            ),
            "flaubert": (
                FlaubertConfig,
                FlaubertForMultiLabelSequenceClassification,
                FlaubertTokenizer,
            ),
            "layoutlm": (
                LayoutLMConfig,
                LayoutLMForMultiLabelSequenceClassification,
                LayoutLMTokenizerFast,
            ),
            "longformer": (
                LongformerConfig,
                LongformerForMultiLabelSequenceClassification,
                LongformerTokenizer,
            ),
            "nystromformer": (
                NystromformerConfig,
                NystromformerForMultiLabelSequenceClassification,
                BigBirdTokenizer,
            ),
            "rembert": (
                RemBertConfig,
                RemBertForMultiLabelSequenceClassification,
                RemBertTokenizer,
            ),
            "roberta": (
                RobertaConfig,
                RobertaForMultiLabelSequenceClassification,
                RobertaTokenizer,
            ),
            "xlm": (XLMConfig, XLMForMultiLabelSequenceClassification, XLMTokenizer),
            "xlmroberta": (
                XLMRobertaConfig,
                XLMRobertaForMultiLabelSequenceClassification,
                XLMRobertaTokenizer,
            ),
            "xlnet": (
                XLNetConfig,
                XLNetForMultiLabelSequenceClassification,
                XLNetTokenizer,
            ),
        }

        self.args = self._load_model_args(model_name)

        if isinstance(args, dict):
            self.args.update_from_dict(args)
        elif isinstance(args, MultiLabelClassificationArgs):
            self.args = args

        if self.args.thread_count:
            torch.set_num_threads(self.args.thread_count)

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

        config_class, model_class, tokenizer_class = MODEL_CLASSES[model_type]
        if num_labels:
            self.config = config_class.from_pretrained(
                model_name, num_labels=num_labels, **self.args.config
            )
            self.num_labels = num_labels
        else:
            self.config = config_class.from_pretrained(model_name, **self.args.config)
            self.num_labels = self.config.num_labels
        self.pos_weight = pos_weight
        self.loss_fct = None

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

        if not self.args.quantized_model:
            if self.pos_weight:
                self.model = model_class.from_pretrained(
                    model_name,
                    config=self.config,
                    pos_weight=torch.Tensor(self.pos_weight).to(self.device),
                    **kwargs,
                )
            else:
                self.model = model_class.from_pretrained(
                    model_name, config=self.config, **kwargs
                )
        else:
            quantized_weights = torch.load(
                os.path.join(model_name, "pytorch_model.bin")
            )
            if self.pos_weight:
                self.model = model_class.from_pretrained(
                    None,
                    config=self.config,
                    state_dict=quantized_weights,
                    weight=torch.Tensor(self.pos_weight).to(self.device),
                )
            else:
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

        if self.args.wandb_project and not wandb_available:
            warnings.warn(
                "wandb_project specified but wandb is not available. Wandb disabled."
            )
            self.args.wandb_project = None

        if global_attention_fn:
            self.global_attention_fn = global_attention_fn
        else:
            self.global_attention_fn = None

        self.weight = None  # Not implemented for multilabel

    def _load_model_args(self, input_dir):
        args = MultiLabelClassificationArgs()
        args.load(input_dir)
        return args

    def train_model(
        self,
        train_df,
        multi_label=True,
        eval_df=None,
        output_dir=None,
        show_running_loss=True,
        args=None,
        verbose=True,
        **kwargs,
    ):
        return super().train_model(
            train_df,
            multi_label=multi_label,
            eval_data=eval_df,
            output_dir=output_dir,
            show_running_loss=show_running_loss,
            verbose=True,
            args=args,
            **kwargs,
        )

    def eval_model(
        self,
        eval_df,
        multi_label=True,
        output_dir=None,
        verbose=False,
        silent=False,
        **kwargs,
    ):
        return super().eval_model(
            eval_df,
            output_dir=output_dir,
            multi_label=multi_label,
            verbose=verbose,
            silent=silent,
            **kwargs,
        )

    def evaluate(
        self,
        eval_df,
        output_dir,
        multi_label=True,
        prefix="",
        verbose=True,
        silent=False,
        **kwargs,
    ):
        return super().evaluate(
            eval_df,
            output_dir,
            multi_label=multi_label,
            prefix=prefix,
            verbose=verbose,
            silent=silent,
            **kwargs,
        )

    def load_and_cache_examples(
        self,
        examples,
        evaluate=False,
        no_cache=False,
        multi_label=True,
        verbose=True,
        silent=False,
    ):
        return super().load_and_cache_examples(
            examples,
            evaluate=evaluate,
            no_cache=no_cache,
            multi_label=multi_label,
            verbose=verbose,
            silent=silent,
        )

    def compute_metrics(
        self, preds, model_outputs, labels, eval_examples, multi_label=True, **kwargs
    ):
        return super().compute_metrics(
            preds,
            model_outputs,
            labels,
            eval_examples,
            multi_label=multi_label,
            **kwargs,
        )

    def predict(self, to_predict, multi_label=True):
        return super().predict(to_predict, multi_label=multi_label)
