import torch

from multiprocessing import cpu_count
import warnings

from simpletransformers.classification import ClassificationModel
from simpletransformers.custom_models.models import (
    BertForMultiLabelSequenceClassification,
    RobertaForMultiLabelSequenceClassification,
    XLNetForMultiLabelSequenceClassification,
    XLMForMultiLabelSequenceClassification,
    DistilBertForMultiLabelSequenceClassification,
    AlbertForMultiLabelSequenceClassification,
    FlaubertForMultiLabelSequenceClassification,
)
from simpletransformers.config.global_args import global_args

from transformers import (
    WEIGHTS_NAME,
    BertConfig,
    BertTokenizer,
    XLNetConfig,
    XLNetTokenizer,
    XLMConfig,
    XLMTokenizer,
    RobertaConfig,
    RobertaTokenizer,
    DistilBertConfig,
    DistilBertTokenizer,
    AlbertConfig,
    AlbertTokenizer,
    FlaubertConfig,
    FlaubertTokenizer,
)

try:
    import wandb
    wandb_available = True
except ImportError:
    wandb_available = False


class MultiLabelClassificationModel(ClassificationModel):
    def __init__(self, model_type, model_name, num_labels=None, pos_weight=None, args=None, use_cuda=True, **kwargs):

        """
        Initializes a MultiLabelClassification model.

        Args:
            model_type: The type of model (bert, roberta)
            model_name: Default Transformer model name or path to a directory containing Transformer model file (pytorch_nodel.bin).
            num_labels (optional): The number of labels or classes in the dataset.
            pos_weight (optional): A list of length num_labels containing the weights to assign to each label for loss calculation.
            args (optional): Default args will be used if this parameter is not provided. If provided, it should be a dict containing the args that should be changed in the default args.
            use_cuda (optional): Use GPU if available. Setting to False will force model to use CPU only.
            **kwargs (optional): For providing proxies, force_download, resume_download, cache_dir and other options specific to the 'from_pretrained' implementation where this will be supplied.
        """  # noqa: ignore flake8"

        MODEL_CLASSES = {
            "bert": (BertConfig, BertForMultiLabelSequenceClassification, BertTokenizer,),
            "roberta": (RobertaConfig, RobertaForMultiLabelSequenceClassification, RobertaTokenizer,),
            "xlnet": (XLNetConfig, XLNetForMultiLabelSequenceClassification, XLNetTokenizer,),
            "xlm": (XLMConfig, XLMForMultiLabelSequenceClassification, XLMTokenizer),
            "distilbert": (DistilBertConfig, DistilBertForMultiLabelSequenceClassification, DistilBertTokenizer,),
            "albert": (AlbertConfig, AlbertForMultiLabelSequenceClassification, AlbertTokenizer,),
            "flaubert": (FlaubertConfig, FlaubertForMultiLabelSequenceClassification, FlaubertTokenizer,),
        }

        config_class, model_class, tokenizer_class = MODEL_CLASSES[model_type]
        if num_labels:

            self.config = config_class.from_pretrained(model_name, num_labels=num_labels, **kwargs)

            self.num_labels = num_labels
        else:
            self.config = config_class.from_pretrained(model_name, **kwargs)
            self.num_labels = self.config.num_labels
        self.pos_weight = pos_weight

        if use_cuda:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                raise ValueError(
                    "'use_cuda' set to True when cuda is unavailable."
                    " Make sure CUDA is available or set use_cuda=False."
                )
        else:
            self.device = "cpu"

        if self.pos_weight:
            self.model = model_class.from_pretrained(
                model_name, config=self.config, pos_weight=torch.Tensor(self.pos_weight).to(self.device), **kwargs
            )
        else:
            self.model = model_class.from_pretrained(model_name, config=self.config, **kwargs)

        self.results = {}

        self.args = {
            "threshold": 0.5,
            "sliding_window": False,
            "tie_value": 1,
            "stride": False,
        }

        self.args.update(global_args)

        if not use_cuda:
            self.args["fp16"] = False

        if args:
            self.args.update(args)

        self.tokenizer = tokenizer_class.from_pretrained(model_name, do_lower_case=self.args["do_lower_case"], **kwargs)

        self.args["model_name"] = model_name
        self.args["model_type"] = model_type

        if self.args["wandb_project"] and not wandb_available:
            warnings.warn("wandb_project specified but wandb is not available. Wandb disabled.")
            self.args["wandb_project"] = None

    def train_model(
        self,
        train_df,
        multi_label=True,
        eval_df=None,
        output_dir=None,
        show_running_loss=True,
        args=None,
        verbose=True,
        **kwargs
    ):
        return super().train_model(
            train_df,
            multi_label=multi_label,
            eval_df=eval_df,
            output_dir=output_dir,
            show_running_loss=show_running_loss,
            verbose=True,
            args=args,
        )

    def eval_model(self, eval_df, multi_label=True, output_dir=None, verbose=False, silent=False, **kwargs):
        return super().eval_model(
            eval_df, output_dir=output_dir, multi_label=multi_label, verbose=verbose, silent=silent, **kwargs
        )

    def evaluate(self, eval_df, output_dir, multi_label=True, prefix="", verbose=True, silent=False, **kwargs):
        return super().evaluate(
            eval_df, output_dir, multi_label=multi_label, prefix=prefix, verbose=verbose, silent=silent, **kwargs
        )

    def load_and_cache_examples(
        self, examples, evaluate=False, no_cache=False, multi_label=True, verbose=True, silent=False
    ):
        return super().load_and_cache_examples(
            examples, evaluate=evaluate, no_cache=no_cache, multi_label=multi_label, verbose=verbose, silent=silent
        )

    def compute_metrics(self, preds, labels, eval_examples, multi_label=True, **kwargs):
        return super().compute_metrics(preds, labels, eval_examples, multi_label=multi_label, **kwargs)

    def predict(self, to_predict, multi_label=True):
        return super().predict(to_predict, multi_label=multi_label)
