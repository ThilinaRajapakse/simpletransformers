#!/usr/bin/env python
# coding: utf-8


from __future__ import absolute_import, division, print_function

import json
import math
import os
import random
import warnings
from multiprocessing import cpu_count

import numpy as np
import torch
from scipy.stats import pearsonr
from sklearn.metrics import (
    confusion_matrix,
    label_ranking_average_precision_score,
    matthews_corrcoef,
    mean_squared_error,
)
from tensorboardX import SummaryWriter
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from tqdm.auto import tqdm, trange
from transformers import (
    WEIGHTS_NAME,
    AdamW,
    AlbertConfig,
    AlbertTokenizer,
    BertConfig,
    BertTokenizer,
    CamembertConfig,
    CamembertTokenizer,
    DistilBertConfig,
    DistilBertTokenizer,
    RobertaConfig,
    RobertaTokenizer,
    XLMConfig,
    XLMTokenizer,
    XLNetConfig,
    XLNetTokenizer,
    get_linear_schedule_with_warmup,
)

from simpletransformers.experimental.classification.classification_utils import (
    InputExample,
    convert_examples_to_features,
)
from simpletransformers.experimental.classification.transformer_models.albert_model import (
    AlbertForSequenceClassification,
)
from simpletransformers.experimental.classification.transformer_models.bert_model import (
    BertForSequenceClassification,
)
from simpletransformers.experimental.classification.transformer_models.camembert_model import (
    CamembertForSequenceClassification,
)
from simpletransformers.experimental.classification.transformer_models.distilbert_model import (
    DistilBertForSequenceClassification,
)
from simpletransformers.experimental.classification.transformer_models.roberta_model import (
    RobertaForSequenceClassification,
)
from simpletransformers.experimental.classification.transformer_models.xlm_model import (
    XLMForSequenceClassification,
)
from simpletransformers.experimental.classification.transformer_models.xlnet_model import (
    XLNetForSequenceClassification,
)


class ClassificationModel:
    def __init__(
        self,
        model_type,
        model_name,
        num_labels=None,
        weight=None,
        sliding_window=False,
        args=None,
        use_cuda=True,
    ):
        """
        Initializes a ClassificationModel model.

        Args:
            model_type: The type of model (bert, xlnet, xlm, roberta, distilbert, albert, camembert)
            model_name: Default Transformer model name or path to a directory containing Transformer model file (pytorch_nodel.bin).
            num_labels (optional): The number of labels or classes in the dataset.
            weight (optional): A list of length num_labels containing the weights to assign to each label for loss calculation.
            sliding_window (optional): Use a sliding window when tokenizing to prevent truncating long sequences. Default = False.
            args (optional): Default args will be used if this parameter is not provided. If provided, it should be a dict containing the args that should be changed in the default args.
            use_cuda (optional): Use GPU if available. Setting to False will force model to use CPU only.
        """

        MODEL_CLASSES = {
            "bert": (BertConfig, BertForSequenceClassification, BertTokenizer),
            "xlnet": (XLNetConfig, XLNetForSequenceClassification, XLNetTokenizer),
            "xlm": (XLMConfig, XLMForSequenceClassification, XLMTokenizer),
            "roberta": (
                RobertaConfig,
                RobertaForSequenceClassification,
                RobertaTokenizer,
            ),
            "distilbert": (
                DistilBertConfig,
                DistilBertForSequenceClassification,
                DistilBertTokenizer,
            ),
            "albert": (AlbertConfig, AlbertForSequenceClassification, AlbertTokenizer),
            "camembert": (
                CamembertConfig,
                CamembertForSequenceClassification,
                CamembertTokenizer,
            ),
        }

        config_class, model_class, tokenizer_class = MODEL_CLASSES[model_type]
        if num_labels:
            self.config = config_class.from_pretrained(
                model_name, num_labels=num_labels
            )
            self.num_labels = num_labels
        else:
            self.config = config_class.from_pretrained(model_name)
            self.num_labels = self.config.num_labels
        self.tokenizer = tokenizer_class.from_pretrained(model_name)
        self.num_labels = num_labels
        self.weight = weight
        self.sliding_window = sliding_window

        if use_cuda:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                raise ValueError(
                    "'use_cuda' set to True when cuda is unavailable. Make sure CUDA is available or set use_cuda=False."
                )
        else:
            self.device = "cpu"

        if self.weight:
            self.model = model_class.from_pretrained(
                model_name,
                config=self.config,
                weight=torch.Tensor(self.weight).to(self.device),
                sliding_window=self.sliding_window,
            )
        else:
            self.model = model_class.from_pretrained(
                model_name, config=self.config, sliding_window=self.sliding_window
            )

        self.results = {}

        self.args = {
            "output_dir": "outputs/",
            "cache_dir": "cache_dir/",
            "fp16": True,
            "max_seq_length": 128,
            "train_batch_size": 8,
            "gradient_accumulation_steps": 1,
            "eval_batch_size": 8,
            "num_train_epochs": 1,
            "weight_decay": 0,
            "learning_rate": 4e-5,
            "adam_epsilon": 1e-8,
            "warmup_ratio": 0.06,
            "warmup_steps": 0,
            "max_grad_norm": 1.0,
            "stride": False,
            "logging_steps": 50,
            "save_steps": 2000,
            "evaluate_during_training": False,
            "tensorboard_dir": None,
            "overwrite_output_dir": False,
            "reprocess_input_data": False,
            "process_count": cpu_count() - 2 if cpu_count() > 2 else 1,
            "n_gpu": 1,
            "use_multiprocessing": True,
            "silent": False,
        }

        if not use_cuda:
            self.args["fp16"] = False

        if args:
            self.args.update(args)

        self.args["model_name"] = model_name
        self.args["model_type"] = model_type

        if model_type == "camembert":
            warnings.warn(
                "use_multiprocessing automatically disabled as CamemBERT fails when using multiprocessing for feature conversion."
            )
            self.args["use_multiprocessing"] = False

        if self.args["stride"] and not sliding_window:
            warnings.warn(
                "Stride argument specified but sliding_window is disabled. Stride will be ignored."
            )

    def train_model(
        self,
        train_df,
        multi_label=False,
        output_dir=None,
        show_running_loss=True,
        args=None,
        eval_df=None,
    ):
        """
        Trains the model using 'train_df'

        Args:
            train_df: Pandas Dataframe containing at least two columns. If the Dataframe has a header, it should contain a 'text' and a 'labels' column. If no header is present,
            the Dataframe should contain at least two columns, with the first column containing the text, and the second column containing the label. The model will be trained on this Dataframe.
            output_dir: The directory where model files will be saved. If not given, self.args['output_dir'] will be used.
            show_running_loss (optional): Set to False to prevent running loss from being printed to console. Defaults to True.
            args (optional): Optional changes to the args dict of the model. Any changes made will persist for the model.
            eval_df (optional): A DataFrame against which evaluation will be performed when evaluate_during_training is enabled. Is required if evaluate_during_training is enabled.

        Returns:
            None
        """

        if args:
            self.args.update(args)

        if self.args["silent"]:
            show_running_loss = False

        if self.args["evaluate_during_training"] and eval_df is None:
            raise ValueError(
                "evaluate_during_training is enabled but eval_df is not specified. Pass eval_df to model.train_model() if using evaluate_during_training."
            )

        if not output_dir:
            output_dir = self.args["output_dir"]

        if (
            os.path.exists(output_dir)
            and os.listdir(output_dir)
            and not self.args["overwrite_output_dir"]
        ):
            raise ValueError(
                "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                    output_dir
                )
            )

        self._move_model_to_device()

        if "text" in train_df.columns and "labels" in train_df.columns:
            train_examples = [
                InputExample(i, text, None, label)
                for i, (text, label) in enumerate(
                    zip(train_df["text"], train_df["labels"])
                )
            ]
        else:
            train_examples = [
                InputExample(i, text, None, label)
                for i, (text, label) in enumerate(
                    zip(train_df.iloc[:, 0], train_df.iloc[:, 1])
                )
            ]

        train_dataset = self.load_and_cache_examples(train_examples)
        global_step, training_details = self.train(
            train_dataset,
            output_dir,
            show_running_loss=show_running_loss,
            eval_df=eval_df,
        )

        os.makedirs(output_dir, exist_ok=True)

        model_to_save = (
            self.model.module if hasattr(self.model, "module") else self.model
        )
        model_to_save.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        torch.save(self.args, os.path.join(output_dir, "training_args.bin"))

        print(
            "Training of {} model complete. Saved to {}.".format(
                self.args["model_type"], output_dir
            )
        )

    def train(self, train_dataset, output_dir, show_running_loss=True, eval_df=None):
        """
        Trains the model on train_dataset.

        Utility function to be used by the train_model() method. Not intended to be used directly.
        """

        tokenizer = self.tokenizer
        device = self.device
        model = self.model
        args = self.args

        tb_writer = SummaryWriter(logdir=args["tensorboard_dir"])
        train_sampler = RandomSampler(train_dataset)
        train_dataloader = DataLoader(
            train_dataset, sampler=train_sampler, batch_size=args["train_batch_size"]
        )

        t_total = (
            len(train_dataloader)
            // args["gradient_accumulation_steps"]
            * args["num_train_epochs"]
        )

        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": args["weight_decay"],
            },
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]

        warmup_steps = math.ceil(t_total * args["warmup_ratio"])
        args["warmup_steps"] = (
            warmup_steps if args["warmup_steps"] == 0 else args["warmup_steps"]
        )

        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=args["learning_rate"],
            eps=args["adam_epsilon"],
        )
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=args["warmup_steps"], num_training_steps=t_total
        )

        if args["n_gpu"] > 1:
            model = torch.nn.DataParallel(model)

        global_step = 0
        tr_loss, logging_loss = 0.0, 0.0
        model.zero_grad()
        train_iterator = trange(
            int(args["num_train_epochs"]), desc="Epoch", disable=args["silent"]
        )

        if args["fp16"]:
            from torch.cuda import amp

            scaler = amp.GradScaler()

        for _ in train_iterator:
            model.train()
            # epoch_iterator = tqdm(train_dataloader, desc="Iteration")
            for step, batch in enumerate(
                tqdm(train_dataloader, desc=f"Running Training", disable=args["silent"])
            ):
                batch = tuple(t.to(self.device) for t in batch)

                inputs = self._get_inputs_dict(batch)
                if args["fp16"]:
                    with amp.autocast():
                        if self.sliding_window:
                            outputs = model(inputs)
                        else:
                            outputs = model(**inputs)
                        # model outputs are always tuple in pytorch-transformers (see doc)
                else:
                    if self.sliding_window:
                        outputs = model(inputs)
                    else:
                        outputs = model(**inputs)
                    # model outputs are always tuple in pytorch-transformers (see doc)
                    loss = outputs[0]
                if show_running_loss:
                    print("\rRunning loss: %f" % loss, end="")

                if args["n_gpu"] > 1:
                    loss = (
                        loss.mean()
                    )  # mean() to average on multi-gpu parallel training
                if args["gradient_accumulation_steps"] > 1:
                    loss = loss / args["gradient_accumulation_steps"]

                if args["fp16"]:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

                tr_loss += loss.item()
                if (step + 1) % args["gradient_accumulation_steps"] == 0:
                    if args["fp16"]:
                        scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), args["max_grad_norm"]
                    )

                    if args["fp16"]:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    scheduler.step()  # Update learning rate schedule
                    model.zero_grad()
                    global_step += 1

                    if (
                        args["logging_steps"] > 0
                        and global_step % args["logging_steps"] == 0
                    ):
                        # Log metrics
                        if args["evaluate_during_training"]:
                            # Only evaluate when single GPU otherwise metrics may not average well
                            results, _, _ = self.eval_model(eval_df, verbose=True)
                            for key, value in results.items():
                                tb_writer.add_scalar(
                                    "eval_{}".format(key), value, global_step
                                )
                        tb_writer.add_scalar(
                            "lr", scheduler.get_last_lr()[0], global_step
                        )
                        tb_writer.add_scalar(
                            "loss",
                            (tr_loss - logging_loss) / args["logging_steps"],
                            global_step,
                        )
                        logging_loss = tr_loss

                    if args["save_steps"] > 0 and global_step % args["save_steps"] == 0:
                        # Save model checkpoint
                        output_dir_current = os.path.join(
                            output_dir, "checkpoint-{}".format(global_step)
                        )

                        os.makedirs(output_dir_current, exist_ok=True)

                        # Take care of distributed/parallel training
                        model_to_save = (
                            model.module if hasattr(model, "module") else model
                        )
                        model_to_save.save_pretrained(output_dir_current)
                        self.tokenizer.save_pretrained(output_dir_current)

        return (
            global_step,
            tr_loss / global_step
            if not self.args.evaluate_during_training
            else training_progress_scores,
        )

    def eval_model(
        self, eval_df, multi_label=False, output_dir=None, verbose=False, **kwargs
    ):
        """
        Evaluates the model on eval_df. Saves results to output_dir.

        Args:
            eval_df: Pandas Dataframe containing at least two columns. If the Dataframe has a header, it should contain a 'text' and a 'labels' column. If no header is present,
            the Dataframe should contain at least two columns, with the first column containing the text, and the second column containing the label. The model will be evaluated on this Dataframe.
            output_dir: The directory where model files will be saved. If not given, self.args['output_dir'] will be used.
            verbose: If verbose, results will be printed to the console on completion of evaluation.
            **kwargs: Additional metrics that should be used. Pass in the metrics as keyword arguments (name of metric: function to use). E.g. f1=sklearn.metrics.f1_score.
                        A metric function should take in two parameters. The first parameter will be the true labels, and the second parameter will be the predictions.

        Returns:
            result: Dictionary containing evaluation results. (Matthews correlation coefficient, tp, tn, fp, fn)
            model_outputs: List of model outputs for each row in eval_df
            wrong_preds: List of InputExample objects corresponding to each incorrect prediction by the model
        """

        if not output_dir:
            output_dir = self.args["output_dir"]

        self._move_model_to_device()

        result, model_outputs, wrong_preds = self.evaluate(
            eval_df, output_dir, multi_label=multi_label, **kwargs
        )
        self.results.update(result)

        if verbose:
            print(self.results)

        return result, model_outputs, wrong_preds

    def evaluate(self, eval_df, output_dir, multi_label=False, prefix="", **kwargs):
        """
        Evaluates the model on eval_df.

        Utility function to be used by the eval_model() method. Not intended to be used directly.
        """

        tokenizer = self.tokenizer
        device = self.device
        model = self.model
        args = self.args
        eval_output_dir = output_dir

        results = {}

        if "text" in eval_df.columns and "labels" in eval_df.columns:
            eval_examples = [
                InputExample(i, text, None, label)
                for i, (text, label) in enumerate(
                    zip(eval_df["text"], eval_df["labels"])
                )
            ]
        else:
            eval_examples = [
                InputExample(i, text, None, label)
                for i, (text, label) in enumerate(
                    zip(eval_df.iloc[:, 0], eval_df.iloc[:, 1])
                )
            ]

        eval_dataset = self.load_and_cache_examples(eval_examples, evaluate=True)
        os.makedirs(eval_output_dir, exist_ok=True)

        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(
            eval_dataset, sampler=eval_sampler, batch_size=args["eval_batch_size"]
        )

        eval_loss = 0.0
        nb_eval_steps = 0
        preds = None
        out_label_ids = None
        model.eval()

        for batch in tqdm(eval_dataloader, disable=args["silent"]):
            batch = tuple(t.to(device) for t in batch)

            with torch.no_grad():
                inputs = self._get_inputs_dict(batch)

                if self.sliding_window:
                    outputs = model(inputs)
                else:
                    outputs = model(**inputs)

                tmp_eval_loss, logits = outputs[:2]

                if multi_label:
                    logits = logits.sigmoid()
                if self.args.n_gpu > 1:
                    tmp_eval_loss = tmp_eval_loss.mean()
                eval_loss += tmp_eval_loss.item()

            nb_eval_steps += 1

            if self.sliding_window:
                if preds is None:
                    preds = logits.detach().cpu().numpy()
                    out_label_ids = inputs[0]["labels"].detach().cpu().numpy()
                else:
                    preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                    out_label_ids = np.append(
                        out_label_ids,
                        inputs[0]["labels"].detach().cpu().numpy(),
                        axis=0,
                    )
            else:
                if preds is None:
                    preds = logits.detach().cpu().numpy()
                    out_label_ids = inputs["labels"].detach().cpu().numpy()
                else:
                    preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                    out_label_ids = np.append(
                        out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0
                    )

        eval_loss = eval_loss / nb_eval_steps
        model_outputs = preds

        if not multi_label:
            preds = np.argmax(preds, axis=1)

        result, wrong = self.compute_metrics(
            preds, out_label_ids, eval_examples, **kwargs
        )
        result["eval_loss"] = eval_loss
        results.update(result)

        output_eval_file = os.path.join(eval_output_dir, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            for key in sorted(result.keys()):
                writer.write("{} = {}\n".format(key, str(result[key])))

        return results, model_outputs, wrong

    def load_and_cache_examples(
        self, examples, evaluate=False, no_cache=False, multi_label=False
    ):
        """
        Converts a list of InputExample objects to a TensorDataset containing InputFeatures. Caches the InputFeatures.

        Utility function for train() and eval() methods. Not intended to be used directly.
        """

        process_count = self.args["process_count"]

        tokenizer = self.tokenizer
        output_mode = "classification"
        args = self.args

        if not os.path.isdir(self.args["cache_dir"]) and not no_cache:
            os.makedirs(self.args["cache_dir"])

        mode = "dev" if evaluate else "train"
        cached_features_file = os.path.join(
            args["cache_dir"],
            "cached_{}_{}_{}_{}_{}".format(
                mode,
                args["model_type"],
                args["max_seq_length"],
                self.num_labels,
                len(examples),
            ),
        )

        if (
            os.path.exists(cached_features_file)
            and not args["reprocess_input_data"]
            and not no_cache
        ):
            features = torch.load(cached_features_file)
            print(f"Features loaded from cache at {cached_features_file}")
        else:
            print(f"Converting to features started.")
            features = convert_examples_to_features(
                examples,
                args["max_seq_length"],
                tokenizer,
                output_mode,
                # XLNet has a CLS token at the end
                cls_token_at_end=bool(args["model_type"] in ["xlnet"]),
                cls_token=tokenizer.cls_token,
                cls_token_segment_id=2 if args["model_type"] in ["xlnet"] else 0,
                sep_token=tokenizer.sep_token,
                # RoBERTa uses an extra separator b/w pairs of sentences, cf. github.com/pytorch/fairseq/commit/1684e166e3da03f5b600dbb7855cb98ddfcd0805
                sep_token_extra=bool(args["model_type"] in ["roberta"]),
                # PAD on the left for XLNet
                pad_on_left=bool(args["model_type"] in ["xlnet"]),
                pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
                pad_token_segment_id=4 if args["model_type"] in ["xlnet"] else 0,
                process_count=process_count,
                multi_label=multi_label,
                silent=args["silent"],
                use_multiprocessing=args["use_multiprocessing"],
                sliding_window=self.sliding_window,
                stride=self.args["stride"],
            )

            if not no_cache:
                torch.save(features, cached_features_file)

        if self.sliding_window:
            # features = pad_sequence([torch.tensor(features_per_sequence) for features_per_sequence in features])
            all_input_ids = pad_sequence(
                [
                    torch.tensor(
                        [f.input_ids for f in features_per_sequence], dtype=torch.long
                    )
                    for features_per_sequence in features
                ],
                batch_first=True,
            )
            all_input_mask = pad_sequence(
                [
                    torch.tensor(
                        [f.input_mask for f in features_per_sequence], dtype=torch.long
                    )
                    for features_per_sequence in features
                ],
                batch_first=True,
            )
            all_segment_ids = pad_sequence(
                [
                    torch.tensor(
                        [f.segment_ids for f in features_per_sequence], dtype=torch.long
                    )
                    for features_per_sequence in features
                ],
                batch_first=True,
            )

            # all_input_ids = torch.tensor([f.input_ids for feature in features for f in feature], dtype=torch.long)
            # all_input_mask = torch.tensor([f.input_mask for feature in features for f in feature], dtype=torch.long)
            # all_segment_ids = torch.tensor([f.segment_ids for feature in features for f in feature], dtype=torch.long)

            if output_mode == "classification":
                all_label_ids = pad_sequence(
                    [
                        torch.tensor(
                            [f.label_id for f in features_per_sequence],
                            dtype=torch.long,
                        )
                        for features_per_sequence in features
                    ],
                    batch_first=True,
                )
            elif output_mode == "regression":
                all_label_ids = pad_sequence(
                    [
                        torch.tensor(
                            [f.label_id for f in features_per_sequence],
                            dtype=torch.float,
                        )
                        for features_per_sequence in features
                    ],
                    batch_first=True,
                )
        else:
            all_input_ids = torch.tensor(
                [f.input_ids for f in features], dtype=torch.long
            )
            all_input_mask = torch.tensor(
                [f.input_mask for f in features], dtype=torch.long
            )
            all_segment_ids = torch.tensor(
                [f.segment_ids for f in features], dtype=torch.long
            )

            if output_mode == "classification":
                all_label_ids = torch.tensor(
                    [f.label_id for f in features], dtype=torch.long
                )
            elif output_mode == "regression":
                all_label_ids = torch.tensor(
                    [f.label_id for f in features], dtype=torch.float
                )

        dataset = TensorDataset(
            all_input_ids, all_input_mask, all_segment_ids, all_label_ids
        )

        return dataset

    def compute_metrics(
        self, preds, labels, eval_examples, multi_label=False, **kwargs
    ):
        """
        Computes the evaluation metrics for the model predictions.

        Args:
            preds: Model predictions
            labels: Ground truth labels
            eval_examples: List of examples on which evaluation was performed
            **kwargs: Additional metrics that should be used. Pass in the metrics as keyword arguments (name of metric: function to use). E.g. f1=sklearn.metrics.f1_score.
                        A metric function should take in two parameters. The first parameter will be the true labels, and the second parameter will be the predictions.

        Returns:
            result: Dictionary containing evaluation results. (Matthews correlation coefficient, tp, tn, fp, fn)
            wrong: List of InputExample objects corresponding to each incorrect prediction by the model
        """

        assert len(preds) == len(labels)

        extra_metrics = {}
        for metric, func in kwargs.items():
            extra_metrics[metric] = func(labels, preds)

        mismatched = labels != preds
        wrong = [i for (i, v) in zip(eval_examples, mismatched) if v.any()]

        if multi_label:
            label_ranking_score = label_ranking_average_precision_score(labels, preds)
            return {**{"LRAP": label_ranking_score}, **extra_metrics}, wrong

        mcc = matthews_corrcoef(labels, preds)

        if self.model.num_labels == 2:
            tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()
            return (
                {
                    **{"mcc": mcc, "tp": tp, "tn": tn, "fp": fp, "fn": fn},
                    **extra_metrics,
                },
                wrong,
            )

        else:
            return {**{"mcc": mcc}, **extra_metrics}, wrong

    def predict(self, to_predict, multi_label=False):
        """
        Performs predictions on a list of text.

        Args:
            to_predict: A python list of text (str) to be sent to the model for prediction.

        Returns:
            preds: A python list of the predictions (0 or 1) for each text.
            model_outputs: A python list of the raw model outputs for each text.
        """

        tokenizer = self.tokenizer
        device = self.device
        model = self.model
        args = self.args

        self._move_model_to_device()

        if multi_label:
            eval_examples = [
                InputExample(i, text, None, [0 for i in range(self.num_labels)])
                for i, text in enumerate(to_predict)
            ]
        else:
            eval_examples = [
                InputExample(i, text, None, 0) for i, text in enumerate(to_predict)
            ]

        eval_dataset = self.load_and_cache_examples(
            eval_examples, evaluate=True, multi_label=multi_label, no_cache=True
        )

        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(
            eval_dataset, sampler=eval_sampler, batch_size=args["eval_batch_size"]
        )

        eval_loss = 0.0
        nb_eval_steps = 0
        preds = None
        out_label_ids = None

        for batch in tqdm(eval_dataloader, disable=args["silent"]):
            model.eval()
            batch = tuple(t.to(device) for t in batch)

            with torch.no_grad():
                inputs = self._get_inputs_dict(batch)

                if self.sliding_window:
                    outputs = model(inputs)
                else:
                    outputs = model(**inputs)

                tmp_eval_loss, logits = outputs[:2]

                if multi_label:
                    logits = logits.sigmoid()

                if self.args.n_gpu > 1:
                    tmp_eval_loss = tmp_eval_loss.mean()
                eval_loss += tmp_eval_loss.item()

            nb_eval_steps += 1

            if self.sliding_window:
                if preds is None:
                    preds = logits.detach().cpu().numpy()
                    out_label_ids = inputs[0]["labels"].detach().cpu().numpy()
                else:
                    preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                    out_label_ids = np.append(
                        out_label_ids,
                        inputs[0]["labels"].detach().cpu().numpy(),
                        axis=0,
                    )
            else:
                if preds is None:
                    preds = logits.detach().cpu().numpy()
                    out_label_ids = inputs["labels"].detach().cpu().numpy()
                else:
                    preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                    out_label_ids = np.append(
                        out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0
                    )

        eval_loss = eval_loss / nb_eval_steps
        model_outputs = preds
        if multi_label:
            if isinstance(args["threshold"], list):
                threshold_values = args["threshold"]
                preds = [
                    [
                        self._threshold(pred, threshold_values[i])
                        for i, pred in enumerate(example)
                    ]
                    for example in preds
                ]
            else:
                preds = [
                    [self._threshold(pred, args["threshold"]) for pred in example]
                    for example in preds
                ]
        else:
            preds = np.argmax(preds, axis=1)

        return preds, model_outputs

    def _threshold(self, x, threshold):
        if x >= threshold:
            return 1
        return 0

    def _move_model_to_device(self):
        self.model.to(self.device)

    def _get_inputs_dict(self, batch):
        if self.sliding_window:
            inputs_all = []
            inputs = batch[0].permute(1, 0, 2)
            attentions = batch[1].permute(1, 0, 2)
            labels = batch[3].permute(1, 0)

            if self.args["model_type"] != "distilbert":
                tokens = (
                    batch[2].permute(1, 0, 2)
                    if self.args["model_type"] in ["bert", "xlnet"]
                    else None
                )

            for i in range(len(labels)):
                input_single = {
                    "input_ids": inputs[i],
                    "attention_mask": attentions[i],
                    "labels": labels[i],
                }

                # XLM, DistilBERT and RoBERTa don't use segment_ids
                if self.args["model_type"] != "distilbert":
                    input_single["token_type_ids"] = (
                        tokens[i]
                        if self.args["model_type"] in ["bert", "xlnet"]
                        else None
                    )
                inputs_all.append(input_single)
            return inputs_all
        else:
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "labels": batch[3],
            }

            # XLM, DistilBERT and RoBERTa don't use segment_ids
            if self.args["model_type"] != "distilbert":
                inputs["token_type_ids"] = (
                    batch[2] if self.args["model_type"] in ["bert", "xlnet"] else None
                )

            return inputs
