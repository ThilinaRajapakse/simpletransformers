from __future__ import absolute_import, division, print_function

import os
import math
import json
import random
import warnings

from multiprocessing import cpu_count

import torch
import numpy as np


from scipy.stats import pearsonr
from seqeval.metrics import precision_score, recall_score, f1_score
from tensorboardX import SummaryWriter
from tqdm.auto import trange, tqdm

from torch.nn import CrossEntropyLoss
from torch.utils.data import (
    DataLoader,
    RandomSampler,
    SequentialSampler,
    TensorDataset
)

from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import WEIGHTS_NAME, BertConfig, BertForTokenClassification, BertTokenizer
from transformers import DistilBertConfig, DistilBertForTokenClassification, DistilBertTokenizer
try:
    from transformers import RobertaConfig, RobertaForTokenClassification, RobertaTokenizer
    roberta_available = True
except ImportError:
    print("Warning: Importing RobertaForTokenClassification unsuccessful. Please use BERT for now. See issue on https://github.com/huggingface/transformers/issues/1631.")
    roberta_available = False


from simpletransformers.ner.ner_utils import InputExample, convert_examples_to_features, get_labels, read_examples_from_file, get_examples_from_df
from transformers import CamembertConfig, CamembertForTokenClassification, CamembertTokenizer

class NERModel:
    def __init__(self, model_type, model_name, labels=None, args=None, use_cuda=True, cuda_device=-1):
        """
        Initializes a NERModel

        Args:
            model_type: The type of model (bert, roberta)
            model_name: Default Transformer model name or path to a directory containing Transformer model file (pytorch_nodel.bin).
            labels (optional): A list of all Named Entity labels.  If not given, ["O", "B-MISC", "I-MISC",  "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"] will be used.
            args (optional): Default args will be used if this parameter is not provided. If provided, it should be a dict containing the args that should be changed in the default args.
            use_cuda (optional): Use GPU if available. Setting to False will force model to use CPU only.
            cuda_device (optional): Specific GPU that should be used. Will use the first available GPU by default.
        """

        if labels:
            self.labels = labels
        else:
            self.labels = ["O", "B-MISC", "I-MISC",  "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"]
        self.num_labels = len(self.labels)

        if roberta_available:
            MODEL_CLASSES = {
                'bert': (BertConfig, BertForTokenClassification, BertTokenizer),
                'roberta': (RobertaConfig, RobertaForTokenClassification, RobertaTokenizer),
                'distilbert': (DistilBertConfig, DistilBertForTokenClassification, DistilBertTokenizer),
                'camembert': (CamembertConfig, CamembertForTokenClassification, CamembertTokenizer)
            }
        else:
            MODEL_CLASSES = {
                'bert': (BertConfig, BertForTokenClassification, BertTokenizer),
                'distilbert': (DistilBertConfig, DistilBertForTokenClassification, BertTokenizer),
                'camembert': (CamembertConfig, CamembertForTokenClassification, CamembertTokenizer)
            }

        config_class, model_class, tokenizer_class = MODEL_CLASSES[model_type]

        self.model = model_class.from_pretrained(model_name, num_labels=self.num_labels)

        if use_cuda:
            if torch.cuda.is_available():
                if cuda_device == -1:
                    self.device = torch.device("cuda")
                else:
                    self.device = torch.device(f"cuda:{cuda_device}")
            else:
                raise ValueError("'use_cuda' set to True when cuda is unavailable. Make sure CUDA is available or set use_cuda=False.")
        else:
            self.device = "cpu"

        self.results = {}

        self.args = {
            'output_dir': 'outputs/',
            'cache_dir': 'cache_dir/',

            'fp16': True,
            'fp16_opt_level': 'O1',
            'max_seq_length': 128,
            'train_batch_size': 8,
            'gradient_accumulation_steps': 1,
            'eval_batch_size': 8,
            'num_train_epochs': 1,
            'weight_decay': 0,
            'learning_rate': 4e-5,
            'adam_epsilon': 1e-8,
            'warmup_ratio': 0.06,
            'warmup_steps': 0,
            'max_grad_norm': 1.0,
            'do_lower_case': False,

            'logging_steps': 50,
            'save_steps': 2000,
            'evaluate_during_training': False,
            'evaluate_during_training_steps': 2000,

            'overwrite_output_dir': False,
            'reprocess_input_data': False,

            'process_count': cpu_count() - 2 if cpu_count() > 2 else 1,
            'n_gpu': 1,
            'silent': False,
            'use_multiprocessing': True,
        }

        if not use_cuda:
            self.args['fp16'] = False

        if args:
            self.args.update(args)

        self.tokenizer = tokenizer_class.from_pretrained(model_name, do_lower_case=self.args['do_lower_case'])

        self.args['model_name'] = model_name
        self.args['model_type'] = model_type

        self.pad_token_label_id = CrossEntropyLoss().ignore_index

        if model_type == 'camembert':
            warnings.warn("use_multiprocessing automatically disabled as CamemBERT fails when using multiprocessing for feature conversion.")
            self.args['use_multiprocessing'] = False

    def train_model(self, train_data, output_dir=None, show_running_loss=True, args=None, eval_df=None):
        """
        Trains the model using 'train_data'

        Args:
            train_data: train_data should be the path to a .txt file containing the training data OR a pandas DataFrame with 3 columns.
                        If a text file is given the data should be in the CoNLL format. i.e. One word per line, with sentences seperated by an empty line. 
                        The first word of the line should be a word, and the last should be a Name Entity Tag.
                        If a DataFrame is given, each sentence should be split into words, with each word assigned a tag, and with all words from the same sentence given the same sentence_id.

            output_dir: The directory where model files will be saved. If not given, self.args['output_dir'] will be used.
            show_running_loss (optional): Set to False to prevent running loss from being printed to console. Defaults to True.
            args (optional): Optional changes to the args dict of the model. Any changes made will persist for the model.

        Returns:
            None
        """

        if args:
            self.args.update(args)

        if self.args['silent']:
            show_running_loss = False

        if self.args['evaluate_during_training'] and eval_df is None:
            raise ValueError("evaluate_during_training is enabled but eval_df is not specified. Pass eval_df to model.train_model() if using evaluate_during_training.")

        if not output_dir:
            output_dir = self.args['output_dir']

        if os.path.exists(output_dir) and os.listdir(output_dir) and not self.args["overwrite_output_dir"]:
            raise ValueError("Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(output_dir))

        self._move_model_to_device()

        train_dataset = self.load_and_cache_examples(train_data)

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        global_step, tr_loss = self.train(train_dataset, output_dir, show_running_loss=show_running_loss, eval_df=eval_df)

        model_to_save = self.model.module if hasattr(self.model, "module") else self.model
        model_to_save.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        torch.save(self.args, os.path.join(output_dir, "training_args.bin"))

        print("Training of {} model complete. Saved to {}.".format(self.args["model_type"], output_dir))


    def train(self, train_dataset, output_dir, show_running_loss=True, eval_df=None):
        """
        Trains the model on train_dataset.

        Utility function to be used by the train_model() method. Not intended to be used directly.
        """

        tokenizer = self.tokenizer
        device = self.device
        model = self.model
        args = self.args

        tb_writer = SummaryWriter()
        train_sampler = RandomSampler(train_dataset)
        train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args["train_batch_size"])

        t_total = len(train_dataloader) // args["gradient_accumulation_steps"] * args["num_train_epochs"]

        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {"params": [p for n, p in model.named_parameters() if not any(
                nd in n for nd in no_decay)], "weight_decay": args["weight_decay"]},
            {"params": [p for n, p in model.named_parameters() if any(
                nd in n for nd in no_decay)], "weight_decay": 0.0}
        ]

        warmup_steps = math.ceil(t_total * args["warmup_ratio"])
        args["warmup_steps"] = warmup_steps if args["warmup_steps"] == 0 else args["warmup_steps"]

        optimizer = AdamW(optimizer_grouped_parameters, lr=args["learning_rate"], eps=args["adam_epsilon"])
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args["warmup_steps"], num_training_steps=t_total)

        if args["fp16"]:
            try:
                from apex import amp
            except ImportError:
                raise ImportError(
                    "Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")

            model, optimizer = amp.initialize(model, optimizer, opt_level=args["fp16_opt_level"])

        if args["n_gpu"] > 1:
            model = torch.nn.DataParallel(model)

        global_step = 0
        tr_loss, logging_loss = 0.0, 0.0
        model.zero_grad()
        train_iterator = trange(int(args["num_train_epochs"]), desc="Epoch", disable=args['silent'])
        epoch_number = 0

        for _ in train_iterator:
            # epoch_iterator = tqdm(train_dataloader, desc="Iteration")
            for step, batch in enumerate(tqdm(train_dataloader, desc="Current iteration", disable=args['silent'])):
                model.train()
                batch = tuple(t.to(device) for t in batch)

                inputs = {"input_ids": batch[0],
                      "attention_mask": batch[1],
                      "labels": batch[3]}
                # XLM and RoBERTa don"t use segment_ids
                if args['model_type'] in ["bert", "xlnet"]:
                    inputs["token_type_ids"] = batch[2] 

                outputs = model(**inputs)
                # model outputs are always tuple in pytorch-transformers (see doc)
                loss = outputs[0]

                if args['n_gpu'] > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu parallel training

                if show_running_loss:
                    print("\rRunning loss: %f" % loss, end="")

                if args["gradient_accumulation_steps"] > 1:
                    loss = loss / args["gradient_accumulation_steps"]

                if args["fp16"]:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args["max_grad_norm"])
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args["max_grad_norm"])

                tr_loss += loss.item()
                if (step + 1) % args["gradient_accumulation_steps"] == 0:
                    optimizer.step()
                    scheduler.step()  # Update learning rate schedule
                    model.zero_grad()
                    global_step += 1

                    if args["logging_steps"] > 0 and global_step % args["logging_steps"] == 0:
                        # Log metrics
                        tb_writer.add_scalar("lr", scheduler.get_lr()[0], global_step)
                        tb_writer.add_scalar("loss", (tr_loss - logging_loss)/args["logging_steps"], global_step)
                        logging_loss = tr_loss

                    if args["save_steps"] > 0 and global_step % args["save_steps"] == 0:
                        # Save model checkpoint
                        output_dir_current = os.path.join(output_dir, "checkpoint-{}".format(global_step))

                        if not os.path.exists(output_dir_current):
                            os.makedirs(output_dir_current)

                        # Take care of distributed/parallel training
                        model_to_save = model.module if hasattr(model, "module") else model
                        model_to_save.save_pretrained(output_dir_current)
                        self.tokenizer.save_pretrained(output_dir_current)

                    if args['evaluate_during_training'] and (args["evaluate_during_training_steps"] > 0 and global_step % args["evaluate_during_training_steps"] == 0):
                    # Only evaluate when single GPU otherwise metrics may not average well
                        results, _, _ = self.eval_model(eval_df, verbose=True)
                        for key, value in results.items():
                            tb_writer.add_scalar('eval_{}'.format(key), value, global_step)

                        output_dir_current = os.path.join(output_dir, "checkpoint-{}".format(global_step))

                        if not os.path.exists(output_dir_current):
                            os.makedirs(output_dir_current)

                        model_to_save = model.module if hasattr(model, "module") else model
                        model_to_save.save_pretrained(output_dir_current)
                        self.tokenizer.save_pretrained(output_dir_current)

                        output_eval_file = os.path.join(output_dir_current, "eval_results.txt")
                        with open(output_eval_file, "w") as writer:
                            for key in sorted(results.keys()):
                                writer.write("{} = {}\n".format(key, str(results[key])))

            epoch_number += 1
            output_dir_current = os.path.join(output_dir, "epoch-{}".format(epoch_number))

            if not os.path.exists(output_dir_current):
                os.makedirs(output_dir_current)

            model_to_save = model.module if hasattr(model, "module") else model
            model_to_save.save_pretrained(output_dir_current)
            self.tokenizer.save_pretrained(output_dir_current)

            if args['evaluate_during_training']:
                results, _, _ = self.eval_model(eval_df, verbose=True)

                output_eval_file = os.path.join(output_dir_current, "eval_results.txt")
                with open(output_eval_file, "w") as writer:
                    for key in sorted(results.keys()):
                        writer.write("{} = {}\n".format(key, str(results[key])))


        return global_step, tr_loss / global_step


    def eval_model(self, eval_data, output_dir=None, verbose=True):
        """
        Evaluates the model on eval_data. Saves results to output_dir.

        Args:
            eval_file: eval_data should be the path to a .txt file containing the evaluation data or a pandas DataFrame. 
                        If a text file is used the data should be in the CoNLL format. I.e. One word per line, with sentences seperated by an empty line. 
                        The first word of the line should be a word, and the last should be a Name Entity Tag.
                        If a DataFrame is given, each sentence should be split into words, with each word assigned a tag, and with all words from the same sentence given the same sentence_id.

            output_dir: The directory where model files will be saved. If not given, self.args['output_dir'] will be used.
            verbose: If verbose, results will be printed to the console on completion of evaluation.
            
        Returns:
            result: Dictionary containing evaluation results. (eval_loss, precision, recall, f1_score)
            model_outputs: List of raw model outputs
            preds_list: List of predicted tags
        """
        if not output_dir:
            output_dir = self.args["output_dir"]

        self._move_model_to_device()

        eval_dataset = self.load_and_cache_examples(eval_data, evaluate=True)

        result, model_outputs, preds_list = self.evaluate(eval_dataset, output_dir)
        self.results.update(result)

        if verbose:
            print(self.results)

        return result, model_outputs, preds_list

    def evaluate(self, eval_dataset, output_dir):
        """
        Evaluates the model on eval_dataset.

        Utility function to be used by the eval_model() method. Not intended to be used directly.
        """

        tokenizer = self.tokenizer
        device = self.device
        model = self.model
        args = self.args
        pad_token_label_id = self.pad_token_label_id
        eval_output_dir = output_dir

        results = {}

        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args["eval_batch_size"])

        eval_loss = 0.0
        nb_eval_steps = 0
        preds = None
        out_label_ids = None
        model.eval()

        for batch in tqdm(eval_dataloader, disable=args['silent']):
            batch = tuple(t.to(device) for t in batch)

            with torch.no_grad():
                inputs = {"input_ids": batch[0],
                      "attention_mask": batch[1],
                      "labels": batch[3]}
                # XLM and RoBERTa don"t use segment_ids
                if args['model_type'] in ["bert", "xlnet"]:
                    inputs["token_type_ids"] = batch[2] 
                outputs = model(**inputs)
                tmp_eval_loss, logits = outputs[:2]

                eval_loss += tmp_eval_loss.mean().item()

            nb_eval_steps += 1

            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = inputs["labels"].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0)

        eval_loss = eval_loss / nb_eval_steps
        model_outputs = preds
        preds = np.argmax(preds, axis=2)

        label_map = {i: label for i, label in enumerate(self.labels)}

        out_label_list = [[] for _ in range(out_label_ids.shape[0])]
        preds_list = [[] for _ in range(out_label_ids.shape[0])]

        for i in range(out_label_ids.shape[0]):
            for j in range(out_label_ids.shape[1]):
                if out_label_ids[i, j] != pad_token_label_id:
                    out_label_list[i].append(label_map[out_label_ids[i][j]])
                    preds_list[i].append(label_map[preds[i][j]])

        result = {
            "eval_loss": eval_loss,
            "precision": precision_score(out_label_list, preds_list),
            "recall": recall_score(out_label_list, preds_list),
            "f1_score": f1_score(out_label_list, preds_list)
        }

        results.update(result)

        output_eval_file = os.path.join(eval_output_dir, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            for key in sorted(result.keys()):
                writer.write("{} = {}\n".format(key, str(result[key])))

        return results, model_outputs, preds_list

    
    def predict(self, to_predict):
        """
        Performs predictions on a list of text.

        Args:
            to_predict: A python list of text (str) to be sent to the model for prediction.

        Returns:
            preds: A Python list of lists with dicts containg each word mapped to its NER tag.
            model_outputs: A python list of the raw model outputs for each text.
        """

        tokenizer = self.tokenizer
        device = self.device
        model = self.model
        args = self.args
        pad_token_label_id = self.pad_token_label_id

        self._move_model_to_device()

        predict_examples = [InputExample(i, sentence.split(), ["O" for word in sentence.split()]) for i, sentence in enumerate(to_predict)]

        eval_dataset = self.load_and_cache_examples(None, to_predict=predict_examples)

        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args["eval_batch_size"])

        eval_loss = 0.0
        nb_eval_steps = 0
        preds = None
        out_label_ids = None
        model.eval()

        for batch in tqdm(eval_dataloader, disable=args['silent']):
            batch = tuple(t.to(device) for t in batch)

            with torch.no_grad():
                inputs = {"input_ids": batch[0],
                      "attention_mask": batch[1],
                      "labels": batch[3]}
                # XLM and RoBERTa don"t use segment_ids
                if args['model_type'] in ["bert", "xlnet"]:
                    inputs["token_type_ids"] = batch[2] 
                outputs = model(**inputs)
                tmp_eval_loss, logits = outputs[:2]

                eval_loss += tmp_eval_loss.mean().item()

            nb_eval_steps += 1

            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = inputs["labels"].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0)

        eval_loss = eval_loss / nb_eval_steps
        model_outputs = preds
        preds = np.argmax(preds, axis=2)

        label_map = {i: label for i, label in enumerate(self.labels)}

        out_label_list = [[] for _ in range(out_label_ids.shape[0])]
        preds_list = [[] for _ in range(out_label_ids.shape[0])]

        for i in range(out_label_ids.shape[0]):
            for j in range(out_label_ids.shape[1]):
                if out_label_ids[i, j] != pad_token_label_id:
                    out_label_list[i].append(label_map[out_label_ids[i][j]])
                    preds_list[i].append(label_map[preds[i][j]])

        preds = [[{word: preds_list[i][j]} for j, word in enumerate(sentence.split()[:len(preds_list[i])])] for i, sentence in enumerate(to_predict)]

        return preds, model_outputs


    def load_and_cache_examples(self, data, evaluate=False, no_cache=False, to_predict=None):
        """
        Reads data_file and generates a TensorDataset containing InputFeatures. Caches the InputFeatures.
        Utility function for train() and eval() methods. Not intended to be used directly.

        Args:
            data: Path to a .txt file containing training or evaluation data OR a pandas DataFrame containing 3 columns - sentence_id, words, labels.
                    If a DataFrame is given, each sentence should be split into words, with each word assigned a tag, and with all words from the same sentence given the same sentence_id.
            evaluate (optional): Indicates whether the examples are for evaluation or for training.
            no_cache (optional): Force feature conversion and prevent caching. I.e. Ignore cached features even if present.

        """

        process_count = self.args["process_count"]

        tokenizer = self.tokenizer
        output_mode = "classification"
        args = self.args

        mode = "dev" if evaluate else "train"

        if not to_predict:
            if isinstance(data, str):
                examples = read_examples_from_file(data, mode)
            else:
                examples = get_examples_from_df(data)
        else:
            examples = to_predict
            no_cache = True

        cached_features_file = os.path.join(args["cache_dir"], "cached_{}_{}_{}_{}_{}".format(mode, args["model_type"], args["max_seq_length"], self.num_labels, len(examples)))


        if not os.path.isdir(self.args["cache_dir"]):
            os.mkdir(self.args["cache_dir"])

        if os.path.exists(cached_features_file) and not args["reprocess_input_data"] and not no_cache:
            features = torch.load(cached_features_file)
            print(f"Features loaded from cache at {cached_features_file}")
        else:
            print(f"Converting to features started.")
            features = convert_examples_to_features(
                examples,
                self.labels,
                self.args['max_seq_length'],
                self.tokenizer,
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
                pad_token_label_id=self.pad_token_label_id,
                process_count=process_count,
                silent=args['silent'],
                use_multiprocessing=args['use_multiprocessing']
            )

            if not no_cache:
                torch.save(features, cached_features_file)

        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_ids for f in features], dtype=torch.long)

        dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)

        return dataset


    def _move_model_to_device(self):
        self.model.to(self.device)
        
