#!/usr/bin/env python
# coding: utf-8


from __future__ import absolute_import, division, print_function

import os
import random
import json

import numpy as np
from sklearn.metrics import mean_squared_error, matthews_corrcoef, confusion_matrix
from scipy.stats import pearsonr
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler

from transformers import (WEIGHTS_NAME, BertConfig,
                                  BertForSequenceClassification, BertTokenizer,
                                  RobertaConfig,
                                  RobertaForSequenceClassification,
                                  RobertaTokenizer,
                                  XLMConfig, XLMForSequenceClassification,
                                  XLMTokenizer, XLNetConfig,
                                  XLNetForSequenceClassification,
                                  XLNetTokenizer,
                                  DistilBertConfig,
                                  DistilBertForSequenceClassification,
                                  DistilBertTokenizer)

from transformers import AdamW, WarmupLinearSchedule

from simpletransformers.utils import (convert_examples_to_features, BinaryProcessor, InputExample)

import math
from tensorboardX import SummaryWriter
from tqdm import trange, tqdm



class TransformerModel:
    def __init__(self, model_type, model_name, args=None, use_cuda=True):
        """
        Initializes a Transformer model.

        Args:
            model_type: The type of model (bert, xlnet, xlm, roberta)
            model_name: Default Transformer model name or path to Transformer model file (pytorch_nodel.bin).
            args: Default args will be used if this parameter is not provided. If provided, it should be a dict containing the args that should be changed in the default args.
            use_cuda: Use GPU if available. Setting to False will force model to use CPU only.

        """

        MODEL_CLASSES = {
                    'bert': (BertConfig, BertForSequenceClassification, BertTokenizer),
                    'xlnet': (XLNetConfig, XLNetForSequenceClassification, XLNetTokenizer),
                    'xlm': (XLMConfig, XLMForSequenceClassification, XLMTokenizer),
                    'roberta': (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer),
                }

        config_class, model_class, tokenizer_class = MODEL_CLASSES[model_type]
        self.tokenizer = tokenizer_class.from_pretrained(model_name)
        self.model = model_class.from_pretrained(model_name)
        
        if use_cuda:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = "cpu"
        
        self.results = {}

        self.args = {
            'data_dir': 'data/',
            'model_type':  'roberta',
            'model_name': 'roberta-base',
            'task_name': 'binary',
            'output_dir': 'outputs/',

            'fp16': True,
            'fp16_opt_level': 'O1',
            'max_seq_length': 128,
            'train_batch_size': 8,
            'eval_batch_size': 8,
            'gradient_accumulation_steps': 1,
            'num_train_epochs': 1,
            'weight_decay': 0,
            'learning_rate': 4e-5,
            'adam_epsilon': 1e-8,
            'warmup_ratio': 0.06,
            'warmup_steps': 0,
            'max_grad_norm': 1.0,

            'logging_steps': 50,
            'evaluate_during_training': False,
            'save_steps': 2000,
            'eval_all_checkpoints': True,
            'use_tensorboard': True,

            'overwrite_output_dir': False,
            'reprocess_input_data': False,
        }

        if args:
            self.args.update(args)

    def train_model(self, train_df, output_dir=None):
        """
        Trains the model using 'train_df'

        Args:
            train_df: Pandas Dataframe (no header) of two columns, first column containing the text, and the second column containing the label. The model will be trained on this Dataframe.
            output_dir: The directory where model files will be saved. If not given, self.args['output_dir'] will be used.

        Returns:
            None

        """
        if not output_dir:
            output_dir = self.args['output_dir']
        
        self.model.to(self.device)

        train_examples = [InputExample(i, text, None, label) for i, (text, label) in enumerate(zip(train_df.iloc[:, 0], train_df.iloc[:, 1]))]

        train_dataset = self.load_and_cache_examples(train_examples)
        global_step, tr_loss = self.train(train_dataset, output_dir)
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
        model_to_save.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        torch.save(self.args, os.path.join(output_dir, 'training_args.bin'))

        print(f'Training of {self.args["model_type"]} model complete. Saved to {output_dir}.')

    def eval_model(self, eval_df, output_dir=None, verbose=False):
        """
        Evaluates the model on eval_df. Saves results to output_dir.

        Args:
            eval_df: Pandas Dataframe (no header) of two columns, first column containing the text, and the second column containing the label. The model will be evaluated on this Dataframe.
            output_dir: The directory where model files will be saved. If not given, self.args['output_dir'] will be used.
            verbose: If verbose, results will be printed to the console on completion of evaluation.

        Returns:
            result: Dictionary containing evaluation results. (Matthews correlation coefficient, tp, tn, fp, fn)
            model_outputs: List of model outputs for each row in eval_df
            wrong_preds: List of InputExample objects corresponding to each incorrect prediction by the model

        """

        if not output_dir:
            output_dir = self.args['output_dir']

        self.model.to(self.device)

        result, model_outputs, wrong_preds = self.evaluate(eval_df, output_dir)
        self.results.update(result)

        if not verbose:
            print(self.results)

        return result, model_outputs, wrong_preds

    def evaluate(self, eval_df, output_dir, prefix=""):
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
        EVAL_TASK = args['task_name']

        eval_examples = [InputExample(i, text, None, label) for i, (text, label) in enumerate(zip(eval_df.iloc[:, 0], eval_df.iloc[:, 1]))]

        eval_dataset = self.load_and_cache_examples(eval_examples, evaluate=True)
        if not os.path.exists(eval_output_dir):
            os.makedirs(eval_output_dir)

        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args['eval_batch_size'])

        eval_loss = 0.0
        nb_eval_steps = 0
        preds = None
        out_label_ids = None
        for batch in tqdm(eval_dataloader):
            model.eval()
            batch = tuple(t.to(device) for t in batch)

            with torch.no_grad():
                inputs = {'input_ids':      batch[0],
                        'attention_mask': batch[1],
                        # XLM don't use segment_ids
                        'token_type_ids': batch[2] if args['model_type'] in ['bert', 'xlnet'] else None,
                        'labels':         batch[3]}
                outputs = model(**inputs)
                tmp_eval_loss, logits = outputs[:2]

                eval_loss += tmp_eval_loss.mean().item()
            nb_eval_steps += 1
            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = inputs['labels'].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(
                    out_label_ids, inputs['labels'].detach().cpu().numpy(), axis=0)

        eval_loss = eval_loss / nb_eval_steps
        model_outputs = preds
        preds = np.argmax(preds, axis=1)
        result, wrong = self.compute_metrics(EVAL_TASK, preds, out_label_ids)
        results.update(result)

        output_eval_file = os.path.join(eval_output_dir, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            for key in sorted(result.keys()):
                writer.write("%s = %s\n" % (key, str(result[key])))

        return results, model_outputs, wrong





    def load_and_cache_examples(self, examples, evaluate=False):
        """
        Converts a list of InputExample objects to a TensorDataset containing InputFeatures. Caches the InputFeatures.

        Utility function for train() and eval() methods. Not intended to be used directly.
        """
        processor = BinaryProcessor()
        tokenizer = self.tokenizer
        output_mode = 'classification'
        args=self.args

        mode = 'dev' if evaluate else 'train'
        cached_features_file = os.path.join(args['data_dir'], f"cached_{mode}_{args['model_name']}_{args['max_seq_length']}_binary")

        if os.path.exists(cached_features_file) and not args['reprocess_input_data']:
            features = torch.load(cached_features_file)

        else:
            label_list = processor.get_labels()

            features = convert_examples_to_features(examples, label_list, args['max_seq_length'], tokenizer, output_mode,
                                                    # xlnet has a cls token at the end
                                                    cls_token_at_end=bool(
                                                        args['model_type'] in ['xlnet']),
                                                    cls_token=tokenizer.cls_token,
                                                    cls_token_segment_id=2 if args['model_type'] in [
                                                        'xlnet'] else 0,
                                                    sep_token=tokenizer.sep_token,
                                                    # roberta uses an extra separator b/w pairs of sentences, cf. github.com/pytorch/fairseq/commit/1684e166e3da03f5b600dbb7855cb98ddfcd0805
                                                    sep_token_extra=bool(
                                                        args['model_type'] in ['roberta']),
                                                    # pad on the left for xlnet
                                                    pad_on_left=bool(
                                                        args['model_type'] in ['xlnet']),
                                                    pad_token=tokenizer.convert_tokens_to_ids(
                [tokenizer.pad_token])[0],
                pad_token_segment_id=4 if args['model_type'] in ['xlnet'] else 0)

            torch.save(features, cached_features_file)

        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
        if output_mode == "classification":
            all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
        elif output_mode == "regression":
            all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.float)

        dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
        return dataset


    def train(self, train_dataset, output_dir):
        """
        Trains the model on train_dataset.

        Utility function to be used by the train_model() method. Not intended to be used directly.
        """
        tokenizer = self.tokenizer
        device = self.device
        model = self.model
        args = self.args
        if args['use_tensorboard']:
            tb_writer = SummaryWriter()
        train_sampler = RandomSampler(train_dataset)
        train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args['train_batch_size'])

        t_total = len(train_dataloader) // args['gradient_accumulation_steps'] * args['num_train_epochs']

        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(
                nd in n for nd in no_decay)], 'weight_decay': args['weight_decay']},
            {'params': [p for n, p in model.named_parameters() if any(
                nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

        warmup_steps = math.ceil(t_total * args['warmup_ratio'])
        args['warmup_steps'] = warmup_steps if args['warmup_steps'] == 0 else args['warmup_steps']

        optimizer = AdamW(optimizer_grouped_parameters, lr=args['learning_rate'], eps=args['adam_epsilon'])
        scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args['warmup_steps'], t_total=t_total)

        if args['fp16']:
            try:
                from apex import amp
            except ImportError:
                raise ImportError(
                    "Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
            model, optimizer = amp.initialize(model, optimizer, opt_level=args['fp16_opt_level'])

        global_step = 0
        tr_loss, logging_loss = 0.0, 0.0
        model.zero_grad()
        train_iterator = trange(int(args['num_train_epochs']), desc="Epoch")
        
        for _ in train_iterator:
            epoch_iterator = tqdm(train_dataloader, desc="Iteration")
            for step, batch in enumerate(epoch_iterator):
                model.train()
                batch = tuple(t.to(device) for t in batch)
                inputs = {'input_ids':      batch[0],
                        'attention_mask': batch[1],
                        # XLM don't use segment_ids
                        'token_type_ids': batch[2] if args['model_type'] in ['bert', 'xlnet'] else None,
                        'labels':         batch[3]}
                outputs = model(**inputs)
                # model outputs are always tuple in pytorch-transformers (see doc)
                loss = outputs[0]
                print("\rRunning loss: %f" % loss, end='')

                if args['gradient_accumulation_steps'] > 1:
                    loss = loss / args['gradient_accumulation_steps']

                if args['fp16']:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args['max_grad_norm'])

                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args['max_grad_norm'])

                tr_loss += loss.item()
                if (step + 1) % args['gradient_accumulation_steps'] == 0:
                    optimizer.step()
                    scheduler.step()  # Update learning rate schedule
                    model.zero_grad()
                    global_step += 1

                    if args['logging_steps'] > 0 and global_step % args['logging_steps'] == 0:
                        # Log metrics
                        # Only evaluate when single GPU otherwise metrics may not average well
                        if args['evaluate_during_training']:
                            results = evaluate(model, tokenizer)
                            for key, value in results.items():
                                tb_writer.add_scalar('eval_{}'.format(key), value, global_step)
                        tb_writer.add_scalar('lr', scheduler.get_lr()[0], global_step)
                        tb_writer.add_scalar('loss', (tr_loss - logging_loss)/args['logging_steps'], global_step)
                        logging_loss = tr_loss

                    if args['save_steps'] > 0 and global_step % args['save_steps'] == 0:
                        # Save model checkpoint
                        output_dir = os.path.join(
                            output_dir, 'checkpoint-{}'.format(global_step))
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)
                        # Take care of distributed/parallel training
                        model_to_save = model.module if hasattr(
                            model, 'module') else model
                        model_to_save.save_pretrained(output_dir)
        return global_step, tr_loss / global_step


    def get_mismatched(self, labels, preds):
        mismatched = labels != preds
        examples = BinaryProcessor().get_dev_examples(self.args['data_dir'])
        wrong = [i for (i, v) in zip(examples, mismatched) if v]

        return wrong


    def get_eval_report(self, labels, preds):
        mcc = matthews_corrcoef(labels, preds)
        tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()
        return {
            "mcc": mcc,
            "tp": tp,
            "tn": tn,
            "fp": fp,
            "fn": fn
        }, self.get_mismatched(labels, preds)


    def compute_metrics(self, task_name, preds, labels):
        assert len(preds) == len(labels)
        return self.get_eval_report(labels, preds)
