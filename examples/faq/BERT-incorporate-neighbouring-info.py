import simpletransformers
from simpletransformers.classification import ClassificationModelAddFeatures
from simpletransformers.metrics.ranking_metrics import MRR, MAP, NDCG, Prec, Recall, F1
from simpletransformers.data.data_utils import load_url_vocab
from simpletransformers.data.data_utils import load_url_data_with_neighbouring_info
import numpy as np
import pandas as pd
import sklearn
import os

train_data = \
    [["I'm Example sentence 1 for multilabel classification.", "Article number one", "1", 1, [0,1,0]],
     ["I'm Example sentence 1 for multilabel classification.", "Article number two", "1", 0, [0,0,1]],
     ["I'm Example sentence 1 for multilabel classification.", "Article number three", "1", 0, [1,0,0]],
     ["This is another example sentence. ", "Article number one", "2", 0, [0,1,0]],
     ["This is another example sentence. ", "Article number two", "2", 1, [0,0,1]],
     ["This is another example sentence. ", "Article number three", "2", 0, [1,0,0]]]
df_train = pd.DataFrame(train_data, columns=['text_a', 'text_b', 'reps_response_id', 'labels', 'addfeatures'])

eval_data = \
    [["Example eval sentence for multilabel classification.", "Article number one", "3", 1, [0,1,0]],
     ["Example eval sentence for multilabel classification.", "Article number two", "3", 0, [0,0,1]],
     ["Example eval sentence for multilabel classification.", "Article number three", "3", 0, [1,0,0]],
     ["Example eval senntence belonging to class 2", "Article number one", "4", 0, [0,1,0]],
     ["Example eval senntence belonging to class 2", "Article number two", "4", 1, [0,0,1]],
     ["Example eval senntence belonging to class 2", "Article number three", "4", 0, [1,0,0]]]
eval_df = pd.DataFrame(eval_data, columns=['text_a', 'text_b', 'reps_response_id', 'labels', 'addfeatures'])
df_dev = eval_df
df_test = eval_df

# # Load URL vocab from S3
# file_path = os.path.join('jie-faq', 'faq', 'outreach_support_url_meta.json')
# urlvocab = load_url_vocab(file_path)
# print('Number of recommendation candidate URLs: %d' % urlvocab.vocab_size)
#
# # Load dataset for training and testing from S3.
# datafolder = os.path.join('jie-faq', 'faq', 'outreach_msgs_w_support_url_random_split')
# df_train, df_dev, df_test = load_url_data_with_neighbouring_info(datafolder, urlvocab)
# print(f'Dataset size: train={len(df_train)}/{urlvocab.vocab_size}={len(df_train) / urlvocab.vocab_size}, ' +
#       f'dev={len(df_dev)}/{urlvocab.vocab_size}={len(df_dev) / urlvocab.vocab_size}, ' +
#       f'test={len(df_test)}/{urlvocab.vocab_size}={len(df_test) / urlvocab.vocab_size}')

train_args = {'learning_rate': 5e-5,
                  'reprocess_input_data': True,
                  'overwrite_output_dir': True,
                  'overwrite_cache': True,
                  'num_train_epochs': 10,
                  'n_gpu': 1,
                  'max_seq_length': (128, 128),
                  'do_lower_case': True,
                  'output_dir': './tmp/BERT_sent_pair/outputs',
                  'cache_dir': './tmp/BERT_sent_pair/cache',
                  'train_batch_size': 32,
                  'eval_batch_size': 32,
                  'fp16': False,
                  'use_multiprocessing': False,
                  'silent': True,
                  'faq_evaluate_during_training': True,
                  'additional_features_size': 3,
                  }
model = ClassificationModelAddFeatures(
    'bert', 'bert-base-uncased', num_labels=2, use_cuda=False, args=train_args)
model.train_model(df_train, eval_df=df_dev, test_df=df_test)