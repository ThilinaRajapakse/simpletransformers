import simpletransformers
from simpletransformers.classification import SemanticMatchingClassificationModel
from simpletransformers.metrics.ranking_metrics import MRR, MAP, NDCG, Prec, Recall, F1
from simpletransformers.data.data_utils import load_url_vocab
from simpletransformers.data.data_utils import load_url_data_email_article_pair
import numpy as np
import pandas as pd
import sklearn
import os


def evaluate(model, df_eval_input):
    predictions, raw_outputs = model.predict(list(map(list, zip(df_eval_input['text_a'], df_eval_input['text_b']))))
    df_eval = df_eval_input.copy()
    df_eval['predictions'] = predictions
    df_eval['scores'] = raw_outputs[:, 1]

    # Transfering the results back from pairwise binary classification to ranking
    labels, predictions, raw_outputs = [], [], []
    for _, df in df_eval.groupby(by='reps_response_id'):
        # All lines should share the same "url_labels"
        assert all([x == df.iloc[0]['url_labels'] for x in df['url_labels']])
        # "url_labels" should be recovered from individual "labels"
        assert set(list(df[df['labels'] == 1]['url'])) == set(df.iloc[0]['url_labels'])
        labels.append(list(df['labels']))
        predictions.append(list(df['predictions']))
        raw_outputs.append(list(df['scores']))
    predictions = np.array(predictions)
    raw_outputs = np.array(raw_outputs)
    labels = np.array(labels)

    # Evaluation metrics
    pos_idx = [_.nonzero()[0].tolist() for _ in labels]
    rank_idx = np.flip(np.argsort(raw_outputs, axis=1), axis=1).tolist()

    mrrs = [MRR(x, y) for x, y in zip(pos_idx, rank_idx)]
    maps = [MAP(x, y) for x, y in zip(pos_idx, rank_idx)]
    ndcgs = [NDCG(x, y) for x, y in zip(pos_idx, rank_idx)]
    p_5 = [Prec(x, y, 5) for x, y in zip(pos_idx, rank_idx)]
    r_5 = [Recall(x, y, 5) for x, y in zip(pos_idx, rank_idx)]
    f1_5 = [F1(x, y, 5) for x, y in zip(pos_idx, rank_idx)]
    p_10 = [Prec(x, y, 10) for x, y in zip(pos_idx, rank_idx)]
    r_10 = [Recall(x, y, 10) for x, y in zip(pos_idx, rank_idx)]
    f1_10 = [F1(x, y, 10) for x, y in zip(pos_idx, rank_idx)]
    metrics = {
        'MRR': np.mean(mrrs),
        'MAP': np.mean(maps),
        'NDCG': np.mean(ndcgs),
        'P@5': np.mean(p_5),
        'R@5': np.mean(r_5),
        'F1@5': np.mean(f1_5),
        'P@10': np.mean(p_10),
        'R@10': np.mean(r_10),
        'F1@10': np.mean(f1_10),
    }
    return metrics, raw_outputs, rank_idx


def print_metrics(metrics):
  print('MRR  = %.4f, MAP  = %.4f, NDCG  = %.4f' % (metrics['MRR'], metrics['MAP'], metrics['NDCG']))
  print('P@5  = %.4f, R@5  = %.4f, F1@5  = %.4f' % (metrics['P@5'], metrics['R@5'], metrics['F1@5']))
  print('P@10 = %.4f, R@10 = %.4f, F1@10 = %.4f' % (metrics['P@10'], metrics['R@10'], metrics['F1@10']))


def train(train_df, dev_df, test_df, train_args, use_cuda=True):
    if not os.path.exists(train_args['cache_dir']):
        os.makedirs(train_args['cache_dir'])

    # Create a ClassificationModel
    model = SemanticMatchingClassificationModel(
        'bert', 'bert-base-uncased', num_labels=2, use_cuda=use_cuda, args=train_args)

    # Train the model
    model.train_model(train_df)
    return model


if __name__ == '__main__':

    # Load URL vocab from S3
    file_path = os.path.join('jie-faq', 'faq', 'outreach_support_url_meta.json')
    urlvocab = load_url_vocab(file_path)
    print('Number of recommendation candidate URLs: %d' % urlvocab.vocab_size)

    # Load dataset for training and testing from S3.
    datafolder = os.path.join('jie-faq', 'faq', 'outreach_msgs_w_support_url_random_split')
    df_train, df_dev, df_test = load_url_data_email_article_pair(datafolder, urlvocab)
    print(f'Dataset size: train={len(df_train)}/{urlvocab.vocab_size}={len(df_train) / urlvocab.vocab_size}, ' +
          f'dev={len(df_dev)}/{urlvocab.vocab_size}={len(df_dev) / urlvocab.vocab_size}, ' +
          f'test={len(df_test)}/{urlvocab.vocab_size}={len(df_test) / urlvocab.vocab_size}')

    # Truncate dataset
    df_train = df_train[: urlvocab.vocab_size * 1]
    df_dev = df_dev[: urlvocab.vocab_size * 1]
    df_test = df_test[: urlvocab.vocab_size * 1]

    # train_args = {'learning_rate': 5e-5,
    #               'reprocess_input_data': True,
    #               'overwrite_output_dir': True,
    #               'overwrite_cache': True,
    #               'num_train_epochs': 10,
    #               'n_gpu': 1,
    #               'max_seq_length': (128, 128),
    #               'do_lower_case': True,
    #               'output_dir': './tmp/BERT_sent_pair/outputs',
    #               'cache_dir': './tmp/BERT_sent_pair/cache',
    #               'train_batch_size': 32,
    #               'fp16': False,
    #               'use_multiprocessing': False
    #               }
    # model = train(df_train, df_dev, df_test, train_args, use_cuda=False)
    #
    # # Evaluate
    # dev_metrics, dev_rank_scores, dev_rank_idxes = evaluate(model, df_dev)
    # print_metrics(dev_metrics)

    # Load trained model
    ckpt = 'checkpoint-9-epoch-1'
    print(f'Evaluating based on {ckpt}')
    eval_args = {'learning_rate': 5e-5,
                 'reprocess_input_data': True,
                 'overwrite_output_dir': True,
                 'overwrite_cache': True,
                 'num_train_epochs': 10,
                 'n_gpu': 1,
                 'max_seq_length': (128, 128),
                 'do_lower_case': True,
                 'train_batch_size': 32,
                 'fp16': False,
                 }
    model = SemanticMatchingClassificationModel(
        'bert',
        f'/Users/jiezhao/Documents/GitHub/simpletransformers/tmp/BERT_sent_pair/outputs/{ckpt}',
        num_labels=2, use_cuda=False, args=eval_args)

    # Evaluate
    dev_metrics, dev_rank_scores, dev_rank_idxes = evaluate(model, df_train)
    print_metrics(dev_metrics)