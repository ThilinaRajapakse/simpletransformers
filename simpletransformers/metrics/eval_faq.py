from simpletransformers.metrics.ranking_metrics import MAP, MRR, NDCG, Prec, Recall, F1
import numpy as np
import sys


def print_metrics(metrics, f=sys.stdout):
  print('MRR  = %.4f, MAP  = %.4f, NDCG  = %.4f' % (metrics['MRR'], metrics['MAP'], metrics['NDCG']), file=f)
  print('P@5  = %.4f, R@5  = %.4f, F1@5  = %.4f' % (metrics['P@5'], metrics['R@5'], metrics['F1@5']), file=f)
  print('P@10 = %.4f, R@10 = %.4f, F1@10 = %.4f' % (metrics['P@10'], metrics['R@10'], metrics['F1@10']), file=f)


def faq_evaluate(model, df_eval_input):
    if 'text' in df_eval_input.columns:
        # This is evaluated as a 'classification' task
        predictions, raw_outputs = model.predict(df_eval_input['text'])
        labels = np.array(list(df_eval_input['labels']))
        # predictions = np.array(predictions)
    elif 'text_a' in df_eval_input.columns and 'text_b' in df_eval_input.columns:
        # This is evaluated as a 'sentence pair' matching task
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
            # predictions.append(list(df['predictions']))
            raw_outputs.append(list(df['scores']))
        # predictions = np.array(predictions)
        raw_outputs = np.array(raw_outputs)
        labels = np.array(labels)
    else:
        raise ValueError(f'FAQ evaluation mode not defined.')

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