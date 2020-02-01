import os
import csv


def write_progress_to_csv(filename, write_header=False, metrics=None):
    with open(filename, 'a', newline='') as outcsv:
        writer = csv.DictWriter(outcsv, fieldnames=['epoch', 'ckpt',
                                                    'dev-MRR', 'dev-MAP', 'dev-NDCG',
                                                    'dev-P@5', 'dev-R@5', 'dev-F1@5',
                                                    'dev-P@10', 'dev-R@10', 'dev-F1@10',
                                                    'test-MRR', 'test-MAP', 'test-NDCG',
                                                    'test-P@5', 'test-R@5', 'test-F1@5',
                                                    'test-P@10', 'test-R@10', 'test-F1@10'])
        if write_header:
            writer.writeheader()

        if metrics is not None:
            writer.writerow(metrics)
