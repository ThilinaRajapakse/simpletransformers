import csv
import os
import platform

PLATFORM = platform.system()
if PLATFORM != 'Darwin':
    from shutil import copyfile


def write_progress_to_csv(path, filename, write_header=False, metrics=None):
    if PLATFORM == 'Darwin':
        filename_ = os.path.join(path, filename)
    else:
        # For Databricks, DBFS does not support random file write.
        # See https://docs.databricks.com/data/databricks-file-system.html#local-file-api-limitations for details.
        path_ = os.path.join('/local_disk0/tmp', '.' + path)
        if not os.path.exists(path_):
            os.makedirs(path_)  # This requires sudo access
        filename_ = os.path.join(path_, filename)

    with open(filename_, 'a', newline='') as outcsv:
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

    if PLATFORM != 'Darwin':
        # For Databricks, copy file to DBFS
        copyfile(filename_, os.path.join(path, filename))
