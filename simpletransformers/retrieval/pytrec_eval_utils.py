import logging


logger = logging.getLogger(__name__)


def convert_predictions_to_pytrec_format(predicted_doc_ids, query_dataset, id_column="_id"):
    run_dict = {}
    for query_id, doc_ids in zip(query_dataset[id_column], predicted_doc_ids):
        run_dict[query_id] = {doc_id: 1 / (i + 1) for i, doc_id in enumerate(doc_ids)}

    return run_dict


def convert_qrels_dataset_to_pytrec_format(qrels_dataset):
    qrels_dict = {}
    for query_id, doc_id, relevance in zip(
        qrels_dataset["query_id"],
        qrels_dataset["passage_id"],
        qrels_dataset["relevance"],
    ):
        query_id = str(query_id)
        doc_id = str(doc_id)
        if query_id not in qrels_dict:
            qrels_dict[query_id] = {}
        qrels_dict[query_id][doc_id] = relevance

    return qrels_dict


def convert_metric_dict_to_scores_list(metric_dict):
    scores_list = [score for metric, score in metric_dict.items()]
    return scores_list
