import logging


logger = logging.getLogger(__name__)


def convert_predictions_to_pytrec_format(
    predicted_doc_ids, query_dataset, id_column="_id", predicted_scores=None
):
    run_dict = {}
    if predicted_scores is None:
        logger.warning(
            "No scores provided. Using 1 / (rank + 1) for all documents in the run file."
        )
        for query_id, doc_ids in zip(query_dataset[id_column], predicted_doc_ids):
            # run_dict[query_id] = {doc_id: 1 / (i + 1) for i, doc_id in enumerate(doc_ids)}
            # This doesn't work when there are duplicate doc_ids

            run_dict[query_id] = {}
            for i, doc_id in enumerate(doc_ids):
                if doc_id not in run_dict[query_id]:
                    run_dict[query_id][doc_id] = 1 / (i + 1)
                else:
                    logger.warning(
                        f"Duplicate doc_id {doc_id} for query_id {query_id} at position {i}"
                    )
    else:
        for query_id, doc_ids, scores in zip(
            query_dataset[id_column], predicted_doc_ids, predicted_scores
        ):
            run_dict[query_id] = {}
            for doc_id, score in zip(doc_ids, scores):
                if doc_id not in run_dict[query_id]:
                    run_dict[query_id][doc_id] = float(score)
                else:
                    logger.warning(
                        f"Duplicate doc_id {doc_id} for query_id {query_id} with score {score}"
                    )

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
