import logging
import os
import pickle
from multiprocessing import Pool
from functools import partial

import torch
import transformers
import numpy as np

from torch.utils.data import Dataset
from tqdm.auto import tqdm

from datasets import Features, Sequence, Value, load_dataset, concatenate_datasets
from datasets import Dataset as HFDataset

from transformers.models.rag.retrieval_rag import Index


logger = logging.getLogger(__name__)


def load_hf_dataset(data, context_tokenizer, query_tokenizer, args):
    if isinstance(data, str):
        if data.endswith(".json"):
            dataset = load_dataset(
                os.path.join(
                    os.path.dirname(os.path.abspath(__file__)), "retrieval_dataset_loading_script"
                ),
                data_files=data,
                hard_negatives=args.hard_negatives,
                download_mode="force_redownload"
                if args.reprocess_input_data
                else "reuse_dataset_if_exists",
            )
        else:
            dataset = load_dataset(
                "csv",
                data_files=data,
                delimiter="\t",
                download_mode="force_redownload"
                if args.reprocess_input_data
                else "reuse_dataset_if_exists",
                cache_dir=args.dataset_cache_dir,
            )
    else:
        dataset = HFDataset.from_pandas(data)

    dataset = dataset.map(
        lambda x: preprocess_batch_for_hf_dataset(
            x,
            context_tokenizer=context_tokenizer,
            query_tokenizer=query_tokenizer,
            args=args,
        ),
        batched=True,
    )

    if args.hard_negatives:
        column_names = [
            "context_ids",
            "query_ids",
            "hard_negatives_ids",
            "context_mask",
            "query_mask",
            "hard_negatives_mask",
        ]
    else:
        column_names = [
            "context_ids",
            "query_ids",
            "context_mask",
            "query_mask",
        ]

    dataset.set_format(type="pt", columns=column_names)

    if isinstance(data, str):
        # This is not necessarily a train dataset. The datasets library insists on calling it train.
        return dataset["train"]
    else:
        return dataset


def preprocess_batch_for_hf_dataset(dataset, context_tokenizer, query_tokenizer, args):
    context_inputs = context_tokenizer(
        dataset["gold_passage"],
        max_length=args.max_seq_length,
        padding="max_length",
        return_tensors="np",
        truncation=True,
    )

    query_inputs = query_tokenizer(
        dataset["query_text"],
        max_length=args.max_seq_length,
        padding="max_length",
        return_tensors="np",
        truncation=True,
    )

    context_ids = context_inputs["input_ids"].squeeze()
    query_ids = query_inputs["input_ids"].squeeze()
    context_mask = context_inputs["attention_mask"].squeeze()
    query_mask = query_inputs["attention_mask"].squeeze()

    if args.hard_negatives:
        hard_negatives_inputs = query_tokenizer(
            dataset["hard_negatives"],
            max_length=args.max_seq_length,
            padding="max_length",
            return_tensors="np",
            truncation=True,
        )
        hard_negatives_ids = hard_negatives_inputs["input_ids"].squeeze()
        hard_negatives_mask = hard_negatives_inputs["attention_mask"].squeeze()

        return {
            "context_ids": context_ids,
            "query_ids": query_ids,
            "hard_negatives_ids": hard_negatives_ids,
            "context_mask": context_mask,
            "query_mask": query_mask,
            "hard_negatives_mask": hard_negatives_mask,
        }

    return {
        "context_ids": context_ids,
        "query_ids": query_ids,
        "context_mask": context_mask,
        "query_mask": query_mask,
    }


def embed(documents, encoder, tokenizer, device):
    """Compute the DPR embeddings of document passages"""
    input_ids = tokenizer(
        documents["passages"],
        truncation=True,
        padding="longest",
        return_tensors="pt",
    )["input_ids"]
    embeddings = encoder(input_ids.to(device=device), return_dict=True).pooler_output
    return {"embeddings": embeddings.detach().cpu().numpy()}


def get_evaluation_passage_dataset(
    eval_data, additional_passages, encoder, tokenizer, context_config, args, device
):
    import faiss

    logger.info("Loading evaluation passages to a Huggingface Dataset")
    if isinstance(eval_data, str):
        if eval_data.endswith(".json"):
            passage_dataset = load_dataset(
                os.path.join(
                    os.path.dirname(os.path.abspath(__file__)), "retrieval_dataset_loading_script"
                ),
                data_files=eval_data,
                hard_negatives=False,
                download_mode="force_redownload"
                if args.reprocess_input_data
                else "reuse_dataset_if_exists",
            )
        else:
            passage_dataset = load_dataset(
                "csv",
                data_files=eval_data,
                delimiter="\t",
                download_mode="force_redownload"
                if args.reprocess_input_data
                else "reuse_dataset_if_exists",
                cache_dir=args.dataset_cache_dir,
            )
        passage_dataset = passage_dataset["train"]
    else:
        passage_dataset = HFDataset.from_pandas(eval_data)
        passage_dataset = passage_dataset.remove_columns("query_text")
    passage_dataset = passage_dataset.rename_column("gold_passage", "passages")

    if additional_passages is not None:
        if isinstance(additional_passages, str):
            additional_passages = load_dataset(
                "csv",
                data_files=additional_passages,
                delimiter="\t",
                column_names=["passages"],
                cache_dir=args.dataset_cache_dir,
            )
            additional_passages = additional_passages["train"]
        elif isinstance(additional_passages, list):
            additional_passages = HFDataset.from_dict({"passages": additional_passages})
        else:
            additional_passages = HFDataset.from_pandas(additional_passages)

        passage_dataset = concatenate_datasets([passage_dataset, additional_passages])
    logger.info("Loading evaluation passages to a Huggingface Dataset completed.")

    logger.info("Generating embeddings for evaluation passages")
    passage_dataset = passage_dataset.map(
        partial(embed, encoder=encoder, tokenizer=tokenizer, device=device),
        batched=True,
        batch_size=args.embed_batch_size,
    )

    logger.info("Generating embeddings for evaluation passages completed.")
    if args.save_passage_dataset:
        output_dataset_directory = os.path.join(args.output_dir, "passage_dataset")
        os.makedirs(output_dataset_directory, exist_ok=True)
        passage_dataset.save_to_disk(output_dataset_directory)

    logger.info("Adding FAISS index to evaluation passages")
    index = faiss.IndexHNSWFlat(args.faiss_d, args.faiss_m, faiss.METRIC_INNER_PRODUCT)
    passage_dataset.add_faiss_index("embeddings", custom_index=index)
    passage_index = DPRIndex(passage_dataset, context_config.hidden_size)
    logger.info("Adding FAISS index to evaluation passages completed.")
    if args.save_passage_dataset:
        faiss_save_path = os.path.join(output_dataset_directory, "index.faiss")
        passage_dataset.save_faiss_index("embeddings", faiss_save_path)

    return passage_index


class DPRIndex(Index):
    def __init__(self, dataset, vector_size):
        self.dataset = dataset
        self.vector_size = vector_size

    def get_doc_dicts(self, doc_ids):
        return [self.dataset[doc_ids[i].tolist()] for i in range(doc_ids.shape[0])]

    def get_top_docs(self, question_hidden_states, n_docs=5):
        _, ids = self.dataset.search_batch("embeddings", question_hidden_states, n_docs)
        docs = [self.dataset[[i for i in indices if i >= 0]] for indices in ids]
        vectors = [doc["embeddings"] for doc in docs]
        for i in range(len(vectors)):
            if len(vectors[i]) < n_docs:
                vectors[i] = np.vstack(
                    [vectors[i], np.zeros((n_docs - len(vectors[i]), self.vector_size))]
                )
        return np.array(ids), np.array(
            vectors
        )  # shapes (batch_size, n_docs) and (batch_size, n_docs, d)


def mean_reciprocal_rank_at_k(rs, k):
    """
    Adapted from https://gist.github.com/bwhite/3726239

    Score is reciprocal of the rank of the first relevant item
    First element is 'rank 1'.  Relevance is binary (nonzero is relevant).
    Example from http://en.wikipedia.org/wiki/Mean_reciprocal_rank

    Args:
        rs: Iterator of relevance scores (list or numpy) in rank order
            (first element is the first item)
    Returns:
        Mean reciprocal rank
    """
    rs = rs[:, :k]
    rs = (np.asarray(r).nonzero()[0] for r in rs)
    return np.mean([1. / (r[0] + 1) if r.size else 0. for r in rs])