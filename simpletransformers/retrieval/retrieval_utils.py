import logging
import os
import pickle
from multiprocessing import Pool, cpu_count
from functools import partial
from simpletransformers.seq2seq.seq2seq_utils import add_faiss_index_to_dataset
import datasets
from datasets.load import load_from_disk

import torch
import transformers
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.model_selection import train_test_split

from torch.utils.data import Dataset
from tqdm.auto import tqdm

from datasets import Features, Sequence, Value, load_dataset, concatenate_datasets
from datasets import Dataset as HFDataset

from transformers.models.rag.retrieval_rag import Index


logger = logging.getLogger(__name__)


def load_hf_dataset(data, context_tokenizer, query_tokenizer, args, evaluate=False):
    if isinstance(data, str):
        if data.endswith(".json"):
            dataset = load_dataset(
                # os.path.join(
                #     os.path.dirname(os.path.abspath(__file__)),
                "retrieval_dataset_loading_script",
                # ),
                data_files=data,
                hard_negatives=args.hard_negatives,
                include_title=args.include_title,
                download_mode="force_redownload"
                if args.reprocess_input_data
                else "reuse_dataset_if_exists",
            )
            dataset = dataset["train"]
        # If data is a directory, then it should be a HF dataset
        elif os.path.isdir(data):
            dataset = load_from_disk(data)
            try:
                dataset = dataset["train"]
            except KeyError:
                pass
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
            dataset = dataset["train"]
            if args.include_title:
                if "title" not in dataset.column_names:
                    raise ValueError(
                        "The dataset must contain a column named 'title' if args.include_title is True."
                    )
                dataset = dataset.map(
                    lambda example: {
                        "gold_passage": example["title"] + " " + example["gold_passage"]
                    }
                )
    else:
        dataset = HFDataset.from_pandas(data)
        if args.include_title:
            if "title" not in dataset.column_names:
                raise ValueError(
                    "The dataset must contain a column named 'title' if args.include_title is True."
                )
            dataset = dataset.map(
                lambda example: {
                    "gold_passage": example["title"] + " " + example["gold_passage"]
                }
            )

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
            "hard_negative_ids",
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

    if evaluate:
        gold_passages = dataset["gold_passage"]
        dataset.set_format(type="pt", columns=column_names)

        return dataset, gold_passages
    else:
        dataset.set_format(type="pt", columns=column_names)
        return dataset


def preprocess_batch_for_hf_dataset(dataset, context_tokenizer, query_tokenizer, args):
    try:
        context_inputs = context_tokenizer(
            dataset["gold_passage"],
            max_length=args.max_seq_length,
            padding="max_length",
            return_tensors="np",
            truncation=True,
        )
    except (TypeError, ValueError) as e:
        logger.warn(e)
        logger.warn(
            """Error encountered while converting target_text.
        All target_text values have been manually cast to String as a workaround.
        This may have been caused by NaN values present in the data."""
        )
        dataset["gold_passage"] = [str(p) for p in dataset["gold_passage"]]
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
        try:
            hard_negatives_inputs = context_tokenizer(
                dataset["hard_negative"],
                max_length=args.max_seq_length,
                padding="max_length",
                return_tensors="np",
                truncation=True,
            )
        except (TypeError, ValueError) as e:
            logger.warn(e)
            logger.warn(
                """Error encountered while converting target_text.
            All target_text values have been manually cast to String as a workaround.
            This may have been caused by NaN values present in the data."""
            )
            dataset["hard_negative"] = [str(p) for p in dataset["hard_negative"]]
            hard_negatives_inputs = context_tokenizer(
                dataset["hard_negative"],
                max_length=args.max_seq_length,
                padding="max_length",
                return_tensors="np",
                truncation=True,
            )
        hard_negative_ids = hard_negatives_inputs["input_ids"].squeeze()
        hard_negatives_mask = hard_negatives_inputs["attention_mask"].squeeze()

        return {
            "context_ids": context_ids,
            "query_ids": query_ids,
            "hard_negative_ids": hard_negative_ids,
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


def get_output_embeddings(embeddings, concatenate_embeddings=False, n_cls_tokens=3):
    """
    Extracts the embeddings from the output of the model.
    Concatenates CLS embeddings if concatenate_embeddings is True.
    """
    if concatenate_embeddings:
        return embeddings.last_hidden_state[:, :n_cls_tokens, :].reshape(
            embeddings.last_hidden_state.shape[0], -1
        )
    else:
        return embeddings.pooler_output


def embed(
    documents,
    rank=None,
    encoder=None,
    tokenizer=None,
    concatenate_embeddings=False,
    extra_cls_token_count=0,
    device=None,
    fp16=None,
    amp=None,
    pretokenized=False,
):
    """Compute the DPR embeddings of document passages"""
    if rank is not None:
        device = torch.device("cuda", rank)
        encoder = encoder.to(device)
    with torch.no_grad():
        if fp16:
            with amp.autocast():
                if not pretokenized:
                    try:
                        input_ids = tokenizer(
                            documents["passages"],
                            truncation=True,
                            padding="longest",
                            return_tensors="pt",
                        )
                        embeddings = encoder(
                            input_ids["input_ids"].to(device=device),
                            input_ids["attention_mask"].to(device=device),
                            return_dict=True,
                        )
                        embeddings = get_output_embeddings(
                            embeddings,
                            concatenate_embeddings=concatenate_embeddings,
                            n_cls_tokens=(1 + extra_cls_token_count),
                        )
                    except (TypeError, ValueError) as e:
                        logger.warn(e)
                        logger.warn(
                            """Error encountered while converting target_text.
                        All target_text values have been manually cast to String as a workaround.
                        This may have been caused by NaN values present in the data."""
                        )
                        documents["passages"] = [str(p) for p in documents["passages"]]
                        input_ids = tokenizer(
                            documents["passages"],
                            truncation=True,
                            padding="longest",
                            return_tensors="pt",
                        )
                        embeddings = encoder(
                            input_ids["input_ids"].to(device=device),
                            input_ids["attention_mask"].to(device=device),
                            return_dict=True,
                        )
                        embeddings = get_output_embeddings(
                            embeddings,
                            concatenate_embeddings=concatenate_embeddings,
                            n_cls_tokens=(1 + extra_cls_token_count),
                        )
                else:
                    embeddings = encoder(
                        documents["context_ids"].to(device=device),
                        documents["context_mask"].to(device=device),
                        return_dict=True,
                    )
                    embeddings = get_output_embeddings(
                        embeddings,
                        concatenate_embeddings=concatenate_embeddings,
                        n_cls_tokens=(1 + extra_cls_token_count),
                    )
            # Embeddings need to be float32 for indexing
            embeddings = embeddings.float()
        else:
            if not pretokenized:
                try:
                    input_ids = tokenizer(
                        documents["passages"],
                        truncation=True,
                        padding="longest",
                        return_tensors="pt",
                    )
                    embeddings = encoder(
                        input_ids["input_ids"].to(device=device),
                        input_ids["attention_mask"].to(device=device),
                        return_dict=True,
                    )
                    embeddings = get_output_embeddings(
                        embeddings,
                        concatenate_embeddings=concatenate_embeddings,
                        n_cls_tokens=(1 + extra_cls_token_count),
                    )
                except (TypeError, ValueError) as e:
                    logger.warn(e)
                    logger.warn(
                        """Error encountered while converting target_text.
                    All target_text values have been manually cast to String as a workaround.
                    This may have been caused by NaN values present in the data."""
                    )
                    documents["passages"] = [str(p) for p in documents["passages"]]
                    input_ids = tokenizer(
                        documents["passages"],
                        truncation=True,
                        padding="longest",
                        return_tensors="pt",
                    )
                    embeddings = encoder(
                        input_ids["input_ids"].to(device=device),
                        input_ids["attention_mask"].to(device=device),
                        return_dict=True,
                    )
                    embeddings = get_output_embeddings(
                        embeddings,
                        concatenate_embeddings=concatenate_embeddings,
                        n_cls_tokens=(1 + extra_cls_token_count),
                    )
            else:
                embeddings = encoder(
                    documents["context_ids"].to(device=device),
                    documents["context_mask"].to(device=device),
                    return_dict=True,
                )
                embeddings = get_output_embeddings(
                    embeddings,
                    concatenate_embeddings=concatenate_embeddings,
                    n_cls_tokens=(1 + extra_cls_token_count),
                )
    return {"embeddings": embeddings.detach().cpu().numpy()}


def add_hard_negatives_to_evaluation_dataset(dataset):
    return {"passages": [passage for passage in dataset["hard_negative"]]}


def get_evaluation_passage_dataset(
    eval_data,
    additional_passages,
    encoder,
    tokenizer,
    context_config,
    args,
    device,
    passage_dataset=None,
):
    import faiss

    if not passage_dataset:
        logger.info("Loading evaluation passages to a Huggingface Dataset")
        if isinstance(eval_data, str):
            if eval_data.endswith(".json"):
                passage_dataset = load_dataset(
                    "retrieval_dataset_loading_script",
                    data_files=eval_data,
                    hard_negatives=args.hard_negatives,
                    include_title=args.include_title_in_knowledge_dataset,
                    download_mode="force_redownload"
                    if args.reprocess_input_data
                    else "reuse_dataset_if_exists",
                )
                passage_dataset = passage_dataset["train"]
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
                if args.include_title_in_knowledge_dataset:
                    if "title" not in passage_dataset.column_names:
                        raise ValueError(
                            "The dataset must contain a column named 'title' if args.include_title_in_knowledge_dataset is True."
                        )
                    passage_dataset = passage_dataset.map(
                        lambda example: {
                            "gold_passage": example["title"]
                            + " "
                            + example["gold_passage"]
                        }
                    )

        else:
            passage_dataset = HFDataset.from_pandas(eval_data)
            if args.include_title_in_knowledge_dataset:
                if "title" not in passage_dataset.column_names:
                    raise ValueError(
                        "The dataset must contain a column named 'title' if args.include_title_in_knowledge_dataset is True."
                    )
                passage_dataset = passage_dataset.map(
                    lambda example: {
                        "gold_passage": example["title"] + " " + example["gold_passage"]
                    }
                )

        try:
            passage_dataset = passage_dataset.remove_columns("query_text")
        except ValueError:
            # It's fine, query_text is not here
            pass

        passage_dataset = passage_dataset.rename_column("gold_passage", "passages")

        if args.hard_negatives:
            passage_dataset = passage_dataset.map(
                add_hard_negatives_to_evaluation_dataset,
                batched=True,
                remove_columns=["hard_negative"],
            )

        # try:
        #     if additional_passages is not None:
        #         passage_dataset = passage_dataset.remove_columns("hard_negative")
        # except ValueError:
        #     pass

        if additional_passages is not None:
            if args.fp16:
                from torch.cuda import amp
            else:
                amp = None
            if isinstance(additional_passages, str):
                if os.path.isdir(additional_passages):
                    # To be used if you want to reuse the embeddings from a previous eval but
                    # with new eval data
                    additional_passages = load_from_disk(additional_passages)
                    encoder = encoder.to(device)
                    passage_dataset = passage_dataset.map(
                        partial(
                            embed,
                            encoder=encoder,
                            tokenizer=tokenizer,
                            concatenate_embeddings=args.larger_representations,
                            extra_cls_token_count=args.extra_cls_token_count,
                            device=device,
                            fp16=args.fp16,
                            amp=amp,
                        ),
                        batched=True,
                        batch_size=args.embed_batch_size,
                    )
                else:
                    additional_passages = load_dataset(
                        "csv",
                        data_files=additional_passages,
                        delimiter="\t",
                        column_names=["passages"],
                        cache_dir=args.dataset_cache_dir,
                    )
                    if args.include_title_in_knowledge_dataset:
                        if "title" not in passage_dataset.column_names:
                            raise ValueError(
                                "The dataset must contain a column named 'title' if args.include_title_in_knowledge_dataset is True."
                            )
                        passage_dataset = passage_dataset.map(
                            lambda example: {
                                "gold_passage": example["title"]
                                + " "
                                + example["gold_passage"]
                            }
                        )
                    additional_passages = additional_passages["train"]
            elif isinstance(additional_passages, list):
                additional_passages = HFDataset.from_dict(
                    {"passages": additional_passages}
                )
            else:
                additional_passages = HFDataset.from_pandas(additional_passages)
                if args.include_title_in_knowledge_dataset:
                    if "title" not in passage_dataset.column_names:
                        raise ValueError(
                            "The dataset must contain a column named 'title' if args.include_title_in_knowledge_dataset is True."
                        )
                    passage_dataset = passage_dataset.map(
                        lambda example: {
                            "gold_passage": example["title"]
                            + " "
                            + example["gold_passage"]
                        }
                    )
            try:
                passage_dataset = concatenate_datasets(
                    [passage_dataset, additional_passages]
                )
            except ValueError:
                # Log the features in the two datasets
                logger.warning(
                    "Mismatched features (columns) in the passage dataset and additional passages."
                )
                logger.info(
                    "The following features are in the first dataset:\n{}".format(
                        "\n".join(passage_dataset.column_names)
                    )
                )
                logger.info(
                    "The following features are in the second dataset:\n{}".format(
                        "\n".join(additional_passages.column_names)
                    )
                )
                logger.warning("Removing all features except passages as a workaround.")

                passage_dataset = passage_dataset.remove_columns(
                    [c for c in passage_dataset.column_names if c != "passages"]
                )

                additional_passages = additional_passages.remove_columns(
                    [c for c in additional_passages.column_names if c != "passages"]
                )

                passage_dataset = concatenate_datasets(
                    [passage_dataset, additional_passages]
                )

        if args.remove_duplicates_from_eval_passages:
            passage_dataset = HFDataset.from_pandas(
                passage_dataset.to_pandas().drop_duplicates(subset=["passages"])
            )
            passage_dataset = passage_dataset.remove_columns("__index_level_0__")

        logger.info("Loading evaluation passages to a Huggingface Dataset completed.")

        if "embeddings" not in passage_dataset.column_names:
            logger.info("Generating embeddings for evaluation passages")
            if args.fp16:
                from torch.cuda import amp
            else:
                amp = None
            encoder = encoder.to(device)
            passage_dataset = passage_dataset.map(
                partial(
                    embed,
                    encoder=encoder,
                    tokenizer=tokenizer,
                    concatenate_embeddings=args.larger_representations,
                    extra_cls_token_count=args.extra_cls_token_count,
                    device=device,
                    fp16=args.fp16,
                    amp=amp,
                ),
                batched=True,
                batch_size=args.embed_batch_size,
            )

            logger.info("Generating embeddings for evaluation passages completed.")
            if args.save_passage_dataset:
                output_dataset_directory = os.path.join(
                    args.output_dir, "passage_dataset"
                )
                os.makedirs(output_dataset_directory, exist_ok=True)
                passage_dataset.save_to_disk(output_dataset_directory)

        logger.info("Adding FAISS index to evaluation passages")
        index = faiss.IndexHNSWFlat(
            args.faiss_d, args.faiss_m, faiss.METRIC_INNER_PRODUCT
        )
        passage_dataset.add_faiss_index("embeddings", custom_index=index)
        passage_index = DPRIndex(passage_dataset, context_config.hidden_size)
        logger.info("Adding FAISS index to evaluation passages completed.")
        if args.save_passage_dataset:
            output_dataset_directory = os.path.join(args.output_dir, "passage_dataset")
            os.makedirs(output_dataset_directory, exist_ok=True)
            faiss_save_path = os.path.join(
                output_dataset_directory, "hf_dataset_index.faiss"
            )
            passage_dataset.save_faiss_index("embeddings", faiss_save_path)
    else:
        logger.info(f"Loading passage dataset from {passage_dataset}")
        passage_data = load_from_disk(passage_dataset)
        index_path = os.path.join(passage_dataset, "hf_dataset_index.faiss")
        if os.path.isfile(index_path):
            passage_data.load_faiss_index("embeddings", index_path)
            passage_dataset = passage_data
        else:
            logger.info("Adding FAISS index to evaluation passages")
            index = faiss.IndexHNSWFlat(
                args.faiss_d, args.faiss_m, faiss.METRIC_INNER_PRODUCT
            )
            passage_dataset.add_faiss_index("embeddings", custom_index=index)
            logger.info("Adding FAISS index to evaluation passages completed.")
            if args.save_passage_dataset:
                output_dataset_directory = os.path.join(
                    args.output_dir, "passage_dataset"
                )
                faiss_save_path = os.path.join(
                    output_dataset_directory, "hf_dataset_index.faiss"
                )
                passage_dataset.save_faiss_index("embeddings", faiss_save_path)

        logger.info(f"Succesfully loaded passage dataset from {passage_dataset}")
        passage_index = DPRIndex(passage_dataset, context_config.hidden_size)
    return passage_index


def get_prediction_passage_dataset(
    prediction_passages,
    encoder,
    tokenizer,
    context_config,
    args,
    device,
):
    import faiss

    logger.info("Preparing prediction passages started")
    if isinstance(prediction_passages, str):
        if os.path.isdir(prediction_passages):
            prediction_passages_dataset = load_from_disk(prediction_passages)
        else:
            prediction_passages_dataset = load_dataset(
                "csv",
                data_files=prediction_passages,
                delimiter="\t",
                column_names=["passages"],
                cache_dir=args.dataset_cache_dir,
            )
            prediction_passages_dataset = prediction_passages_dataset["train"]
            if args.include_title_in_knowledge_dataset:
                if "title" not in prediction_passages_dataset.column_names:
                    raise ValueError(
                        "The dataset must contain a column named 'title' if args.include_title_in_knowledge_dataset is True."
                    )
                prediction_passages_dataset = prediction_passages_dataset.map(
                    lambda example: {
                        "gold_passage": example["title"] + " " + example["gold_passage"]
                    }  # Should these be "passage" instead of "gold_passage"?
                )
    elif isinstance(prediction_passages, list):
        prediction_passages_dataset = HFDataset.from_dict(
            {"passages": prediction_passages}
        )
    else:
        prediction_passages_dataset = HFDataset.from_pandas(prediction_passages)
        if "title" not in prediction_passages_dataset.column_names:
            raise ValueError(
                "The dataset must contain a column named 'title' if args.include_title_in_knowledge_dataset is True."
            )
        prediction_passages_dataset = prediction_passages_dataset.map(
            lambda example: {
                "gold_passage": example["title"] + " " + example["gold_passage"]
            }
        )

    logger.info("Preparing prediction passages completed")
    if "embeddings" not in prediction_passages_dataset.column_names:
        logger.info("Generating embeddings for prediction passages started")

        if args.fp16:
            from torch.cuda import amp
        else:
            amp = None

        encoder = encoder.to(device)
        prediction_passages_dataset = prediction_passages_dataset.map(
            partial(
                embed,
                encoder=encoder,
                tokenizer=tokenizer,
                concatenate_embeddings=args.larger_representations,
                extra_cls_token_count=args.extra_cls_token_count,
                device=device,
                fp16=args.fp16,
                amp=amp,
            ),
            batched=True,
            batch_size=args.embed_batch_size,
            with_rank=args.n_gpu > 1,
            num_proc=args.n_gpu,
        )

        logger.info("Generating embeddings for prediction passages completed")
        if args.save_passage_dataset:
            output_dataset_directory = os.path.join(
                args.output_dir, "prediction_passage_dataset"
            )
            os.makedirs(output_dataset_directory, exist_ok=True)
            prediction_passages_dataset.save_to_disk(output_dataset_directory)

    index_added = False
    index_path = None
    if isinstance(prediction_passages, str):
        index_path = os.path.join(prediction_passages, "hf_dataset_index.faiss")
        if os.path.isfile(index_path):
            logger.info(f"Loaded FAISS index from {index_path}")
            prediction_passages_dataset.load_faiss_index("embeddings", index_path)
            index_added = True

    if not index_added:
        logger.info("Adding FAISS index to prediction passages")
        if args.faiss_index_type == "IndexHNSWFlat":
            index = faiss.IndexHNSWFlat(
                args.faiss_d, args.faiss_m, faiss.METRIC_INNER_PRODUCT
            )
        elif args.faiss_index_type == "IndexFlatIP":
            index = faiss.IndexFlatIP(args.faiss_d)
        else:
            raise ValueError(
                f"Unsupported FAISS index type {args.faiss_index_type}. Choose from IndexHNSWFlat and IndexFlatIP"
            )
        prediction_passages_dataset.add_faiss_index(
            "embeddings", custom_index=index, faiss_verbose=True
        )
        logger.info("Adding FAISS index to prediction passages completed")
        if args.save_passage_dataset:
            output_dataset_directory = os.path.join(
                args.output_dir, "prediction_passage_dataset"
            )
            os.makedirs(output_dataset_directory, exist_ok=True)
            faiss_save_path = os.path.join(
                output_dataset_directory, "hf_dataset_index.faiss"
            )
            prediction_passages_dataset.save_faiss_index("embeddings", faiss_save_path)

    passage_index = DPRIndex(prediction_passages_dataset, context_config.hidden_size)
    return passage_index


class DPRIndex(Index):
    def __init__(self, dataset, vector_size):
        self.dataset = dataset
        self.vector_size = vector_size

    def get_doc_dicts(self, doc_ids):
        return [
            self.dataset[doc_ids[i].tolist()]
            for i in tqdm(range(doc_ids.shape[0]), desc="Retrieving doc dicts")
        ]

    def get_top_docs(self, question_hidden_states, n_docs=5):
        _, ids = self.dataset.search_batch("embeddings", question_hidden_states, n_docs)
        docs = [self.dataset[[i for i in indices if i >= 0]] for indices in ids]
        vectors = [doc["embeddings"] for doc in docs]
        for i in range(len(vectors)):
            if len(vectors[i]) < n_docs:
                vectors[i] = np.vstack(
                    [vectors[i], np.zeros((n_docs - len(vectors[i]), self.vector_size))]
                )
        return (
            np.array(ids),
            np.array(vectors),
            docs,
        )  # shapes (batch_size, n_docs) and (batch_size, n_docs, d)

    def __len__(self):
        return len(self.dataset)


class PrecomputedEmbeddingsDataset(Dataset):
    def __init__(self, embeddings_path):
        self.embeddings = torch.load(embeddings_path)

    def __getitem__(self, index):
        return self.embeddings[index]

    def __len__(self):
        return len(self.embeddings)


def mean_reciprocal_rank_at_k(rs, k, return_individual_scores=False):
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

    mrr_scores = [1.0 / (r[0] + 1) if r.size else 0.0 for r in rs]
    if return_individual_scores:
        return np.mean(mrr_scores), mrr_scores
    else:
        return np.mean(mrr_scores)


def get_recall_at_k(rs, rt, k):
    rs = rs[:, :k]
    recall_scores = []
    for r, t in zip(rs, rt):
        recall_scores.append(np.sum(r) / min(t, k))
    return np.mean(recall_scores), recall_scores


class ClusteredDataset(Dataset):
    def __init__(self, clusters):
        self.clusters = clusters

    def __getitem__(self, index):
        return self.clusters[index]

    def __len__(self):
        return len(self.clusters)


def get_relevant_labels(context_ids_all, current_context_ids):
    current_context_ids = current_context_ids["context_ids"]
    label_tensor = torch.zeros(context_ids_all.shape[0], dtype=torch.long)
    relevant_indices = torch.where(
        torch.all(context_ids_all == current_context_ids, dim=1)
    )[0]
    label_tensor[relevant_indices] = 1
    return {"labels": label_tensor.float()}


def get_relevant_labels_batched(context_ids_all, context_ids_batch):
    context_ids_batch = context_ids_batch["context_ids"]
    labels = np.zeros((context_ids_batch.shape[0], context_ids_all.shape[0]))
    for i in range(context_ids_batch.shape[0]):
        relevant_indices = np.argwhere(
            np.isin(context_ids_all, context_ids_batch[i]).all(axis=1)
        )
        labels[i, relevant_indices] = 1
    return {"labels": labels.astype(float)}


def dataset_map_multiprocessed(batch_dataset):
    return batch_dataset.map(partial(get_relevant_labels, batch_dataset["context_ids"]))


def get_clustered_passage_dataset(
    passage_dataset,
    train_batch_size,
    encoder,
    tokenizer,
    args,
    device,
):
    logger.info("Generating embeddings for passage clustering started")

    if args.fp16:
        from torch.cuda import amp
    else:
        amp = None

    encoder = encoder.to(device)
    passage_dataset = passage_dataset.rename_column("gold_passage", "passages")
    passage_dataset = passage_dataset.map(
        partial(
            embed,
            encoder=encoder,
            tokenizer=tokenizer,
            concatenate_embeddings=args.larger_representations,
            extra_cls_token_count=args.extra_cls_token_count,
            device=device,
            fp16=args.fp16,
            amp=amp,
            pretokenized=True,
        ),
        batched=True,
        batch_size=args.embed_batch_size,
        # with_rank=args.n_gpu > 1,
        # num_proc=args.n_gpu,
    )
    passage_dataset = passage_dataset.rename_column("passages", "gold_passage")
    logger.info("Generating embeddings for passage clustering completed")

    logger.info("Clustering passages started")

    k = int(len(passage_dataset["embeddings"]) / train_batch_size)
    km = MiniBatchKMeans(
        n_clusters=k,
        init="k-means++",
    )

    if args.cluster_train_size is not None:
        clustering_subset, _ = train_test_split(
            passage_dataset, train_size=args.cluster_train_size
        )
    else:
        clustering_subset = passage_dataset
    km.fit(clustering_subset["embeddings"])
    passage_dataset = passage_dataset.add_column(
        "cluster_id", km.predict(passage_dataset["embeddings"])
    )

    logger.info("Clustering passages completed")

    clusters = passage_dataset["cluster_id"].tolist()
    clustered_batches = {i: [[]] for i in range(k)}
    for i, cluster in enumerate(clusters):
        mini_batch_num = len(clustered_batches[cluster]) - 1
        if len(clustered_batches[cluster][mini_batch_num]) < train_batch_size:
            clustered_batches[cluster][mini_batch_num].append(i)
        else:
            clustered_batches[cluster].append([i])

    clustered_batches = [
        batch for batch_list in clustered_batches.values() for batch in batch_list
    ]

    clustered_batches = sorted(clustered_batches, key=lambda x: len(x), reverse=True)

    final_batches = []
    remaining_batches = []
    for batch in clustered_batches:
        if len(batch) == train_batch_size:
            final_batches.append(batch)
        else:
            remaining_batches.append(batch)
    for _ in range(len(remaining_batches)):
        combined_batches = []
        for i, batch in enumerate(remaining_batches):
            if len(batch) == 0:
                continue
            else:
                for j in range(i + 1, len(remaining_batches)):
                    if (
                        len(batch) + len(remaining_batches[j]) <= train_batch_size
                        and i != j
                    ):
                        combined_batches.append(batch + remaining_batches[j])
                        remaining_batches[j] = []
                        remaining_batches[i] = []
                        break
        remaining_batches = combined_batches + [
            batch for batch in remaining_batches if len(batch) > 0
        ]
        if not combined_batches:
            final_batches.extend(
                [batch for batch in remaining_batches if len(batch) > 0]
            )
            break

    batch_datasets = [passage_dataset.select(batch) for batch in final_batches]

    if datasets.logging.is_progress_bar_enabled():
        datasets.logging.disable_progress_bar()
        reenable_progress_bar = True
    else:
        reenable_progress_bar = False

    batch_datasets = [
        batch_dataset.map(partial(get_relevant_labels, batch_dataset["context_ids"]))
        for batch_dataset in tqdm(batch_datasets, desc="Generating labels")
    ]

    # batch_datasets = [
    #     batch_dataset.map(
    #         partial(get_relevant_labels_batched, batch_dataset["context_ids"]),
    #         batched=True,
    #         batch_size=len(batch_dataset),
    #     )
    #     for batch_dataset in tqdm(batch_datasets, desc="Generating labels")
    # ]

    # if args.multiprocessing_chunksize == -1:
    #     multiprocessing_chunksize = len(batch_datasets) // (cpu_count() * 10)
    # else:
    #     multiprocessing_chunksize = args.multiprocessing_chunksize

    # with Pool(args.process_count) as p:
    #     batch_datasets = list(
    #         tqdm(
    #             p.imap(dataset_map_multiprocessed, batch_datasets, chunksize=multiprocessing_chunksize),
    #             desc="Generating labels",
    #             total=len(batch_datasets),
    #         )
    #     )

    if reenable_progress_bar:
        datasets.logging.enable_progress_bar()

    return ClusteredDataset(batch_datasets)
