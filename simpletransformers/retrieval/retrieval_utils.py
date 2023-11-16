import logging
import math
import os
import pickle
import json
import random
from collections import deque
from multiprocessing import Pool, cpu_count
from functools import partial
from simpletransformers.seq2seq.seq2seq_utils import add_faiss_index_to_dataset
from simpletransformers.config.model_args import get_default_process_count
import datasets
from datasets.load import load_from_disk

import torch
import torch.nn as nn
import transformers
import numpy as np
import faiss
from sklearn.cluster import MiniBatchKMeans
from sklearn.model_selection import train_test_split

from torch.utils.data import Dataset, IterableDataset
from tqdm.auto import tqdm

from datasets import Features, Sequence, Value, load_dataset, concatenate_datasets
from datasets import Dataset as HFDataset

from transformers.models.rag.retrieval_rag import Index


logger = logging.getLogger(__name__)


# Setting FAISS threads
# faiss.omp_set_num_threads(get_default_process_count())


def load_hf_dataset(
    data,
    context_tokenizer,
    query_tokenizer,
    args,
    evaluate=False,
    teacher_tokenizer=None,
    clustered_training=False,
):
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

    # Assign an id to each unique gold_passage
    # passage_dict = {}

    # for i, passage in enumerate(dataset["gold_passage"]):
    #     if passage not in passage_dict:
    #         passage_dict[passage] = i

    # dataset = dataset.map(
    #     lambda example: {"passage_id": passage_dict[example["gold_passage"]]},
    #     desc="Assigning passage ids",
    # )

    if args.cluster_concatenated:
        dataset = dataset.map(
            lambda x: {
                "passages_for_clustering": [x["query_text"] + " " + x["gold_passage"]]
            },
        )

    n_hard_negatives = args.n_hard_negatives

    dataset = dataset.map(
        lambda x: preprocess_batch_for_hf_dataset(
            x,
            context_tokenizer=context_tokenizer,
            query_tokenizer=query_tokenizer,
            args=args,
            evaluate=evaluate,
            teacher_tokenizer=teacher_tokenizer,
            n_hard_negatives=n_hard_negatives,
        ),
        batched=True,
    )

    if args.hard_negatives and (args.hard_negatives_in_eval or not evaluate):
        if n_hard_negatives == 1:
            column_names = [
                "context_ids",
                "query_ids",
                "hard_negative_ids",
                "context_mask",
                "query_mask",
                "hard_negatives_mask",
                # "passage_id",
            ]
        else:
            column_names = [
                "context_ids",
                "query_ids",
                "context_mask",
                "query_mask",
                # "passage_id",
            ]
            for i in range(n_hard_negatives):
                column_names.append(f"hard_negative_{i}_ids")
                column_names.append(f"hard_negative_{i}_mask")
    else:
        if args.cluster_concatenated:
            column_names = [
                "context_ids",
                "query_ids",
                "context_mask",
                "query_mask",
                "clustering_context_ids",
                "clustering_context_mask",
            ]
        else:
            column_names = [
                "context_ids",
                "query_ids",
                "context_mask",
                "query_mask",
                # "passage_id",
            ]

    if args.unified_cross_rr and teacher_tokenizer:
        column_names += [
            "reranking_context_ids",
            "reranking_context_mask",
            "reranking_query_ids",
            "reranking_query_mask",
        ]

    if evaluate:
        gold_passages = dataset["gold_passage"]
        dataset.set_format(type="pt", columns=column_names)

        return dataset, gold_passages
    else:
        dataset.set_format(type="pt", columns=column_names)
        if args.unified_cross_rr and not clustered_training and not evaluate:
            dataset = dataset.to_pandas()
            dataset = np.array_split(
                dataset, math.ceil(len(dataset) / args.train_batch_size)
            )
            batch_datasets = [HFDataset.from_pandas(df) for df in dataset]

            dataset = ClusteredDataset(batch_datasets, len(batch_datasets))

        return dataset


def preprocess_batch_for_hf_dataset(
    dataset,
    context_tokenizer,
    query_tokenizer,
    args,
    evaluate=False,
    teacher_tokenizer=None,
    n_hard_negatives=1,
):
    if teacher_tokenizer is None:
        unified_rr = False
    else:
        unified_rr = True

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

    if unified_rr or (args.unified_cross_rr and teacher_tokenizer):
        reranking_query_inputs = teacher_tokenizer(
            dataset["query_text"],
            padding=False,
            return_tensors="np",
            truncation=True,
        )

        reranking_context_inputs = teacher_tokenizer(
            dataset["gold_passage"],
            padding=False,
            return_tensors="np",
            truncation=True,
        )

        reranking_context_ids = reranking_context_inputs["input_ids"]
        reranking_context_mask = reranking_context_inputs["attention_mask"]
        reranking_query_ids = reranking_query_inputs["input_ids"]
        reranking_query_mask = reranking_query_inputs["attention_mask"]

    if args.cluster_concatenated:
        try:
            clustering_context_inputs = context_tokenizer(
                dataset["passages_for_clustering"],
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
            dataset["passages_for_clustering"] = [
                str(p) for p in dataset["passages_for_clustering"]
            ]
            clustering_context_inputs = context_tokenizer(
                dataset["passages_for_clustering"],
                max_length=args.max_seq_length,
                padding="max_length",
                return_tensors="np",
                truncation=True,
            )

        clustering_context_ids = clustering_context_inputs["input_ids"].squeeze()
        clustering_context_mask = clustering_context_inputs["attention_mask"].squeeze()

    if args.hard_negatives and (args.hard_negatives_in_eval or not evaluate):
        try:
            if n_hard_negatives > 1:
                hard_negatives_inputs = [
                    context_tokenizer(
                        dataset[f"hard_negative_{i}"],
                        max_length=args.max_seq_length,
                        padding="max_length",
                        return_tensors="np",
                        truncation=True,
                    )
                    for i in range(n_hard_negatives)
                ]
            else:
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
        if n_hard_negatives > 1:
            hard_negatives_inputs = [
                {
                    "input_ids": hard_negatives_input["input_ids"].squeeze(),
                    "attention_mask": hard_negatives_input["attention_mask"].squeeze(),
                }
                for hard_negatives_input in hard_negatives_inputs
            ]
            return {
                "context_ids": context_ids,
                "query_ids": query_ids,
                "context_mask": context_mask,
                "query_mask": query_mask,
                **{
                    f"hard_negative_{i}_ids": hard_negatives_input[
                        "input_ids"
                    ].squeeze()
                    for i, hard_negatives_input in enumerate(hard_negatives_inputs)
                },
                **{
                    f"hard_negative_{i}_mask": hard_negatives_input[
                        "attention_mask"
                    ].squeeze()
                    for i, hard_negatives_input in enumerate(hard_negatives_inputs)
                },
            }
        else:
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

    if args.cluster_concatenated:
        return {
            "context_ids": context_ids,
            "query_ids": query_ids,
            "context_mask": context_mask,
            "query_mask": query_mask,
            "clustering_context_ids": clustering_context_ids,
            "clustering_context_mask": clustering_context_mask,
        }
    else:
        if unified_rr:
            return {
                "context_ids": context_ids,
                "query_ids": query_ids,
                "context_mask": context_mask,
                "query_mask": query_mask,
                "reranking_context_ids": reranking_context_ids,
                "reranking_context_mask": reranking_context_mask,
                "reranking_query_ids": reranking_query_ids,
                "reranking_query_mask": reranking_query_mask,
            }
        else:
            return {
                "context_ids": context_ids,
                "query_ids": query_ids,
                "context_mask": context_mask,
                "query_mask": query_mask,
            }


def get_output_embeddings(
    embeddings, concatenate_embeddings=False, n_cls_tokens=3, use_pooler_output=False
):
    """
    Extracts the embeddings from the output of the model.
    Concatenates CLS embeddings if concatenate_embeddings is True.
    """
    if concatenate_embeddings:
        return embeddings.last_hidden_state[:, :n_cls_tokens, :].reshape(
            embeddings.last_hidden_state.shape[0], -1
        )
    else:
        if use_pooler_output:
            return embeddings.pooler_output
        try:
            return embeddings[0][:, 0, :]
        except IndexError:
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
    cluster_concatenated=False,
    unified_rr=False,
    passage_column="passages",
):
    """Compute the DPR embeddings of document passages"""
    if rank is not None:
        device = torch.device("cuda", rank)
        encoder = encoder.to(device)

    if cluster_concatenated:
        context_column = "clustering_context_ids"
        context_mask_column = "clustering_context_mask"
    else:
        if passage_column == "query_text":
            context_column = "query_ids"
            context_mask_column = "query_mask"
        else:
            context_column = "context_ids"
            context_mask_column = "context_mask"
    with torch.no_grad():
        if fp16:
            with amp.autocast():
                if not pretokenized:
                    try:
                        tokenized_inputs = tokenizer(
                            documents[passage_column],
                            truncation=True,
                            padding="max_length",
                            max_length=256,
                            return_tensors="pt",
                        )
                        embeddings = encoder(
                            tokenized_inputs["input_ids"].to(device=device),
                            tokenized_inputs["attention_mask"].to(device=device),
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
                        documents[passage_column] = [
                            str(p) for p in documents[passage_column]
                        ]
                        tokenized_inputs = tokenizer(
                            documents[passage_column],
                            truncation=True,
                            padding="longest",
                            return_tensors="pt",
                        )
                        embeddings = encoder(
                            tokenized_inputs["input_ids"].to(device=device),
                            tokenized_inputs["attention_mask"].to(device=device),
                            return_dict=True,
                        )
                        embeddings = get_output_embeddings(
                            embeddings,
                            concatenate_embeddings=concatenate_embeddings,
                            n_cls_tokens=(1 + extra_cls_token_count),
                        )
                else:
                    embeddings = encoder(
                        documents[context_column].to(device=device),
                        documents[context_mask_column].to(device=device),
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
                    tokenized_inputs = tokenizer(
                        documents[passage_column],
                        truncation=True,
                        padding="longest",
                        return_tensors="pt",
                    )
                    embeddings = encoder(
                        tokenized_inputs["input_ids"].to(device=device),
                        tokenized_inputs["attention_mask"].to(device=device),
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
                    documents[passage_column] = [
                        str(p) for p in documents[passage_column]
                    ]
                    tokenized_inputs = tokenizer(
                        documents[passage_column],
                        truncation=True,
                        padding="longest",
                        return_tensors="pt",
                    )
                    embeddings = encoder(
                        tokenized_inputs["input_ids"].to(device=device),
                        tokenized_inputs["attention_mask"].to(device=device),
                        return_dict=True,
                    )
                    embeddings = get_output_embeddings(
                        embeddings,
                        concatenate_embeddings=concatenate_embeddings,
                        n_cls_tokens=(1 + extra_cls_token_count),
                    )
            else:
                embeddings = encoder(
                    documents[context_column].to(device=device),
                    documents[context_mask_column].to(device=device),
                    return_dict=True,
                )
                embeddings = get_output_embeddings(
                    embeddings,
                    concatenate_embeddings=concatenate_embeddings,
                    n_cls_tokens=(1 + extra_cls_token_count),
                )

    if unified_rr:
        embeddings = embeddings.detach().cpu().numpy()
        rerank_embeddings = embeddings[:, : embeddings.shape[1] // 2]
        embeddings = embeddings[:, embeddings.shape[1] // 2 :]
        return {"embeddings": embeddings, "rerank_embeddings": rerank_embeddings}
    else:
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
                    include_title=args.include_title_in_corpus,
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
                if args.include_title_in_corpus:
                    if "title" not in passage_dataset.column_names:
                        raise ValueError(
                            "The dataset must contain a column named 'title' if args.include_title_in_corpus is True."
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
            if args.include_title_in_corpus:
                if "title" not in passage_dataset.column_names:
                    raise ValueError(
                        "The dataset must contain a column named 'title' if args.include_title_in_corpus is True."
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

        if args.hard_negatives and "hard_negative" in passage_dataset.column_names:
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
                    if args.include_title_in_corpus:
                        if "title" not in passage_dataset.column_names:
                            raise ValueError(
                                "The dataset must contain a column named 'title' if args.include_title_in_corpus is True."
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
                if args.include_title_in_corpus:
                    if "title" not in passage_dataset.column_names:
                        raise ValueError(
                            "The dataset must contain a column named 'title' if args.include_title_in_corpus is True."
                        )
                    additional_passages = additional_passages.map(
                        lambda example: {
                            "gold_passage": example["title"]
                            + " "
                            + example["gold_passage"]
                        }
                    )
                    # additional_passages = additional_passages.rename_column("gold_passage", "passages")
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
                logger.warning(
                    "Removing all features except passages and pid as a workaround."
                )

                passage_dataset = passage_dataset.remove_columns(
                    [
                        c
                        for c in passage_dataset.column_names
                        if c not in ["passages", "pid"]
                    ]
                )

                additional_passages = additional_passages.remove_columns(
                    [
                        c
                        for c in additional_passages.column_names
                        if c not in ["passages", "pid"]
                    ]
                )

                passage_dataset = concatenate_datasets(
                    [passage_dataset, additional_passages]
                )

        if args.remove_duplicates_from_eval_passages:
            passage_dataset = HFDataset.from_pandas(
                passage_dataset.to_pandas().drop_duplicates(subset=["passages", "pid"])
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
        index = get_faiss_index(args)
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
            index = get_faiss_index(args)
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
            if args.include_title_in_corpus:
                if "title" not in prediction_passages_dataset.column_names:
                    raise ValueError(
                        "The dataset must contain a column named 'title' if args.include_title_in_corpus is True."
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
        if args.include_title_in_corpus:
            if "title" not in prediction_passages_dataset.column_names:
                raise ValueError(
                    "The dataset must contain a column named 'title' if args.include_title_in_corpus is True."
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
                unified_rr=args.unified_rr,
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
        index = get_faiss_index(args)
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

    def get_top_docs(
        self, question_hidden_states, n_docs=5, passages_only=False, return_indices=True
    ):
        if passages_only:
            _, docs = self.dataset.get_nearest_examples_batch(
                "embeddings", question_hidden_states, n_docs
            )
            return docs

        _, ids = self.dataset.search_batch("embeddings", question_hidden_states, n_docs)
        docs = [self.dataset[[i for i in indices if i >= 0]] for indices in ids]
        try:
            doc_ids = [doc["passage_id"] for doc in docs]
        except KeyError:
            raise KeyError(
                "The dataset must contain a column named 'passage_id' if passages_only is False."
            )
        vectors = [doc["embeddings"] for doc in docs]
        for i in range(len(vectors)):
            if len(vectors[i]) < n_docs:
                vectors[i] = np.vstack(
                    [vectors[i], np.zeros((n_docs - len(vectors[i]), self.vector_size))]
                )
        if return_indices:
            return (
                ids,
                np.array(vectors),
                docs,
            )
        else:
            return (
                np.array(doc_ids),
                np.array(vectors),
                docs,
            )

    def get_top_doc_ids(
        self, question_hidden_states, n_docs=5, reranking_query_outputs=None
    ):
        _, ids = self.dataset.search_batch("embeddings", question_hidden_states, n_docs)
        docs = [self.dataset[[i for i in indices if i >= 0]] for indices in ids]

        doc_ids = [doc["passage_id"] for doc in docs]

        if reranking_query_outputs is None:
            return doc_ids

        rerank_similarity = compute_rerank_similarity(
            reranking_query_outputs, docs, passage_column="passage_text"
        )

        rerank_indices = np.argsort(rerank_similarity, axis=1)[:, ::-1]
        rerank_similarity_reordered = np.take_along_axis(
            rerank_similarity, rerank_indices, 1
        )

        # Original id order could also be returned here if needed

        # Reorder doc_ids. doc_ids is a list of lists, so numpy indexing is not possible
        doc_ids_reordered = []
        for i in range(len(doc_ids)):
            doc_ids_reordered.append([doc_ids[i][j] for j in rerank_indices[i]])

        return doc_ids_reordered, rerank_similarity_reordered

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
    def __init__(self, clusters, length=None):
        self.clusters = clusters
        # self.length = length

    def __getitem__(self, index):
        # return next(self.clusters)
        return self.clusters[index]

    def __len__(self):
        # return self.length
        return len(self.clusters)


# class IterableClusteredDataset(IterableDataset):
#     def __init__(self, passage_dataset, clustered_batches):
#         self.passage_dataset = passage_dataset
#         self.clustered_batches = clustered_batches
#         self.start = 0
#         self.end = len(self.clustered_batches)

#     def __iter__(self):
#         worker_info = torch.utils.data.get_worker_info()
#         if worker_info is None:
#             # single-process data loading, return the full iterator
#             iter_start = self.start
#             iter_end = self.end
#         else:
#             # in a worker process
#             per_worker = int(math.ceil((self.end - self.start) / float(worker_info.num_workers)))
#             worker_id = worker_info.id
#             iter_start = self.start + worker_id * per_worker
#             iter_end = min(iter_start + per_worker, self.end)


def get_relevant_labels(context_ids_all, current_context_ids):
    current_context_ids = current_context_ids["context_ids"]
    label_tensor = torch.zeros(context_ids_all.shape[0], dtype=torch.long)
    relevant_indices = torch.where(
        torch.all(context_ids_all == current_context_ids, dim=1)
    )[0]
    label_tensor[relevant_indices] = 1
    return {"labels": label_tensor.float()}


def get_relevant_labels_with_ids(passage_ids_all, current_passage_ids):
    current_passage_ids = current_passage_ids["passage_id"]
    label_tensor = torch.zeros(passage_ids_all.shape[0], dtype=torch.long)
    relevant_indices = torch.where(passage_ids_all == current_passage_ids)[0]
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


def batch_generator(passage_dataset, batches):
    for batch in batches:
        yield passage_dataset.select(batch)


# def tokenize_for_reranking(dataset, tokenizer):
#     tokenized_query_inputs = tokenizer(
#         dataset["query_text"],
#         padding="longest",
#         truncation=True,
#         return_tensors="np",
#     )
#     dataset = dataset.add_column("rerank_query_input_ids", tokenized_query_inputs["input_ids"])
#     dataset = dataset.add_column("rerank_query_attention_mask", tokenized_query_inputs["attention_mask"])

#     tokenized_passage_inputs = tokenizer(
#         dataset["passage_text"],
#         padding="longest",
#         truncation=True,
#     )
#     dataset = dataset.add_column("rerank_passage_input_ids", tokenized_passage_inputs["input_ids"])
#     dataset = dataset.add_column("rerank_passage_attention_mask", tokenized_passage_inputs["attention_mask"])

#     return dataset


# def prepare_batch_for_reranking(batch, tokenizer):
#     queries = batch["query_text"]
#     passages = batch["gold_passage"]

#     concatenated_text = {
#         "query_text": [],
#         "passage_text": [],
#     }
#     for query in queries:
#         for passage in passages:
#             concatenated_text["query_text"].append(query)
#             concatenated_text["passage_text"].append(passage)

#     reranking_batch_dataset = HFDataset.from_dict(concatenated_text)

#     return reranking_batch_dataset


# def embed_reranking(
#     tokenized_reranking_dataset,
#     rank=None,
#     reranker=None,
#     tokenizer=None,
#     device=None,
#     fp16=False,
#     amp=None,
# ):

#     if rank is not None:
#         device = torch.device("cuda", rank)
#     reranker = reranker.to(device)

#     tokenized_inputs = tokenizer(
#         tokenized_reranking_dataset["query_text"],
#         tokenized_reranking_dataset["passage_text"],
#         truncation=True,
#         padding="longest",
#         return_tensors="pt",
#     )

#     reranker.eval()
#     with torch.no_grad():
#         if fp16:
#             with amp.autocast():
#                 reranking_embeddings = reranker(
#                     tokenized_inputs["input_ids"].to(device),
#                     tokenized_inputs["attention_mask"].to(device),
#                     tokenized_inputs["token_type_ids"].to(device),
#                 ).logits
#         else:
#             reranking_embeddings = reranker(
#                 tokenized_inputs["input_ids"].to(device),
#                 tokenized_inputs["attention_mask"].to(device),
#                 tokenized_inputs["token_type_ids"].to(device),
#             ).logits

#     return {"rerank_embeddings": reranking_embeddings.detach().cpu().numpy()}


# def generate_reranking_labels(batch_datasets, reranker, reranker_tokenizer, args, device):
#     logger.info("Generating labels for reranking started")
#     if args.fp16:
#         from torch.cuda import amp
#     else:
#         amp = None

#     reranker = reranker.to(device)

#     batch_lengths = []
#     reranking_all_batches_dataset = {
#         "query_text": [],
#         "passage_text": [],
#     }
#     for batch_dataset in batch_datasets:
#         reranking_batch_dataset = prepare_batch_for_reranking(batch_dataset, reranker_tokenizer)
#         batch_lengths.append((len(reranking_batch_dataset)))
#         for key in reranking_all_batches_dataset.keys():
#             reranking_all_batches_dataset[key].extend(
#                 reranking_batch_dataset[key]
#             )

#     reranking_all_batches_dataset = HFDataset.from_dict(reranking_all_batches_dataset)
#     reranking_all_batches_dataset = reranking_all_batches_dataset.map(
#         partial(
#             embed_reranking,
#             reranker=reranker,
#             tokenizer=reranker_tokenizer,
#             device=device,
#             fp16=args.fp16,
#             amp=amp,
#         ),
#         batched=True,
#         batch_size=args.rerank_batch_size,
#         with_rank=args.n_gpu > 1,
#         num_proc=args.n_gpu,
#     )

#     current_index = 0
#     reranking_embeddings = []
#     for batch_length in batch_lengths:
#         reranking_embeddings.append(
#             reranking_all_batches_dataset["rerank_embeddings"][current_index : current_index + batch_length]
#         )
#         current_index += batch_length

#     # Add the reranking embeddings to the batch datasets
#     for i, batch_dataset in enumerate(batch_datasets):
#         len_batch = len(batch_dataset)
#         rerank_embeddings_batch = reranking_embeddings[i].reshape(len_batch, len_batch, -1)
#         batch_dataset = batch_dataset.add_column("rerank_embeddings", reranking_embeddings[i])
#         batch_datasets[i] = batch_dataset

#     return batch_datasets


def get_clustered_passage_dataset(
    passage_dataset,
    train_batch_size,
    encoder,
    tokenizer,
    args,
    device,
    teacher_model=None,
    teacher_tokenizer=None,
    clustered_batch_randomize_percentage=0.0,
    epoch_number=None,
):
    if args.fp16:
        from torch.cuda import amp
    else:
        amp = None

    if epoch_number is None or epoch_number % args.cluster_every_n_epochs == 0:
        encoder = encoder.to(device)

        if args.cluster_queries:
            logger.info("Generating embeddings for query clustering started")
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
                    cluster_concatenated=args.cluster_concatenated,
                    passage_column="query_text",
                ),
                batched=True,
                batch_size=args.embed_batch_size,
                with_rank=args.n_gpu > 1,
                num_proc=args.n_gpu,
                # cache_file_name=os.path.join(args.dataset_cache_dir, args.output_dir, "/clustering_embeddings.cache"),
            )
            logger.info("Generating embeddings for query clustering completed")

        else:
            logger.info("Generating embeddings for passage clustering started")
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
                    cluster_concatenated=args.cluster_concatenated,
                ),
                batched=True,
                batch_size=args.embed_batch_size,
                with_rank=args.n_gpu > 1,
                num_proc=args.n_gpu,
                # cache_file_name=os.path.join(args.dataset_cache_dir, args.output_dir, "/clustering_embeddings.cache"),
            )
            passage_dataset = passage_dataset.rename_column("passages", "gold_passage")
            logger.info("Generating embeddings for passage clustering completed")

        # Make passage_dataset bigger by concatenating with itself for testing
        # passage_dataset = concatenate_datasets([passage_dataset for _ in range(500)])
        # print(f"Length of passage dataset: {len(passage_dataset)}")

        logger.info("Clustering passages started")

        k = (
            int(len(passage_dataset["embeddings"]) / train_batch_size)
            if args.kmeans_k == -1
            else args.kmeans_k
        )
        niter = 20
        verbose = not args.silent
        seed = args.manual_seed if args.manual_seed is not None else 42
        embeddings = passage_dataset["embeddings"].numpy()
        d = embeddings.shape[1]

        if args.faiss_clustering:
            use_cuda = True if "cuda" in str(device) else False

            kmeans = faiss.Kmeans(
                d, k, niter=niter, verbose=verbose, seed=seed, gpu=use_cuda
            )
            kmeans.train(embeddings)

            _, indices = kmeans.index.search(embeddings, 1)
            passage_dataset = passage_dataset.add_column(
                "cluster_id", indices.flatten()
            )
        else:
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
        del embeddings
        passage_dataset = passage_dataset.remove_columns(["embeddings"])
    else:
        try:
            logger.info("Reconstructing passage dataset started")
            passage_dataset = concatenate_datasets(passage_dataset)
            k = (
                int(len(passage_dataset["cluster_id"]) / train_batch_size)
                if args.kmeans_k == -1
                else args.kmeans_k
            )
            logger.info("Reconstructing passage dataset completed")
        except ValueError:
            pass

    # Shuffle then sort
    # passage_dataset = passage_dataset.flatten_indices()
    # passage_dataset = passage_dataset.to_iterable_dataset(num_shards=128)
    # passage_dataset = passage_dataset.shuffle(
    #     seed=args.manual_seed
    # )

    logger.info(
        "Converting passage dataset to pandas dataframe to build clustered batches"
    )
    passage_dataset = passage_dataset.to_pandas()
    logger.info("Converting passage dataset to pandas dataframe completed")

    # Sort passage_dataset df by cluster_id
    # passage_dataset = passage_dataset.sort_values(by=["cluster_id"])

    logger.info("Building clustered batches started")
    clusters = passage_dataset["cluster_id"].tolist()
    clustered_batches = {i: [[]] for i in range(k)}
    for i, cluster in enumerate(clusters):
        mini_batch_num = len(clustered_batches[cluster]) - 1
        if len(clustered_batches[cluster][mini_batch_num]) < train_batch_size:
            clustered_batches[cluster][mini_batch_num].append(i)
        else:
            clustered_batches[cluster].append([i])

    # Randomize the mini batches inside a cluster and shuffle items between mini batches of the same cluster
    # This is done to avoid having the same mini batches in the same order in every epoch
    random.seed(args.manual_seed)
    for cluster in clustered_batches.values():
        for mini_batch in cluster:
            random.shuffle(mini_batch)
        random.shuffle(cluster)

    # Move items inside mini batches to other mini batches in the same cluster while keeping the mini batch size constant
    # This is done to avoid having the same mini batches in the same order in every epoch
    for cluster in tqdm(clustered_batches.values(), desc="Randomizing clusters"):
        if len(cluster) == 1:
            continue
        for mini_batch in tqdm(cluster, desc="Randomizing mini batches"):
            for i in range(len(mini_batch)):
                random_mini_batch = random.sample(cluster, 1)[0]
                attempts = 0
                while (
                    len(random_mini_batch) == train_batch_size
                    and attempts <= train_batch_size
                ):
                    random_mini_batch = random.sample(cluster, 1)[0]
                    attempts += 1
                if attempts < train_batch_size:
                    try:
                        random_mini_batch.append(mini_batch.pop())
                    except:
                        logger.info(
                            "Skipping minibatch randomize as all other minibatches in the cluster are full"
                        )

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

    # Randomize each batch in final batches according to clustered_batch_randomize_percentage
    # Here randomize means that a percentage of samples from each batch is moved to other batches and replaced with samples from other batches
    if clustered_batch_randomize_percentage > 0.0:
        random.seed(seed)
        for i, batch in enumerate(final_batches):
            num_samples_to_randomize = int(
                clustered_batch_randomize_percentage * len(batch)
            )
            for _ in range(num_samples_to_randomize):
                random_sample = random.sample(batch, 1)[0]
                random_batch = random.sample(final_batches, 1)[0]
                while len(random_batch) == train_batch_size:
                    random_batch = random.sample(final_batches, 1)[0]
                random_batch.append(random_sample)
                batch.remove(random_sample)

    # if args.save_clustering_idx:
    #     output_dataset_directory = os.path.join(args.output_dir, "clustering_idx")
    #     os.makedirs(output_dataset_directory, exist_ok=True)
    #     current_count = len(os.listdir(output_dataset_directory)) + 1

    #     clustering_idx_save_path = os.path.join(
    #         output_dataset_directory, f"clustering_idx_{current_count}.json"
    #     )
    #     with open(clustering_idx_save_path, "w") as f:
    #         json.dump(clustered_batches, f)

    def select_batch_from_pandas(batch, df):
        batch_dataset = HFDataset.from_pandas(df.iloc[batch].reset_index(drop=True))
        if args.hard_negatives:
            batch_dataset.set_format(
                type="torch",
                columns=[
                    "query_ids",
                    "query_mask",
                    "context_ids",
                    "context_mask",
                    "hard_negative_ids",
                    "hard_negatives_mask",
                    "cluster_id",
                ],
            )
        else:
            batch_dataset.set_format(
                type="torch",
                columns=[
                    "query_ids",
                    "query_mask",
                    "context_ids",
                    "context_mask",
                    "cluster_id",
                ],
            )

        return batch_dataset

    # For testing on large data
    # passage_dataset = datasets.concatenate_datasets([passage_dataset for _ in range(500)])

    clustered_queries = []
    if args.save_clustering_idx:
        batch_datasets = []
        for batch in tqdm(final_batches, desc="Generating batched dataset"):
            # train_batch = passage_dataset.select(batch, keep_in_memory=False)
            train_batch = select_batch_from_pandas(batch, passage_dataset)
            batch_datasets.append(train_batch)
            clustered_queries.append([train_batch["query_text"]])

        output_dataset_directory = os.path.join(args.output_dir, "clustering_idx")
        os.makedirs(output_dataset_directory, exist_ok=True)
        current_count = len(os.listdir(output_dataset_directory)) + 1

        clustering_idx_save_path = os.path.join(
            output_dataset_directory, f"clustering_idx_{current_count}.json"
        )
        with open(clustering_idx_save_path, "w") as f:
            json.dump(clustered_queries, f)
    else:
        batch_datasets = [
            # passage_dataset.select(batch, keep_in_memory=False)
            select_batch_from_pandas(batch, passage_dataset)
            for batch in tqdm(final_batches, desc="Generating batched dataset")
        ]

    # batch_datasets = []
    # for batch in tqdm(final_batches, desc="Generating batched dataset"):
    #     batch_start = batch[0]
    #     batch_end = batch[-1] + 1
    #     batch_datasets.append(passage_dataset[batch_start:batch_end])

    # clustered_batches = sorted(clustered_batches, key=lambda x: len(x), reverse=True)

    # final_batches = []
    # remaining_batches = []
    # for batch in clustered_batches:
    #     if len(batch) == train_batch_size:
    #         final_batches.append(batch)
    #     else:
    #         remaining_batches.append(batch)
    # # for _ in tqdm(range(len(remaining_batches)), desc="Building batches (Max time)"):
    # combined_batches = []
    # for i, batch in enumerate(tqdm(remaining_batches, desc="Building batches")):
    #     if len(batch) == 0:
    #         continue
    #     else:
    #         for j in range(i + 1, len(remaining_batches)):
    #             if (
    #                 len(batch) + len(remaining_batches[j]) <= train_batch_size
    #                 and i != j
    #             ):
    #                 combined_batches.append(batch + remaining_batches[j])
    #                 remaining_batches[j] = []
    #                 remaining_batches[i] = []
    #                 break
    # remaining_batches = combined_batches + [
    #     batch for batch in remaining_batches if len(batch) > 0
    # ]
    # # if not combined_batches:
    # final_batches.extend(
    #     [batch for batch in remaining_batches if len(batch) > 0]
    # )
    # break

    # batch_datasets = [passage_dataset.select(batch) for batch in final_batches]

    # dataset = Dataset.from_generator(my_gen)
    # batch_datasets = HFDataset.from_generator(
    #     lambda: (passage_dataset.select(batch) for batch in clustered_batches)
    # )

    # batch_datasets = [passage_dataset.select(batch) for batch in tqdm(clustered_batches, desc="Building batches")]
    # batch_datasets = []
    # for batch in tqdm(clustered_batches, desc="Generating batched dataset"):
    #     batch_start = batch[0]
    #     batch_end = batch[-1] + 1
    #     batch_datasets.append(passage_dataset[batch_start:batch_end])

    # Use a generator to avoid loading all batches into memory
    # batch_datasets = (passage_dataset.select(batch) for batch in clustered_batches)

    # total_batches = len(batch_datasets)
    # total_length_final = sum([len(batch) for batch in batch_datasets])
    # total_length_initial = len(passage_dataset)
    # logger.info(
    #     f"Total batches: {total_batches}"
    # )
    # logger.info(
    #     f"Total length initial: {total_length_initial}"
    # )
    # logger.info(
    #     f"Total length final: {total_length_final}"
    # )

    if datasets.logging.is_progress_bar_enabled():
        datasets.logging.disable_progress_bar()
        reenable_progress_bar = True
    else:
        reenable_progress_bar = False

    if args.include_bce_loss:
        batch_datasets = [
            batch_dataset.map(
                partial(get_relevant_labels, batch_dataset["context_ids"])
            )
            for batch_dataset in tqdm(batch_datasets, desc="Generating labels")
        ]

    # batch_datasets = [
    #     batch_dataset.map(partial(get_relevant_labels_with_ids, batch_dataset["passage_id"]))
    #     for batch_dataset in tqdm(batch_datasets, desc="Generating labels")
    # ]

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

    logger.info("Building clustered batches completed")

    if reenable_progress_bar:
        datasets.logging.enable_progress_bar()

    return ClusteredDataset(batch_datasets, len(clustered_batches))


def load_trec_file(
    file_name, data_dir=None, header=False, loading_qrels=False, data_format=None
):
    if data_dir:
        if loading_qrels:
            if os.path.exists(os.path.join(data_dir, "qrels", f"{file_name}.tsv")):
                file_path = os.path.join(data_dir, "qrels", f"{file_name}.tsv")
            else:
                raise ValueError(
                    f"{file_name}.tsv or {file_name}.jsonl not found in {data_dir}/qrels"
                )
        else:
            if os.path.exists(os.path.join(data_dir, f"{file_name}.jsonl")):
                file_path = os.path.join(data_dir, f"{file_name}.jsonl")
            elif os.path.exists(os.path.join(data_dir, f"{file_name}.tsv")):
                file_path = os.path.join(data_dir, f"{file_name}.tsv")
            else:
                raise ValueError(
                    f"{file_name}.tsv or {file_name}.jsonl not found in {data_dir}"
                )
    else:
        file_path = file_name

    if not header:
        if file_name in ["qrels", "train", "dev", "test"]:
            column_names = ["query_id", "passage_id", "relevance"]
        elif file_name == "queries":
            column_names = ["query_id", "query_text"]
        elif file_name == "corpus":
            column_names = ["passage_id", "passage_text"]
    else:
        column_names = None

    if file_path.endswith(".tsv"):
        if data_format == "msmarco":
            if loading_qrels:
                column_names = ["query_id", "na", "passage_id", "relevance"]
            dataset = load_dataset(
                "csv",
                data_files=file_path,
                delimiter="\t",
                column_names=column_names,
                # cache_dir=data_dir,
            )
            # Drop na column
            dataset = dataset.remove_columns("na")
        else:
            dataset = load_dataset(
                "csv",
                data_files=file_path,
                delimiter="\t",
                column_names=column_names,
                # cache_dir=data_dir,
                skiprows=1,
            )
    elif file_path.endswith(".jsonl"):
        dataset = load_dataset(
            "json",
            data_files=file_path,
            # cache_dir=data_dir,
        )

    logger.info(f"Loaded {file_path}")

    return dataset


def load_trec_format(
    data_dir=None,
    collection_path=None,
    queries_path=None,
    qrels_path=None,
    collection_header=False,
    queries_header=False,
    qrels_header=False,
    qrels_name=None,
    data_format=None,
    skip_passages=False,
):
    """If data_dir is specified, loads the data from there. Otherwise, loads the data from the specified paths.
    data_dir expects the following structure:
        collection.tsv or collection.jsonl
        queries.tsv or queries.jsonl
        qrels folder containing train.tsv, dev.tsv, and/or test.tsv or jsonl files
    """
    if data_dir is not None:
        if collection_path or queries_path or qrels_path:
            Warning.warn(
                "data_dir is specified. Ignoring collection_path, queries_path, and qrels_path."
            )

        if not skip_passages:
            collection = load_trec_file("corpus", data_dir, collection_header)
        queries = load_trec_file("queries", data_dir, queries_header)
        qrels = load_trec_file(
            qrels_name,
            data_dir,
            qrels_header,
            loading_qrels=True,
            data_format=data_format,
        )

    else:
        if not collection_path or not queries_path or not qrels_path:
            raise ValueError(
                "data_dir is not specified. Please specify collection_path, queries_path, and qrels_path."
            )

        if not skip_passages:
            collection = load_trec_file(collection_path, header=collection_header)
        queries = load_trec_file(queries_path, header=queries_header)
        qrels = load_trec_file(qrels_path, header=qrels_header)

    # Also check if an index exists

    return (
        None if skip_passages else collection["train"],
        queries["train"],
        qrels["train"],
    )


def convert_beir_columns_to_trec_format(
    collection,
    queries,
    qrels,
    include_titles=False,
):
    collection = collection.rename_column("_id", "passage_id")
    if include_titles:
        collection = collection.map(
            lambda row: {"text": row["title"] + " " + row["text"]}
        )
    collection = collection.rename_column("text", "passage_text")
    # queries = queries.rename_column("_id", "query_id")
    # queries = queries.rename_column("text", "query_text")

    try:
        collection = collection.remove_columns("metadata")
    except:
        pass

    return collection, queries, qrels


def embed_passages_trec_format(
    passage_dataset,
    encoder,
    tokenizer,
    args,
    context_config,
    device,
):
    if isinstance(passage_dataset, str):
        # If passage_dataset is a str, then we load from disk.
        logger.info(f"Loading passage dataset from {passage_dataset}")
        passage_data = load_from_disk(passage_dataset)
        index_path = os.path.join(passage_dataset, "hf_dataset_index.faiss")
        if os.path.isfile(index_path):
            passage_data.load_faiss_index("embeddings", index_path)
            passage_dataset = passage_data
        else:
            logger.info("Adding FAISS index to evaluation passages")
            index
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

    else:
        if args.fp16:
            from torch.cuda import amp
        else:
            amp = None

        encoder = encoder.to(device)

        logger.info("Generating embeddings for evaluation passages")
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
                passage_column="passage_text",
                unified_rr=args.unified_rr,
            ),
            batched=True,
            batch_size=args.embed_batch_size,
        )
        logger.info("Generating embeddings for evaluation passages completed.")

        if args.save_passage_dataset:
            output_dataset_directory = os.path.join(args.output_dir, "passage_dataset")
            os.makedirs(output_dataset_directory, exist_ok=True)
            passage_dataset.save_to_disk(output_dataset_directory)

        logger.info("Adding FAISS index to evaluation passages")
        index = get_faiss_index(args)
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
    return passage_index


def compute_rerank_similarity(query_embeddings, doc_dicts, passage_column="passages"):
    """
    Computes the similarity between the reranking query embeddings and the reranking document embeddings
    for the unified reranking method using dot product.
    """
    rerank_similarity = np.zeros((len(doc_dicts), len(doc_dicts[0][passage_column])))
    for i, doc_dict in enumerate(doc_dicts):
        rerank_similarity[i] = np.dot(
            query_embeddings[i], np.array(doc_dict["rerank_embeddings"]).T
        )
    return rerank_similarity


class RetrievalOutput:
    def __init__(
        self,
        loss,
        context_outputs,
        query_outputs,
        correct_predictions_count,
        correct_predictions_percentage=None,
        reranking_context_outputs=None,
        reranking_query_outputs=None,
        reranking_loss=None,
        nll_loss=None,
        teacher_correct_predictions_percentage=None,
        reranking_correct_predictions_percentage=None,
    ):
        self.loss = loss
        self.context_outputs = context_outputs
        self.query_outputs = query_outputs
        self.correct_predictions_count = correct_predictions_count
        self.correct_predictions_percentage = correct_predictions_percentage
        self.reranking_context_outputs = reranking_context_outputs
        self.reranking_query_outputs = reranking_query_outputs
        self.reranking_loss = reranking_loss
        self.nll_loss = nll_loss
        self.teacher_correct_predictions_percentage = (
            teacher_correct_predictions_percentage
        )
        self.reranking_correct_predictions_percentage = (
            reranking_correct_predictions_percentage
        )


class MarginMSELoss(nn.Module):
    def __init__(self, margin=1.0):
        super(MarginMSELoss, self).__init__()
        self.margin = margin

    def forward(self, scores, labels):
        """
        A Margin-MSE loss variant for matrices of relevance scores and labels.
        """
        diff_scores = scores.unsqueeze(1) - scores.unsqueeze(2)
        diff_labels = labels.unsqueeze(1) - labels.unsqueeze(2)
        margin_diff = diff_labels - self.margin

        loss = torch.mean(torch.pow(diff_scores - margin_diff, 2))
        return loss


def colbert_score(teacher_model, query_inputs, context_inputs, device):
    Q_vectors, D_vectors = teacher_model(query_inputs, context_inputs)

    scores = torch.zeros(len(Q_vectors), len(D_vectors), device=device)

    def score(Q, D):
        return (Q @ D.permute(0, 2, 1)).max(2).values.sum(1)
        # return teacher_model.score(Q, D)

    for i, q_vec in enumerate(Q_vectors):
        scores[i, :] = score(q_vec, D_vectors)

    return scores


def cross_encoder_score(teacher_model, query_inputs, context_inputs, device):
    # Build input tensors
    pass


def get_tas_dataset(
    passage_dataset,
    train_batch_size,
    encoder,
    tokenizer,
    args,
    device,
):
    logger.info("Generating embeddings for clustering started")

    if args.fp16:
        from torch.cuda import amp
    else:
        amp = None

    encoder = encoder.to(device)

    if "embeddings" not in passage_dataset.column_names:
        if args.cluster_queries:
            logger.info("Generating embeddings for query clustering started")
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
                    pretokenized=False,
                    cluster_concatenated=args.cluster_concatenated,
                    passage_column="query_text",
                ),
                batched=True,
                batch_size=args.embed_batch_size,
                with_rank=args.n_gpu > 1,
                num_proc=args.n_gpu,
                # cache_file_name=os.path.join(args.dataset_cache_dir, args.output_dir, "/clustering_embeddings.cache"),
            )
            logger.info("Generating embeddings for query clustering completed")

        else:
            logger.info("Generating embeddings for passage clustering started")
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
                    pretokenized=False,
                    cluster_concatenated=args.cluster_concatenated,
                ),
                batched=True,
                batch_size=args.embed_batch_size,
                with_rank=args.n_gpu > 1,
                num_proc=args.n_gpu,
                # cache_file_name=os.path.join(args.dataset_cache_dir, args.output_dir, "/clustering_embeddings.cache"),
            )
            passage_dataset = passage_dataset.rename_column("passages", "gold_passage")
            logger.info("Generating embeddings for passage clustering completed")

    logger.info("Clustering passages started")

    k = (
        int(len(passage_dataset["embeddings"]) / train_batch_size)
        if args.kmeans_k == -1
        else args.kmeans_k
    )
    niter = 20
    verbose = not args.silent
    seed = args.manual_seed if args.manual_seed is not None else 42
    embeddings = passage_dataset["embeddings"].numpy()
    d = embeddings.shape[1]

    if args.faiss_clustering:
        use_cuda = True if "cuda" in str(device) else False

        kmeans = faiss.Kmeans(
            d, k, niter=niter, verbose=verbose, seed=seed, gpu=use_cuda
        )
        kmeans.train(embeddings)

        _, indices = kmeans.index.search(embeddings, 1)
        passage_dataset = passage_dataset.add_column("cluster_id", indices.flatten())
    else:
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

    # Random shuffle passage_dataset
    passage_dataset = passage_dataset.shuffle(seed=seed)

    # Sort passage_dataset by cluster_id
    # passage_dataset = passage_dataset.sort("cluster_id")

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

    # Split any batches bigger than train_batch_size into smaller batches until all batches are of size train_batch_size or smaller
    final_batches = []
    for batch in clustered_batches:
        if len(batch) <= train_batch_size:
            final_batches.append(batch)
        else:
            for i in range(0, len(batch), train_batch_size):
                final_batches.append(batch[i : i + train_batch_size])

    clustered_queries = []
    if args.save_clustering_idx:
        batch_datasets = []
        for batch in tqdm(final_batches, desc="Generating batched dataset"):
            train_batch = passage_dataset.select(batch, keep_in_memory=False)
            batch_datasets.append(train_batch)
            clustered_queries.append([train_batch["query_text"]])

        output_dataset_directory = os.path.join(args.output_dir, "clustering_idx")
        os.makedirs(output_dataset_directory, exist_ok=True)
        current_count = len(os.listdir(output_dataset_directory)) + 1

        clustering_idx_save_path = os.path.join(
            output_dataset_directory, f"clustering_idx_{current_count}.json"
        )
        with open(clustering_idx_save_path, "w") as f:
            json.dump(clustered_queries, f)
    else:
        batch_datasets = [
            passage_dataset.select(batch, keep_in_memory=False)
            for batch in tqdm(final_batches, desc="Generating batched dataset")
        ]

    return ClusteredDataset(batch_datasets, len(clustered_batches))


class MovingLossAverage:
    def __init__(self, window_size):
        self.window_size = window_size
        self.losses = deque(maxlen=window_size)

    def add_loss(self, loss):
        self.losses.append(loss)

    def get_average_loss(self):
        return sum(self.losses) / len(self.losses)

    def size(self):
        return len(self.losses)


def get_faiss_index(args):
    if args.faiss_index_type == "IndexHNSWFlat":
        index = faiss.IndexHNSWFlat(
            args.faiss_d, args.faiss_m, faiss.METRIC_INNER_PRODUCT
        )
    elif args.faiss_index_type == "IndexFlatIP":
        index = faiss.IndexFlatIP(args.faiss_d)
    else:
        raise ValueError(
            f"faiss_index_type {args.faiss_index_type} is not supported. Please use IndexHNSWFlat or IndexFlatIP."
        )
    return index


def calculate_mrr(qrels, results, k_values=(10, 100, 1000)):
    MRR = {}

    for k in k_values:
        MRR[f"MRR@{k}"] = 0.0

    k_max, top_hits = max(k_values), {}

    for query_id, doc_scores in results.items():
        top_hits[query_id] = sorted(
            doc_scores.items(), key=lambda item: item[1], reverse=True
        )[0:k_max]

    for query_id in top_hits:
        query_relevant_docs = set(
            [doc_id for doc_id in qrels[str(query_id)] if qrels[query_id][doc_id] > 0]
        )
        for k in k_values:
            for rank, hit in enumerate(top_hits[query_id][0:k]):
                if hit[0] in query_relevant_docs:
                    MRR[f"MRR@{k}"] += 1.0 / (rank + 1)
                    break

    for k in k_values:
        MRR[f"MRR@{k}"] = round(MRR[f"MRR@{k}"] / len(qrels), 5)

    return MRR
