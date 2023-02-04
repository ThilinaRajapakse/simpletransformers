from functools import partial

import torch
import numpy as np
from tqdm.auto import tqdm

from simpletransformers.retrieval.retrieval_utils import get_output_embeddings, embed
from datasets import Dataset as HFDataset


class BeirRetrievalModel:
    def __init__(
        self,
        context_encoder,
        query_encoder,
        context_tokenizer,
        query_tokenizer,
        context_config,
        query_config,
        args,
        **kwargs
    ):
        self.context_encoder = context_encoder
        self.query_encoder = query_encoder
        self.context_tokenizer = context_tokenizer
        self.query_tokenizer = query_tokenizer
        self.context_config = context_config
        self.query_config = query_config
        self.args = args

    # Write your own encoding query function (Returns: Query embeddings as numpy array)
    def encode_queries(self, queries, **kwargs):
        # def encode_queries(self, queries: List[str], batch_size: int, **kwargs) -> np.ndarray:

        query_model = self.query_encoder
        query_model.to(query_model.device)
        query_config = self.query_config
        query_tokenizer = self.query_tokenizer

        if self.args.larger_representations:
            all_query_embeddings = np.zeros(
                (
                    len(queries),
                    query_config.hidden_size * (1 + self.args.extra_cls_token_count),
                )
            )
        else:
            all_query_embeddings = np.zeros(
                (
                    len(queries),
                    query_config.hidden_size
                    if "projection_dim" not in query_config.to_dict()
                    or not query_config.projection_dim
                    else query_config.projection_dim,
                )
            )

        # if self.args.n_gpu > 1:
        #     query_model = torch.nn.DataParallel(query_model)

        if self.args.fp16:
            from torch.cuda import amp

        query_model.eval()

        # Batching
        for i, batch in tqdm(
            enumerate(
                [
                    queries[i : i + self.args.eval_batch_size]
                    for i in range(0, len(queries), self.args.eval_batch_size)
                ]
            ),
            desc="Generating query embeddings",
            disable=self.args.silent,
            total=len(queries) // self.args.eval_batch_size,
        ):
            query_batch = query_tokenizer(
                batch,
                max_length=self.args.max_seq_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )

            query_inputs = {
                "input_ids": query_batch["input_ids"].to(query_model.device),
                "attention_mask": query_batch["attention_mask"].to(query_model.device),
            }

            with torch.no_grad():
                if self.args.fp16:
                    with amp.autocast():
                        query_outputs = query_model(**query_inputs)
                        query_outputs = get_output_embeddings(
                            query_outputs,
                            concatenate_embeddings=self.args.larger_representations
                            and self.args.model_type == "custom",
                            n_cls_tokens=(1 + self.args.extra_cls_token_count),
                        )
                else:
                    query_outputs = query_model(**query_inputs)
                    query_outputs = get_output_embeddings(
                        query_outputs,
                        concatenate_embeddings=self.args.larger_representations
                        and self.args.model_type == "custom",
                        n_cls_tokens=(1 + self.args.extra_cls_token_count),
                    )

            all_query_embeddings[
                i * self.args.eval_batch_size : (i + 1) * self.args.eval_batch_size
            ] = (query_outputs.cpu().detach().numpy())

        return np.array(all_query_embeddings)

    # Write your own encoding corpus function (Returns: Document embeddings as numpy array)
    def encode_corpus(self, corpus, **kwargs):
        # def encode_corpus(self, corpus: List[Dict[str, str]], batch_size: int, **kwargs) -> np.ndarray:
        args = self.args
        encoder = self.context_encoder
        device = encoder.device
        tokenizer = self.context_tokenizer
        passages = [c["title"] + " " + c["text"] for c in corpus]

        prediction_passages_dataset = HFDataset.from_dict({"passages": passages})

        if "embeddings" not in prediction_passages_dataset.column_names:
            if args.fp16:
                from torch.cuda import amp
            else:
                amp = None

            encoder = encoder.to(device)
            encoder.eval()
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
                # with_rank=args.n_gpu > 1,
                # num_proc=args.n_gpu,
            )

        return np.array(prediction_passages_dataset["embeddings"])
