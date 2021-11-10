---
title: Retrieval Specifics
permalink: /docs/retrieval-specifics/
excerpt: "Specific notes for Retrieval tasks."
last_modified_at: 2021/11/10 21:14:59
toc: true
---

Retrieval models (`RetrievalModel`) are models used to retrieve relevant documents from a corpus given a query.

Currently, only [DPR](https://arxiv.org/abs/2004.04906) models are supported.


## Usage Steps

Using a retrieval model in Simple Transformers follows the [standard pattern](/docs/usage/#task-specific-models).
1. Initialize a `RetrievalModel`
2. Train the model with `train_model()`
3. Evaluate the model with `eval_model()`
4. Make predictions on (unlabelled) data with `predict()`


**Note:** You must have Faiss (GPU or CPU) installed to use RetrievalModel.
Faiss installation instructions can be found [here](https://github.com/facebookresearch/faiss/blob/master/INSTALL.md).
{: .notice--warning}

### Initializing a `RetrievalModel`

The `__init__` arguments for a `RetrievalModel` are a little different from the common format found in the other models. Please refer [here](/docs/retrieval-model/#retrieval-model) for more information.


## Hard Negatives

### Why hard negatives are needed

In dense passage retrieval, the model is typically trained using the in-batch negatives technique which makes the training process much more computationally efficient. The process is quickly outlined below.

For a batch consisting of query and positive passage pairs:

1. Compute the query encodings for each query in the batch.
2. Compute the passage encodings for each positive passage in the batch.
3. Calculate the cosine similarity between each query and all passages in the batch.
4. Optimize for the negative log likelihood of the positive passage for each query.

For more information, refer to the [DPR paper](https://arxiv.org/abs/2004.04906).

While this method is computationally efficient, it is not ideal for training a good retrieval model as the negative samples are chosen at random (batches are randomly sampled). The model can be improved further by training with hard negatives, i.e. passages which might be similar but not the same as the positive passage.

Here, the batch would contain triplets of queries, positive passages, and hard negative passages. Each query embedding would then be compared against the embeddings of all positive passages of the other queries (in-batch negatives) as well as all the hard negatives from each query.

### How to train with hard negatives

In order to train a `RetrievalModel` with hard negatives, the training data must contain a `"hard_negatives"` column containing a hard negative example for each query.

**Note:** You must set `hard_negatives` to `True` in the model args in order for the model to include the hard negatives in training. The extra passage per query increases the size of the batch so you may need to decrease the batch size to avoid running out of memory.
{: .notice--warning}

The hard negative passages may be obtained by external methods (such as BM25 sparse retrieval). However, Simple Transformers offers a method, `build_hard_negatives()`, to generate hard negatives from a given passage dataset. For example, if you are finetuning a DPR model on your own data, you can use the `build_hard_negatives()` function to generate hard negatives from your corpus and a pre-trained DPR model.

```python
import logging

import pandas as pd

from simpletransformers.retrieval import RetrievalModel, RetrievalArgs


logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)


queries = [
    "Where does solar energy come from?",
    "What is anthropology a study of?",
    "In what fields of science is the genome studied?",
]

# Note that the passages have been manually truncated for this example.
# Typically, you would want to use the full passage.
passages = [
    "Solar energy is radiant light and heat...",
    "describes the workings of societies around the world...",
    "The genome includes both the genes and the non-coding sequences of the DNA/RNA....",
    "the genome is the genetic material of an organism",
    "Its main subdivisions are social anthropology and cultural anthropology",
    "Neptune is the eighth and farthest known planet from the Sun in the Solar System"
]


model_type = "dpr"
context_name = "facebook/dpr-ctx_encoder-single-nq-base"
query_name = "facebook/dpr-question_encoder-single-nq-base"

model_args = RetrievalArgs()

# Create a TransformerModel
model = RetrievalModel(
    model_type=model_type,
    context_encoder_name=context_name,
    query_encoder_name=query_name,
    args=model_args
)

# The hard negatives will be written to the output dir by default.
hard_df = model.build_hard_negatives(
    queries=queries,
    passage_dataset=passages,
    retrieve_n_docs=1
)

print(hard_df)

```

You can combine the hard negatives with the queries and their positive passages to create training data with hard negatives.

