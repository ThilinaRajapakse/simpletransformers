---
title: Retrieval Model
permalink: /docs/retrieval-model/
excerpt: "RetrievalModel for Retrieval tasks."
last_modified_at: 2022/02/25 17:17:17
toc: true
---


## `RetrievalModel`

The `RetrievalModel` class is used for Retrieval tasks.


The following parameters can be used to initialize a `RetrievalModel`. Note that it may not be necessary to specify all of them.:
- `model_type` should be a supported model type for `RetrievalModel`. (Currently, only `dpr` is supported)
- `model_name` specifies the exact architecture and trained weights to use for the full model. This may be a Hugging Face Transformers compatible pre-trained model, a community model, or the path to a directory containing model files.

    **Note:** The context encoder and query encoder must be specified explicitly when initializing a pretrained DPR model from HuggingFace. `model_name` can be `None` in this case. <br/><br/>
    Setting `model_name` is useful when you want to load a `RetrievalModel` that was saved with Simple Transformers. In this case, you do not need to specify the context/query parameters.
    {: .notice--info}

    **Note:** For a list of community models, see [here](https://huggingface.co/models).
    {: .notice--info}

- `context_encoder_name` specifies the exact architecture and trained weights to use for the context encoder model. This may be a Hugging Face Transformers compatible pre-trained model, a community model, or the path to a directory containing model files.
- `query_encoder_name` specifies the exact architecture and trained weights to use for the query encoder model. This may be a Hugging Face Transformers compatible pre-trained model, a community model, or the path to a directory containing model files.
- `context_encoder_tokenizer` specifies the tokenizer to use for the context encoder. This may be a Hugging Face Transformers compatible pre-trained tokenizer, a community tokenizer, or the path to a directory containing tokenizer files.
- `query_encoder_tokenizer` specifies the tokenizer to use for the query encoder. This may be a Hugging Face Transformers compatible pre-trained tokenizer, a community tokenizer, or the path to a directory containing tokenizer files.

    **Note:** It is not necessary to provide `context_encoder_tokenizer` and `query_encoder_tokenizer` if `context_encoder_name` and `query_encoder_name` are specified.
    {: .notice--info}

    **Note:** `context_encoder_name` and `query_encoder_name` can be specified if you wish to train a context encoder and/or query encoder from scratch while still using the pretrained tokenizers.
    {: .notice--info}


### Initializing the pretrained DPR model

```python
from simpletransformers.retrieval import RetrievalModel


model_type = "dpr"
context_encoder_name = "facebook/dpr-ctx_encoder-single-nq-base"
question_encoder_name = "facebook/dpr-question_encoder-single-nq-base"

model = RetrievalModel(
    model_type=model_type,
    context_encoder_name=context_encoder_name,
    query_encoder_name=question_encoder_name,
)

```

### Initializing a RetrievalModel trained with Simple Transformers

```python
from simpletransformers.retrieval import RetrievalModel


model_type = "dpr"
model_name = "path/to/model"

# Initialize a RetrievalModel
model = RetrievalModel(
    model_type=model_type,
    model_name=model_name,
)

```

### Initializing a RetrievalModel without pretrained weights but with pretrained tokenizers

```python
from simpletransformers.retrieval import RetrievalModel


model_type = "dpr"
context_encoder_tokenizer = "facebook/dpr-ctx_encoder-single-nq-base"
question_encoder_tokenizer = "facebook/dpr-question_encoder-single-nq-base"

# Initialize a RetrievalModel
model = RetrievalModel(
    model_type=model_type,
    context_encoder_tokenizer=context_encoder_tokenizer,
    query_encoder_tokenizer=question_encoder_tokenizer,
)


```



### Configuring a `RetrievalModel`

`RetrievalModel` has the following task-specific configuration options.

| Argument                                   | Type | Default         | Description                                                                                                  |
| ------------------------------------------ | ---- | --------------- | ------------------------------------------------------------------------------------------------------------ |
| context_config                             | dict | {}              | Configuration for the context encoder model                                                                  |
| embed_batch_size                           | int  | 16              | Batch size to use when generating context embeddings                                                         |
| faiss_index_type                           | str  | `"IndexFlatIP"` | The type of FAISS index to use. `IndexFlatIP` and  `IndexHNSWFlat` are currently supported                   |
| hard_negatives                             | bool | False           | Whether to use hard negatives during training. `hard_negative` column must be present in training data.      |
| include_title                              | bool | True            | If True, the `title` column will be prepended to the passages.                                               |
| query_config                               | dict | {}              | Configuration for the query encoder model                                                                    |
| remove_duplicates_from_additional_passages | bool | False           | Whether to remove duplicate passages in additional_passages used for evaluation. Note that this can be slow. |
| retrieval_batch_size                       | int  | 512             | Batch size to use when retrieving documents from index                                                       |
| retrieve_n_docs                            | int  | 10              | Number of documents to be retrieved when doing retrieval tasks (e.g. `evaluate_model()`, `predict()`)        |
| save_passage_dataset                       | bool | True            | Save passage datasets (during evaluation and prediction) to disk.                                            |
| use_hf_datasets                            | bool | True            | Use Huggingface Datasets for lazy loading of data. Must be set to True for `RetrievalModel`.                 |


**Note:** For configuration options common to all Simple Transformers models, please refer to the [Configuring a Simple Transformers Model section](/docs/usage/#configuring-a-simple-transformers-model).
{: .notice--info}


## `Class RetrievalModel`

> *simpletransformers.retrieval.RetrievalModel*{: .function-name}(self, model_type=None, model_name=None, context_encoder_name=None, query_encoder_name=None, context_encoder_tokenizer=None, query_encoder_tokenizer=None, prediction_passages=config, args=None, use_cuda=True, cuda_device=-1, **kwargs,)

Initializes a RetrievalModel model.
{: .function-text}

> Parameters
{: .parameter-blockquote}

* **model_type** *(`str`, optional)* - The type of model architecture. Defaults to None.

* **model_name** *(`str`, optional)* - The exact architecture and trained weights to use. This may be a Hugging Face Transformers compatible pre-trained model, a community model, or the path to a directory containing model files.

* **context_encoder_name** *(`str`, optional)* - The exact architecture and trained weights to use for the context encoder. This may be a Hugging Face Transformers compatible pre-trained model, a community model, or the path to a directory containing model files.

* **query_encoder_name** *(`str`, optional)* - The exact architecture and trained weights to use for the query encoder. This may be a Hugging Face Transformers compatible pre-trained model, a community model, or the path to a directory containing model files.

* **context_encoder_tokenizer** *(`str`, optional)* - The tokenizer to use for the context encoder. This may be a Hugging Face Transformers compatible pre-trained tokenizer, a community tokenizer, or the path to a directory containing tokenizer files. Defaults to None.

* **query_encoder_tokenizer** *(`str`, optional)* - The tokenizer to use for the query encoder. This may be a Hugging Face Transformers compatible pre-trained tokenizer, a community tokenizer, or the path to a directory containing tokenizer files. Defaults to None.

* **prediction_passages** *(`dict`, optional)* - The passages to be used as the corpus for retrieval when making predictions. Provide this only when using the model for predictions. Defaults to None.

* **args** *(`dict`, optional)* - [Default args](/docs/usage/#configuring-a-simple-transformers-model) will be used if this parameter is not provided. If provided, it should be a dict containing the args that should be changed in the default args or a `RetrievalArgs` object.

* **use_cuda** *(`bool`, optional)* - Use GPU if available. Setting to False will force model to use CPU only. (See [here](/docs/usage/#to-cuda-or-not-to-cuda))

* **cuda_device** *(`int`, optional)* - Specific GPU that should be used. Will use the first available GPU by default. (See [here](/docs/usage/#selecting-a-cuda-device))

* **kwargs** *(optional)* - For providing proxies, force_download, resume_download, cache_dir and other options specific to the 'from_pretrained' implementation where this will be supplied. (See [here](/docs/usage/#options-for-downloading-pre-trained-models))
{: .parameter-list}

> Returns
{: .returns}

* `None`
{: .return-list}


## Training a `RetrievalModel`

The `train_model()`  method is used to train the model.

```python
model.train_model(train_data)
```

> *simpletransformers.retrieval.RetrievalModel.train_model*{: .function-name}(self, train_data, output_dir=None, show_running_loss=True, args=None, eval_data=None, additional_eval_passages=None, verbose=True, **kwargs)

Trains the model using 'train_data'
{: .function-text}

> Parameters
{: .parameter-blockquote}

* **train_data** - Pandas DataFrame containing the 3 columns - `query_text`, `gold_passage`, and `title` (Title is optional). If `use_hf_datasets` is True, then this may also be the path to a TSV file with the same columns.
    - `query_text`: The query text sequence
    - `gold_passage`: The gold passage text sequence
    - `title`: The title of the gold passage

* **output_dir** *(`str`, optional)* - The directory where model files will be saved. If not given, `self.args['output_dir']` will be used.

* **show_running_loss** *(`bool`, optional)* - If True, the running loss (training loss at current step) will be logged to the console.

* **args** *(`dict`, optional)* - A dict of configuration options for the `RetrievalModel`. Any changes made will persist for the model.

* **eval_data** *(optional)* - Evaluation data (same format as train_data) against which evaluation will be performed when `evaluate_during_training` is enabled. Is required if `evaluate_during_training` is enabled.

* **additional_eval_passages** *(optional)* - Additional passages to be used during evaluation.
This may be a list of passages, a pandas DataFrame with the column `passages`, or a TSV file with the column `passages`.

* **kwargs** *(optional)* - Additional metrics that should be calculated. Pass in the metrics as keyword arguments *(name of metric: function to calculate metric)*. Refer to the [additional metrics](/docs/tips-and-tricks/#additional-evaluation-metrics) section.
E.g. `f1=sklearn.metrics.f1_score`.
A metric function should take in two parameters. The first parameter will be the true labels, and the second parameter will be the predictions.
{: .parameter-list}

> Returns
{: .returns}

* global_step: Number of global steps trained
* training_details: Average training loss if evaluate_during_training is False or full training progress scores if evaluate_during_training is True
{: .return-list}

**Note:** For more details on evaluating Retrieval models with custom metrics, please refer to the [Evaluating Generated Sequences](/docs/retrieval-specifics/#evaluating-generated-sequences) section.
{: .notice--info}

**Note:** For more details on training models with Simple Transformers, please refer to the [Tips and Tricks](/docs/tips-and-tricks) section.
{: .notice--info}


## Evaluating a `RetrievalModel`

The `eval_model()`  method is used to evaluate the model.

The following metrics will be calculated by default:

* `eval_loss` - Model loss over the evaluation data


```python
result = model.eval_model(eval_data)
```

> *simpletransformers.retrieval.RetrievalModel.eval_model*{: .function-name}(self, eval_data, evaluate_with_all_passages=True, additional_passages=None, top_k_values=None, retrieve_n_dics=None, return_doc_dicts=True, passage_dataset=None,
> output_dir=None, verbose=True, silent=False, **kwargs)

Evaluates the model using 'eval_data'
{: .function-text}

> Parameters
{: .parameter-blockquote}

* **eval_data** - Pandas DataFrame containing the 3 columns - `query_text`, `gold_passage`, and `title` (Title is optional). If `use_hf_datasets` is True, then this may also be the path to a TSV file with the same columns.
    - `query_text`: The query text sequence
    - `gold_passage`: The gold passage text sequence
    - `title`: The title of the gold passage

* **evaluate_with_all_passages** *(`bool`, optional)* - If True, evaluate with all passages. If False, evaluate only with in-batch negatives.

* **additional_passages** *(optional)* - Additional passages to be used during evaluation.
This may be a list of passages, a pandas DataFrame with the column `passages`, or a TSV file with the column `passages`.

* **top_k_values** *(`list`, optional)* - List of top-k values to be used for evaluation metrics.

* **retrieve_n_docs** *(`int`, optional)* - Number of documents to retrieve for each query. Overrides `args.retrieve_n_docs` for this evaluation.

* **return_doc_dicts** *(`bool`, optional)* - If True, return the doc dicts for the retrieved passages. This is always `True` when `evaluate_with_all_passages` is True.

* **passage_dataset** *(`str`, optional)* - Path to a saved Huggingface dataset (containing generated embeddings) for both the eval_data and additional passages

* **output_dir** *(`str`, optional)* - The directory where model files will be saved. If not given, `self.args['output_dir']` will be used.

* **verbose** *(`bool`, optional)* - If verbose, results will be printed to the console on completion of evaluation.

* **silent** *(`bool`, optional)* - If silent, tqdm progress bars will be hidden.

* **kwargs** *(optional)* - Additional metrics that should be calculated. Pass in the metrics as keyword arguments *(name of metric: function to calculate metric)*. Refer to the [additional metrics](/docs/tips-and-tricks/#additional-evaluation-metrics) section.
E.g. `f1=sklearn.metrics.f1_score`.
A metric function should take in two parameters. The first parameter will be the true labels, and the second parameter will be the predictions.
{: .parameter-list}

> Returns
{: .returns}

* **result** *(`dict`)* - Dictionary containing evaluation results.

* **doc_ids** *(`dict`)* - List of retrieved document ids for each query. (Shape: `(num_queries, retrieve_n_docs)`)

* **doc_vectors** *(`dict`)* - List of retrieved document vectors for each query. (Shape: `(num_queries, retrieve_n_docs, embedding_dim)`)

* **doc_dicts** *(`dict`)* - List of retrieved document dicts for each query. (Shape: `(num_queries, retrieve_n_docs)`)
{: .return-list}


**Note:** For more details on evaluating models with Simple Transformers, please refer to the [Tips and Tricks](/docs/tips-and-tricks) section.
{: .notice--info}

## Making Predictions With a `RetrievalModel`

The `predict()`  method is used to make predictions with the model.

```python
to_predict = [
    'Who was the author of "Dune"?'
]

predicted_passages, doc_ids, doc_vectors, doc_dicts = model.predict(to_predict)
```

**Note:** The input **must** be a List even if there is only one sentence.
{: .notice--info}


> *simpletransformers.retrieval.RetrievalModel.predict*{: .function-name}(to_predict, prediction_passages=None, retrieve_n_docs=None)

Performs predictions on a list of text `to_predict`.
{: .function-text}

> Parameters
{: .parameter-blockquote}

* **to_predict** *(`list`)*- A python list of text (str) to be sent to the model for prediction.

* **prediction_passages** *(`Union[str, DataFrame]`, optional)*- Path to a directory containing a passage dataset, a JSON/TSV file containing the passages, or a Pandas DataFrame. Defaults to None.

* **to_predict** *(`int`, optional)*- Number of docs to retrieve per query. Defaults to None.
{: .parameter-list}

> Returns
{: .returns}

* **passages** *(`list`)* - List of lists containing the retrieved passages per query. (Shape: `(len(to_predict), retrieve_n_docs)`)

* **doc_ids** *(`list`)* - List of lists containing the retrieved doc ids per query. (Shape: `(len(to_predict), retrieve_n_docs)`)

* **doc_vectors** *(`list`)* - List of lists containing the retrieved doc vectors per query. (Shape: `(len(to_predict), retrieve_n_docs)`)

* **doc_dicts** *(`list`)* - List of dicts containing the retrieved doc dicts per query.

{: .return-list}

**Note:** A corpus/index of passages can be specified using the `prediction_passages` parameter when initializing a model for prediction. Any future `predict()` calls will then use this index.
{: .notice--info}
