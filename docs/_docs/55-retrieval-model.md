---
title: Retrieval Model
permalink: /docs/retrieval-model/
excerpt: "RetrievalModel for Retrieval tasks."
last_modified_at: 2021/10/03 22:50:10
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
context_name = "facebook/dpr-ctx_encoder-single-nq-base"
question_name = "facebook/dpr-question_encoder-single-nq-base"

# Initialize a RetrievalModel
model = RetrievalModel(
    model_type=model_type,
    context_encoder_tokenizer=context_encoder_tokenizer,
    query_encoder_tokenizer=question_encoder_tokenizer,
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

| Argument                    | Type    | Default | Description                                                                                                                                   |
| --------------------------- | ------- | ------- | --------------------------------------------------------------------------------------------------------------------------------------------- |
| context_config                    | dict     | {}   | Configuration for the context encoder model                                                                           |
| query_config                    | dict     | {}   | Configuration for the query encoder model                                                                           |
| embed_batch_size                    | int     | 16   | Batch size to use when generating context embeddings                                                                           |
| hard_negatives                    | bool     | False   | Whether to use hard negatives during training. `hard_negative` column must be present in training data.                                                                           |
| retrieve_n_docs                    | int     | 10   | Number of documents to be retrieved when doing retrieval tasks (e.g. `evaluate_model()`, `predict()`)                                                                           |
| remove_duplicates_from_additional_passages                    | bool     | False   | Whether to remove duplicate passages in additional_passages used for evaluation. Note that this can be slow.                                                                           |
| retrieval_batch_size                    | int     | 16   | Batch size to use when retrieving documents from index                                                                          |
| save_passage_dataset                    | bool     | True   | Save passage datasets (during evaluation and prediction) to disk.                                                                           |
| use_hf_datasets                    | bool     | True   | Use Huggingface Datasets for lazy loading of data. Must be set to True for `RetrievalModel`.                                                                  |


```python
from simpletransformers.retrieval import RetrievalModel, RetrievalArgs


model_args = RetrievalArgs()
model_args.num_train_epochs = 3

model_type = "dpr"
context_name = "facebook/dpr-ctx_encoder-single-nq-base"
question_name = "facebook/dpr-question_encoder-single-nq-base"

model = RetrievalModel(
    model_type=model_type,
    context_encoder_tokenizer=context_encoder_tokenizer,
    query_encoder_tokenizer=question_encoder_tokenizer,
    args=model_args,
)

```

**Note:** For configuration options common to all Simple Transformers models, please refer to the [Configuring a Simple Transformers Model section](/docs/usage/#configuring-a-simple-transformers-model).
{: .notice--info}


## `Class RetrievalModel`

> *simpletransformers.retrieval.RetrievalModel*{: .function-name}(self, encoder_type=None, encoder_name=None, decoder_name=None, encoder_decoder_type=None, encoder_decoder_name=None, config=config, args=None, use_cuda=True, cuda_device=-1, **kwargs,)

Initializes a RetrievalModel model.
{: .function-text}

> Parameters
{: .parameter-blockquote}

* **encoder_type** *(`str`, optional)* - The type of model to use as the encoder.

* **encoder_name** *(`str`, optional)* - The exact architecture and trained weights to use. This may be a Hugging Face Transformers compatible pre-trained model, a community model, or the path to a directory containing model files.

* **decoder_name** *(`str`, optional)* - The exact architecture and trained weights to use. This may be a Hugging Face Transformers compatible pre-trained model, a community model, or the path to a directory containing model files.

* **encoder_decoder_type** *(`str`, optional)* - The type of encoder-decoder model. (E.g. bart)

* **encoder_decoder_name** *(`str`, optional)* - The path to a directory containing the saved encoder and decoder of a RetrievalModel. (E.g. "outputs/") OR a valid BART or MarianMT model.
* **config** *(`dict`, optional)* - A configuration file to build an EncoderDecoderModel. See [here](https://huggingface.co/transformers/model_doc/encoderdecoder.html#encoderdecoderconfig).

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

> *simpletransformers.retrieval.RetrievalModel.train_model*{: .function-name}(self, train_data, output_dir=None, show_running_loss=True, args=None, eval_data=None, verbose=True, **kwargs)

Trains the model using 'train_data'
{: .function-text}

> Parameters
{: .parameter-blockquote}

* **train_data** - Pandas DataFrame containing the 2 columns - `input_text`, `target_text`.
    - `input_text`: The input text sequence.
    - `target_text`: The target text sequence.

* **output_dir** *(`str`, optional)* - The directory where model files will be saved. If not given, `self.args['output_dir']` will be used.

* **show_running_loss** *(`bool`, optional)* - If True, the running loss (training loss at current step) will be logged to the console.

* **args** *(`dict`, optional)* - A dict of configuration options for the `RetrievalModel`. Any changes made will persist for the model.

* **eval_data** *(optional)* - Evaluation data (same format as train_data) against which evaluation will be performed when `evaluate_during_training` is enabled. Is required if `evaluate_during_training` is enabled.

* **kwargs** *(optional)* - Additional metrics that should be calculated. Pass in the metrics as keyword arguments *(name of metric: function to calculate metric)*. Refer to the [additional metrics](/docs/usage/#additional-evaluation-metrics) section.
E.g. `f1=sklearn.metrics.f1_score`.
A metric function should take in two parameters. The first parameter will be the true labels, and the second parameter will be the predictions.
{: .parameter-list}

> Returns
{: .returns}

* `None`
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

> *simpletransformers.retrieval.RetrievalModel.eval_model*{: .function-name}(self, eval_data,
> output_dir=None, verbose=True, silent=False, **kwargs)

Evaluates the model using 'eval_data'
{: .function-text}

> Parameters
{: .parameter-blockquote}

* **eval_data** - Pandas DataFrame containing the 2 columns - `input_text`, `target_text`.
    - `input_text`: The input text sequence.
    - `target_text`: The target text sequence.

* **output_dir** *(`str`, optional)* - The directory where model files will be saved. If not given, `self.args['output_dir']` will be used.

* **verbose** *(`bool`, optional)* - If verbose, results will be printed to the console on completion of evaluation.

* **silent** *(`bool`, optional)* - If silent, tqdm progress bars will be hidden.

* **kwargs** *(optional)* - Additional metrics that should be calculated. Pass in the metrics as keyword arguments *(name of metric: function to calculate metric)*. Refer to the [additional metrics](/docs/usage/#additional-evaluation-metrics) section.
E.g. `f1=sklearn.metrics.f1_score`.
A metric function should take in two parameters. The first parameter will be the true labels, and the second parameter will be the predictions.
{: .parameter-list}

> Returns
{: .returns}

* **result** *(`dict`)* - Dictionary containing evaluation results.

**Note:** For more details on evaluating Retrieval models with custom metrics, please refer to the [Evaluating Generated Sequences](/docs/retrieval-specifics/#evaluating-generated-sequences) section.
{: .notice--info}

**Note:** For more details on evaluating models with Simple Transformers, please refer to the [Tips and Tricks](/docs/tips-and-tricks) section.
{: .notice--info}

## Making Predictions With a `RetrievalModel`

The `predict()`  method is used to make predictions with the model.

```python
to_predict = [
    "Tyson is a Cyclops, a son of Poseidon, and Percy Jackson’s half brother. He is the current general of the Cyclopes army."
]

predictions = model.predict(to_predict)
```

**Note:** The input **must** be a List even if there is only one sentence.
{: .notice--info}


> *simpletransformers.retrieval.RetrievalModel.predict*{: .function-name}(to_predict)

Performs predictions on a list of text `to_predict`.
{: .function-text}

> Parameters
{: .parameter-blockquote}

* **to_predict** - A python list of text (str) to be sent to the model for prediction.
{: .parameter-list}

> Returns
{: .returns}

* **preds** *(`list`)* - A python list of the generated sequences.
{: .return-list}
