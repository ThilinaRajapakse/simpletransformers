---
title: Seq2Seq Model
permalink: /docs/seq2seq-model/
excerpt: "Seq2SeqModel for Seq2Seq tasks."
last_modified_at: 2020/12/30 23:50:55
toc: true
---


## `Seq2SeqModel`

The `Seq2SeqModel` class is used for Sequence-to-Sequence tasks.

Currently, four main types of Sequence-to-Sequence models are available.

- Encoder-Decoder *(Generic)*
- MBART *(Translation)*
- MarianMT *(Translation)*
- BART *(Summarization)*
- RAG *(Retrieval Augmented Generation - E,g, Question Answering)


### Generic Encoder-Decoder Models

The following rules currently apply to generic Encoder-Decoder models (does not apply to BART and Marian):

- The decoder must be a `bert` model.
- The encoder can be one of `[bert, roberta, distilbert, camembert, electra]`.
- The encoder and the decoder must be of the same "size". (E.g. `roberta-base` encoder and a `bert-base-uncased` decoder)

To create a generic Encoder-Decoder model with `Seq2SeqModel`, you must provide the three parameters below.

- `encoder_type`: The type of model to use as the encoder.
- `encoder_name`: The exact architecture and trained weights to use. This may be a Hugging Face Transformers compatible pre-trained model, a community model, or the path to a directory containing model files.
- `decoder_name`: The exact architecture and trained weights to use. This may be a Hugging Face Transformers compatible pre-trained model, a community model, or the path to a directory containing model files.

    **Note:** For a list of standard pre-trained models, see [here](https://huggingface.co/transformers/pretrained_models.html).
    {: .notice--info}

    **Note:** For a list of community models, see [here](https://huggingface.co/models).
    {: .notice--info}

    **Note:** There is no `decoder_type` parameter as the decoder must be a `bert` model.
    {: .notice--info}


```python
from simpletransformers.seq2seq import Seq2SeqModel


model = Seq2SeqModel(
    "roberta",
    "roberta-base",
    "bert-base-cased",
)

```

### MarianMT Models

MarianMT models are translation models with support for a huge variety of languages.

The followng information is taken from the Hugging Face docs [here](https://huggingface.co/transformers/model_doc/marian.html#implementation-notes).

- Each model is about 298 MB on disk, there are 1,000+ models.

- The list of supported language pairs can be found here.

- The 1,000+ models were originally trained by Jörg Tiedemann using the Marian C++ library, which supports fast training and translation.

- All models are transformer encoder-decoders with 6 layers in each component. Each model’s performance is documented in a model card.

- The 80 opus models that require BPE preprocessing are not supported.


To create a MarianMT translation model, you must provide the two parameters below.

- `encoder_decoder_type`: This should be `"marian"`.
- `encoder_decoder_name`: The exact architecture and trained weights to use. This may be a Hugging Face Transformers compatible pre-trained model, a community model, or the path to a directory containing model files.

    **Note:** Please refer to the [Naming](https://huggingface.co/transformers/model_doc/marian.html#naming) and [Multilingual Models](https://huggingface.co/transformers/model_doc/marian.html#multilingual-models) sections of the MarianMT docs on Hugging Face for more information on choosing the `encoder_decoder_name`.
    {: .notice--info}


```python
from simpletransformers.seq2seq import Seq2SeqModel


# Initialize a Seq2SeqModel for English to German translation
model = Seq2SeqModel(
    encoder_decoder_type="marian",
    encoder_decoder_name="Helsinki-NLP/opus-mt-en-de",
)


```

### BART Models

- `encoder_decoder_type`: This should be `"bart"`.
- `encoder_decoder_name`: The exact architecture and trained weights to use. This may be a Hugging Face Transformers compatible pre-trained model, a community model, or the path to a directory containing model files.

    **Note:** For a list of standard pre-trained models, see [here](https://huggingface.co/transformers/pretrained_models.html).
    {: .notice--info}

    **Note:** For a list of community models, see [here](https://huggingface.co/models).
    {: .notice--info}


### MBART Models

- `encoder_decoder_type`: This should be `"mbart"`.
- `encoder_decoder_name`: The exact architecture and trained weights to use. This may be a Hugging Face Transformers compatible pre-trained model, a community model, or the path to a directory containing model files.

    **Note:** For a list of standard pre-trained models, see [here](https://huggingface.co/transformers/pretrained_models.html).
    {: .notice--info}

    **Note:** For a list of community models, see [here](https://huggingface.co/models).
    {: .notice--info}


### RAG Models

**Note:** You must have Faiss (GPU or CPU) installed to use RAG Models.
Faiss installation instructions can be found [here](https://github.com/facebookresearch/faiss/blob/master/INSTALL.md).
{: .notice--warning}

- `encoder_decoder_type`: Either `"rag-token"` or `"rag-sequence"`.
- `encoder_decoder_name`: The exact architecture and trained weights to use. This may be a Hugging Face Transformers compatible pre-trained model, a community model, or the path to a directory containing model files.

    **Note:** For a list of standard pre-trained models, see [here](https://huggingface.co/transformers/pretrained_models.html).
    {: .notice--info}

    **Note:** For a list of community models, see [here](https://huggingface.co/models).
    {: .notice--info}

- `index_name` (optional): Name of the index to use - `hf` for a canonical dataset from the datasets library, `custom` for a local index, or `legacy` for the original index. This will default to `custom` (not necessary to specify the parameter) when a local knowledge dataset is used.
- knowledge_dataset (optional): Path to a TSV file (two columns - `title`, `text`) containing a knowledge dataset for RAG or the path to a directory containing a saved Huggingface dataset for RAG. If this is not given for a RAG model, a dummy dataset will be used.
- `index_path` (optional): Path to the faiss index of the custom knowledge dataset. If this is not given and `knowledge_dataset` is given, it will be computed.
- `dpr_ctx_encoder_model_name` (optional): The DPR context encoder model to use. This may be a Hugging Face Transformers compatible pre-trained model, a community model, or the path to a directory containing model files. This is required when using a custom `knowledge_dataset`.




### Configuring a `Seq2SeqModel`

`Seq2SeqModel` has the following task-specific configuration options.

| Argument                    | Type    | Default | Description                                                                                                                                   |
| --------------------------- | ------- | ------- | --------------------------------------------------------------------------------------------------------------------------------------------- |
| base_marian_model_name                    | str     | None   | Name of the base Marian model used to load the tokenizer.                                                                             |
| dataset_class               | Dataset | None    | A custom dataset class to use. (Subclass of Pytorch Dataset)                                                                                  |
| do_sample                   | bool    | False   | If set to False greedy decoding is used. Otherwise sampling is used. Defaults to False as defined in configuration_utils.PretrainedConfig.    |
| early_stopping              | bool    | True    | if set to True beam search is stopped when at least num_beams sentences finished per batch.                                                   |
| evaluate_generated_text     | bool    | False   | Generate sequences for evaluation.                                                                                                            |
| length_penalty              | float   | 2.0     | Exponential penalty to the length. Default to 2.                                                                                              |
| max_length                  | int     | 20      | The max length of the sequence to be generated. Between 0 and infinity. Default to 20.                                                        |
| max_steps                   | int     | -1      | Maximum number of training steps. Will override the effect of num_train_epochs.                                                               |
| num_beams                   | int     | 1       | Number of beams for beam search. Must be between 1 and infinity. 1 means no beam search. Default to 1.                                        |
| num_return_sequences        | int     | 1       | The number of samples to generate.                                                                                                            |
| rag_embed_batch_size        | int     | 1       | The batch size used when generating embeddings for RAG models.                                                                                                            |
| repetition_penalty          | float   | 1.0     | The parameter for repetition penalty. Between 1.0 and infinity. 1.0 means no penalty. Default to 1.0.                                         |
| top_k                       | float   | None    | Filter top-k tokens before sampling (<=0: no filtering)                                                                                       |
| top_p                       | float   | None    | Nucleus filtering (top-p) before sampling (<=0.0: no filtering)                                                                               |
| use_multiprocessed_decoding | bool    | False   | Use multiprocessing when decoding outputs. Significantly speeds up decoding (CPU intensive). Turn off if multiprocessing causes insatibility. |
| save_knowledge_dataset | bool    | True   | Save the Knowledge Dataset when saving a RAG model |
| save_knowledge_dataset_with_checkpoints | bool    | False   | Save the knowledge dataset when saving a RAG model training checkpoint |
| split_text_character | str    | " "   | The character used to split text on when splitting text in a RAG model knowledge dataset |
| split_text_n | int    | 100   | Split text into a new *doc* every `split_text_n` occurences of `split_text_character` when splitting text in a RAG model knowledge dataset |
| src_lang                    | str     | en_XX   | Code for the source language. Only relevant to MBART model.                                                                                   |
| tgt_lang                    | str     | ro_RO   | Code for the target language. Only relevant to MBART model.                                                                                   |


```python
from simpletransformers.seq2seq import Seq2SeqModel, Seq2SeqArgs


model_args = Seq2SeqArgs()
model_args.num_train_epochs = 3

model = Seq2SeqModel(
    encoder_type,
    "roberta-base",
    "bert-base-cased",
    args=model_args,
)

```

**Note:** For configuration options common to all Simple Transformers models, please refer to the [Configuring a Simple Transformers Model section](/docs/usage/#configuring-a-simple-transformers-model).
{: .notice--info}


## `Class Seq2SeqModel`

> *simpletransformers.seq2seq.Seq2SeqModel*{: .function-name}(self, encoder_type=None, encoder_name=None, decoder_name=None, encoder_decoder_type=None, encoder_decoder_name=None, config=config, args=None, use_cuda=True, cuda_device=-1, **kwargs,)

Initializes a Seq2SeqModel model.
{: .function-text}

> Parameters
{: .parameter-blockquote}

* **encoder_type** *(`str`, optional)* - The type of model to use as the encoder.

* **encoder_name** *(`str`, optional)* - The exact architecture and trained weights to use. This may be a Hugging Face Transformers compatible pre-trained model, a community model, or the path to a directory containing model files.

* **decoder_name** *(`str`, optional)* - The exact architecture and trained weights to use. This may be a Hugging Face Transformers compatible pre-trained model, a community model, or the path to a directory containing model files.

* **encoder_decoder_type** *(`str`, optional)* - The type of encoder-decoder model. (E.g. bart)

* **encoder_decoder_name** *(`str`, optional)* - The path to a directory containing the saved encoder and decoder of a Seq2SeqModel. (E.g. "outputs/") OR a valid BART or MarianMT model.
* **config** *(`dict`, optional)* - A configuration file to build an EncoderDecoderModel. See [here](https://huggingface.co/transformers/model_doc/encoderdecoder.html#encoderdecoderconfig).

* **args** *(`dict`, optional)* - [Default args](/docs/usage/#configuring-a-simple-transformers-model) will be used if this parameter is not provided. If provided, it should be a dict containing the args that should be changed in the default args or a `Seq2SeqArgs` object.

* **use_cuda** *(`bool`, optional)* - Use GPU if available. Setting to False will force model to use CPU only. (See [here](/docs/usage/#to-cuda-or-not-to-cuda))

* **cuda_device** *(`int`, optional)* - Specific GPU that should be used. Will use the first available GPU by default. (See [here](/docs/usage/#selecting-a-cuda-device))

* **kwargs** *(optional)* - For providing proxies, force_download, resume_download, cache_dir and other options specific to the 'from_pretrained' implementation where this will be supplied. (See [here](/docs/usage/#options-for-downloading-pre-trained-models))
{: .parameter-list}

> Returns
{: .returns}

* `None`
{: .return-list}


## Training a `Seq2SeqModel`

The `train_model()`  method is used to train the model.

```python
model.train_model(train_data)
```

> *simpletransformers.seq2seq.Seq2SeqModel.train_model*{: .function-name}(self, train_data, output_dir=None, show_running_loss=True, args=None, eval_data=None, verbose=True, **kwargs)

Trains the model using 'train_data'
{: .function-text}

> Parameters
{: .parameter-blockquote}

* **train_data** - Pandas DataFrame containing the 2 columns - `input_text`, `target_text`.
    - `input_text`: The input text sequence.
    - `target_text`: The target text sequence.

* **output_dir** *(`str`, optional)* - The directory where model files will be saved. If not given, `self.args['output_dir']` will be used.

* **show_running_loss** *(`bool`, optional)* - If True, the running loss (training loss at current step) will be logged to the console.

* **args** *(`dict`, optional)* - A dict of configuration options for the `Seq2SeqModel`. Any changes made will persist for the model.

* **eval_data** *(optional)* - Evaluation data (same format as train_data) against which evaluation will be performed when `evaluate_during_training` is enabled. Is required if `evaluate_during_training` is enabled.

* **kwargs** *(optional)* - Additional metrics that should be calculated. Pass in the metrics as keyword arguments *(name of metric: function to calculate metric)*. Refer to the [additional metrics](/docs/usage/#additional-evaluation-metrics) section.
E.g. `f1=sklearn.metrics.f1_score`.
A metric function should take in two parameters. The first parameter will be the true labels, and the second parameter will be the predictions.
{: .parameter-list}

> Returns
{: .returns}

* `None`
{: .return-list}

**Note:** For more details on evaluating Seq2Seq models with custom metrics, please refer to the [Evaluating Generated Sequences](/docs/seq2seq-specifics/#evaluating-generated-sequences) section.
{: .notice--info}

**Note:** For more details on training models with Simple Transformers, please refer to the [Tips and Tricks](/docs/tips-and-tricks) section.
{: .notice--info}


## Evaluating a `Seq2SeqModel`

The `eval_model()`  method is used to evaluate the model.

The following metrics will be calculated by default:

* `eval_loss` - Model loss over the evaluation data


```python
result = model.eval_model(eval_data)
```

> *simpletransformers.seq2seq.Seq2SeqModel.eval_model*{: .function-name}(self, eval_data,
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

**Note:** For more details on evaluating Seq2Seq models with custom metrics, please refer to the [Evaluating Generated Sequences](/docs/seq2seq-specifics/#evaluating-generated-sequences) section.
{: .notice--info}

**Note:** For more details on evaluating models with Simple Transformers, please refer to the [Tips and Tricks](/docs/tips-and-tricks) section.
{: .notice--info}

## Making Predictions With a `Seq2SeqModel`

The `predict()`  method is used to make predictions with the model.

```python
to_predict = [
    "Tyson is a Cyclops, a son of Poseidon, and Percy Jackson’s half brother. He is the current general of the Cyclopes army."
]

predictions = model.predict(to_predict)
```

**Note:** The input **must** be a List even if there is only one sentence.
{: .notice--info}


> *simpletransformers.seq2seq.Seq2SeqModel.predict*{: .function-name}(to_predict)

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
