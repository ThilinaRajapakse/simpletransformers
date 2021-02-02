---
title: Question Answering Specifics
permalink: /docs/qa-specifics/
excerpt: "Specific notes for Question Answering tasks."
last_modified_at: 2021/02/02 11:19:58
toc: true
---

The goal of Question Answering is to find the **answer** to a question given a **question** *and* an accompanying **context**. The predicted answer will be either a span of text from the **context** or an empty string (indicating the question cannot be answered from the context).


## Usage Steps

The process of performing Question Answering in Simple Transformers does not deviate from the [standard pattern](/docs/usage/#task-specific-models).

1. Initialize a `QuestionAnsweringModel`
2. Train the model with `train_model()`
3. Evaluate the model with `eval_model()`
4. Make predictions on (unlabelled) data with `predict()`


## Supported Model Types

New model types are regularly added to the library. Question Answering tasks currently supports the model types given below.

| Model       | Model code for `QuestionAnsweringModel` |
| ----------- | --------------------------------------- |
| ALBERT      | albert                                  |
| BERT        | bert                                    |
| CamemBERT   | camembert                               |
| DistilBERT  | distilbert                              |
| ELECTRA     | electra                                 |
| Longformer  | longformer                              |
| MPNet       | mpnet                                   |
| MobileBERT  | mobilebert                              |
| RoBERTa     | roberta                                 |
| SqueezeBert | squeezebert                             |
| XLM         | xlm                                     |
| XLM-RoBERTa | xlmroberta                              |
| XLNet       | xlnet                                   |

**Tip:** The model code is used to specify the `model_type` in a Simple Transformers model.
{: .notice--success}


## Lazy Loading Data

The system memory required to keep a large dataset in memory can be prohibitively large. In such cases, the data can be lazy loaded from disk to minimize memory consumption.

To enable lazy loading, you must set the `lazy_loading` flag to `True` in `QuestionAnsweringArgs`.


```python
model_args = QuestionAnsweringArgs()
model_args.lazy_loading = True
```

**Note:** This will typically be slower as the feature conversion is done on the fly. However, the tradeoff between speed and memory consumption should be reasonable.
{: .notice--info}

**Tip:** See [Lazy Loading Data Formats](/docs/qa-data-formats/#lazy-loading-data-format) for information on the data formats.
{: .notice--success}

**Tip:** See [Configuring a QuestionAnsweringArgs model](/docs/qa-model/#configuring-a-questionansweringmodel) for information on configuring the model to read the lazy loading data file correctly.
{: .notice--success}

**Tip:** You can find a minimal example script in `examples/question_answering/lazy_qa.py`.
{: .notice--success}
