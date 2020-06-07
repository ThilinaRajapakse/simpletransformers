---
title: Question Answering Specifics
permalink: /docs/qa-specifics/
excerpt: "Specific notes for Question Answering tasks."
last_modified_at: 2020-05-02 17:58:53
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
| DistilBERT  | distilbert                              |
| ELECTRA     | electra                                 |
| Longformer  | longformer                           |
| RoBERTa     | roberta                                 |
| XLM         | xlm                                     |
| XLM-RoBERTa | xlmroberta                              |
| XLNet       | xlnet                                   |

**Tip:** The model code is used to specify the `model_type` in a Simple Transformers model.
{: .notice--success}
