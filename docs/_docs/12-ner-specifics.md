---
title: NER Model
permalink: /docs/ner-model/
excerpt: "NERModel for named entity recognition."
last_modified_at: 2020-05-02 17:58:53
---

The goal of Named Entity Recognition is to locate and classify *named entities* in a sequence. The named entities are pre-defined categories chosen according to the use case such as names of people, organizations, places, codes, time notations, monetary values, etc. Essentially, NER aims to assign a class to each token (usually a single word) in a sequence. Because of this, NER is also referred to as *token classification*.


## Usage Steps

The process of performing Named Entity Recognition in Simple Transformers does not deviate from the [standard pattern](/docs/usage/#task-specific-models).

1. Initialize a `NERModel`
2. Train the model with `train_model()`
3. Evaluate the model with `eval_model()`
4. Make predictions on (unlabelled) data with `predict()`


## Supported model types

New model types are regularly added to the library. Text classification tasks currently supports the model types given below.

| Model       | Model code for `ClassificationModel` |
|-------------|--------------------------------------|
| BERT        | bert                                 |
| *CamemBERT  | camembert                            |
| RoBERTa     | roberta                              |
| DistilBERT  | distilbert                           |
| ELECTRA     | electra                              |
| XLM-RoBERTa | xlmroberta                           |

**Tip:** The model code is used to specify the `model_type` in a Simple Transformers model.
{: .notice--success}


