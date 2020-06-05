---
title: Named Entitty Recognition Specifics
permalink: /docs/ner-specifics/
excerpt: "Specific notes for Named Entity Recognition tasks."
last_modified_at: 2020-05-02 17:58:53
toc: true
---

The goal of Named Entity Recognition is to locate and classify *named entities* in a sequence. The named entities are pre-defined categories chosen according to the use case such as names of people, organizations, places, codes, time notations, monetary values, etc. Essentially, NER aims to assign a class to each token (usually a single word) in a sequence. Because of this, NER is also referred to as *token classification*.


## Usage Steps

The process of performing Named Entity Recognition in Simple Transformers does not deviate from the [standard pattern](/docs/usage/#task-specific-models).

1. Initialize a `NERModel`
2. Train the model with `train_model()`
3. Evaluate the model with `eval_model()`
4. Make predictions on (unlabelled) data with `predict()`


## Supported Model Types

New model types are regularly added to the library. Named Entity Recognition tasks currently supports the model types given below.

| Model       | Model code for `NERModel` |
|-------------|--------------------------------------|
| BERT        | bert                                 |
| CamemBERT  | camembert                             |
| RoBERTa     | roberta                              |
| DistilBERT  | distilbert                           |
| ELECTRA     | electra                              |
| XLM-RoBERTa | xlmroberta                           |

**Tip:** The model code is used to specify the `model_type` in a Simple Transformers model.
{: .notice--success}


## Custom Labels

The default list of labels used in the `NERModel` is from the [CoNLL](https://www.clips.uantwerpen.be/conll2003/ner/) dataset which uses the following tags/labels.

`["O", "B-MISC", "I-MISC",  "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"]`

However, named entity recognition is a very versatile task and has many different applications. It is highly likely that you will wish to define and use your own token tags/labels.

This can be done by passing in your list of labels when creating the `NERModel` to the `labels` parameter.

```python
custom_labels = ["O", "B-SPELL", "I-SPELL", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-PLACE", "I-PLACE"]

model = NERModel(
    "bert", "bert-cased-base", labels=custom_labels
)
```
