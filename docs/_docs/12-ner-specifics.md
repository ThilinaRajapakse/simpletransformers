---
title: Named Entity Recognition Specifics
permalink: /docs/ner-specifics/
excerpt: "Specific notes for Named Entity Recognition tasks."
last_modified_at: 2021/02/19 15:34:38
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
| ----------- | ------------------------- |
| ALBERT      | albert                    |
| BERT        | bert                      |
| BERTweet    | bertweet                  |
| BigBird     | bigbird                   |
| CamemBERT   | camembert                 |
| DeBERTa     | deberta                   |
| DeBERTa     | deberta                   |
| DeBERTaV2   | deberta-v2                |
| DistilBERT  | distilbert                |
| ELECTRA     | electra                   |
| HerBERT     | herbert                   |
| LayoutLM    | layoutlm                  |
| LayoutLMv2  | layoutlmv2                |
| Longformer  | longformer                |
| MobileBERT  | mobilebert                |
| MPNet       | mpnet                     |
| RemBERT     | rembert                   |
| RoBERTa     | roberta                   |
| SqueezeBert | squeezebert               |
| XLM         | xlm                       |
| XLM-RoBERTa | xlmroberta                |
| XLNet       | xlnet                     |

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

## Prediction Caveats

By default, `NERModel` will split input sequences to the `predict()` method on spaces and assign a NER tag to each "word" of the split sequence. This might not be desirable in some languages (e.g. Chinese). To avoid this, you can specify `split_on_space=False` when calling the `NERModel.predict()` method. In this case, you must provide a list of lists as the `to_predict` input to the `predict()` method. The inner list will be the list of split "words" belonging to a single sequence and the outer list is the list of all sequences.

## Lazy Loading Data

The system memory required to keep a large dataset in memory can be prohibitively large. In such cases, the data can be lazy loaded from disk to minimize memory consumption.

To enable lazy loading, you must set the `lazy_loading` flag to `True` in `NERArgs`.


```python
model_args = NERArgs()
model_args.lazy_loading = True
```

**Note:** The data must be input as a path to a file in the CoNLL format to use lazy loading. See [here](/docs/ner-data-formats/#text-file-in-conll-format) for the correct format.
{: .notice--info}


**Note:** This will typically be slower as the feature conversion is done on the fly. However, the tradeoff between speed and memory consumption should be reasonable.
{: .notice--info}

**Tip:** See [Configuring a NER model](/docs/ner-model/#configuring-a-ner-model) for information on configuring the model to read the lazy loading data file correctly.
{: .notice--success}
