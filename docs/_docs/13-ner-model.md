---
title: NER Specifics
permalink: /docs/ner-specifics/
excerpt: "Named entity recognition."
last_modified_at: 2020-05-02 17:58:53
---


## `NERModel`

The `NERModel` class is used for Named Entity Recognition (token classification).

To create a `NERModel`, you must specify a `model_type` and a `model_name`.

- `model_type` should be one of the model types from the [supported models](/docs/ner-specifics/) (e.g. bert, electra, xlnet)
- `model_name` specifies the exact architecture and trained weights to use. This may be a Hugging Face Transformers compatible pre-trained model, a community model, or the path to a directory containing model files.

    **Note:** For a list of standard pre-trained models, see [here](https://huggingface.co/transformers/pretrained_models.html).
    {: .notice--info}

    **Note:** For a list of community models, see [here](https://huggingface.co/models).
    {: .notice--info}

    You may use any of these models provided the `model_type` is supported.

```python
from simpletransformers.classification import NERModel


model = NERModel(
    "roberta", "roberta-base"
)
```

**Note:** For more information on working with Simple Transformers models, please refer to the [General Usage section](/docs/usage/#creating-a-task-specific-model).
{: .notice--info}
