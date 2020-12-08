---
title: Multi-Modal Classification Specifics
permalink: /docs/multi-modal-classification-specifics/
excerpt: "Specific notes for Multi-Modal Classification tasks."
last_modified_at: 2020/12/08 15:21:17
toc: true
---

Multi-Modal Classification fuses text and image data. This is performed using multi-modal bitransformer models introduced in the paper [Supervised Multimodal Bitransformers for Classifying Images and Text](https://arxiv.org/abs/1909.02950).


## Usage Steps

The process of performing Multi-Modal Classification in Simple Transformers does not deviate from the [standard pattern](/docs/usage/#task-specific-models).

1. Initialize a `Model`
2. Train the model with `train_model()`
3. Evaluate the model with `eval_model()`
4. Make predictions on (unlabelled) data with `predict()`


## Supported Model Types


| Model       | Model code for `Model` |
| ----------- | --------------------------------------- |
| BERT        | bert                                    |

**Tip:** The model code is used to specify the `model_type` in a Simple Transformers model.
{: .notice--success}


## Label formats

With Multi-Modal Classification, labels are always given as strings. You may specify a list of labels by passing in the list to `label_list` argument when creating the model. If `label_list` is given, `num_labels` is not required.

If `label_list` is not given, `num_labels` is required and the labels should be strings starting from `"0"` up to `"<num_labels>"`.
