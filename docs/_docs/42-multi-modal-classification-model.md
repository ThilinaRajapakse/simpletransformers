---
title: Multi-Modal Classification Model
permalink: /docs/multi-modal-classification-model/
excerpt: "MultiModalClassificationModel for Multi-Modal Classification tasks."
last_modified_at: 2020/12/29 17:01:58
toc: true
---


## `MultiModalClassificationModel`

The `MultiModalClassificationModel` class is used for Multi-Modal Classification.

To create a `MultiModalClassificationModel`, you must specify a `model_type` and a `model_name`.

- `model_type` should be one of the model types from the [supported models](/docs/multi-modal-classification-specifics/) (e.g. bert)
- `model_name` specifies the exact architecture and trained weights to use. This may be a Hugging Face Transformers compatible pre-trained model, a community model, or the path to a directory containing model files.

    **Note:** For a list of standard pre-trained models, see [here](https://huggingface.co/transformers/pretrained_models.html).
    {: .notice--info}

    **Note:** For a list of community models, see [here](https://huggingface.co/models).
    {: .notice--info}

    You may use any of these models provided the `model_type` is supported.

```python
from simpletransformers.classification import MultiModalClassificationModel


model = MultiModalClassificationModel(
    "bert", "bert-base-uncased"
)
```

**Note:** For more information on working with Simple Transformers models, please refer to the [General Usage section](/docs/usage/#creating-a-task-specific-model).
{: .notice--info}


### Configuring a `MultiModalClassificationModel`

`MultiModalClassificationModel` has several task-specific configuration options.


| Argument                  | Type  | Default | Description                                                                                          |
| ------------------------- | ----- | ------- | ---------------------------------------------------------------------------------------------------- |
| regression                | bool   | `False`   | Set to `True` if predicting a continuous value.               |
| num_image_embeds          | int   | `1`    | Number of image embeddings from the image encoder. |
| text_label               | str   | `text`    | The name of the *text* column/field in the dataset.                                                        |
| labels_label               | str   | `labels`    | The name of the *labels* column/field in the dataset.                                                        |
| images_label               | str   | `images`    | The name of the *images* column/field in the dataset.                                                        |
| image_type_extension               | str   | `""`    | The file type extension for image files (e.g. .json, .jpg). Only required if filepaths in the datasets do not include the file extensions.                                                        |
| data_type_extension               | str   | `""`    | The file type extension for text files (e.g. .json, .jpg). Only required if filepaths in the datasets do not include the file extensions.                                                        |
| special_tokens_list     | list      | []      | The list of special tokens to be added to the model tokenizer                                                                                         |

```python
from simpletransformers.classification import MultiModalClassificationModel, MultiModalClassificationArgs


model_args = MultiModalClassificationArgs()

model = MultiModalClassificationModel(
    "bert",
    "bert-base-uncased",
    args=model_args,
)
```

**Note:** For configuration options common to all Simple Transformers models, please refer to the [Configuring a Simple Transformers Model section](/docs/usage/#configuring-a-simple-transformers-model).
{: .notice--info}


## `Class MultiModalClassificationModel`

> *simpletransformers.classification.MultiModalClassificationModel*{: .function-name}(self, model_type, model_name, multi_label=False, label_list=None, num_labels=None, pos_weight=None, args=None, use_cuda=True, cuda_device=-1, **kwargs,)

Initializes a MultiModalClassificationModel model.
{: .function-text}

> Parameters
{: .parameter-blockquote}

* **model_type** *(`str`)* - The type of model to use ([model types](/docs/multi-modal-classification-specifics/#supported-model-types))

* **model_name** *(`str`)* - The exact architecture and trained weights to use. This may be a Hugging Face Transformers compatible pre-trained model, a community model, or the path to a directory containing model files.

* **multi_label** *(`bool`, optional)* - Set to True for multi label tasks.

* **label_list** *(`list`, optional)* - A list of all the labels (str) in the dataset.

* **num_labels** *(`int`, optional)* - The number of labels or classes in the dataset.

* **pos_weight** *(`list`, optional)* - A list of length `num_labels` containing the weights to assign to each label for loss calculation.

* **args** *(`dict`, optional)* - [Default args](/docs/usage/#configuring-a-simple-transformers-model) will be used if this parameter is not provided. If provided, it should be a dict containing the args that should be changed in the default args.

* **use_cuda** *(`bool`, optional)* - Use GPU if available. Setting to False will force model to use CPU only. (See [here](/docs/usage/#to-cuda-or-not-to-cuda))

* **cuda_device** *(`int`, optional)* - Specific GPU that should be used. Will use the first available GPU by default. (See [here](/docs/usage/#selecting-a-cuda-device))

* **kwargs** *(optional)* - For providing proxies, force_download, resume_download, cache_dir and other options specific to the 'from_pretrained' implementation where this will be supplied. (See [here](/docs/usage/#options-for-downloading-pre-trained-models))
{: .parameter-list}

> Returns
{: .returns}

* `None`
{: .return-list}

**Note:** For configuration options common to all Simple Transformers models, please refer to the [Configuring a Simple Transformers Model section](/docs/usage/#configuring-a-simple-transformers-model).
{: .notice--info}


## Training a `MultiModalClassificationModel`

The `train_model()`  method is used to train the model.

```python
model.train_model(train_data)
```

> *simpletransformers.classification.MultiModalClassificationModel*{: .function-name}(self, train_data, files_list=None, image_path=None, text_label=None, labels_label=None, images_label=None, image_type_extension=None, data_type_extension=None, auto_weights=False, output_dir=None, show_running_loss=True, args=None, eval_data=None, verbose=True, **kwargs)

Trains the model using 'train_data'
{: .function-text}

> Parameters
{: .parameter-blockquote}

* **train_data** - Path to data directory containing text files (JSON) and image files OR a Pandas DataFrame.
                If a DataFrame is given, it should contain the columns [text, labels, images]. When using a DataFrame,
                image_path MUST be specified. The image column of the DataFrame should contain the relative path from
                image_path to the image.
                E.g:
                    For an image file 1.jpeg located in `"data/train/"`;<br>
                        `image_path = "data/train/"`<br>
                        `images = "1.jpeg"`

* **files_list** *(`list` or `str`, optional)* - If given, only the files specified in this list will be taken from data directory. `files_list` can be a Python list or the path (`str`) to a JSON file containing a list of files.

* **image_path** *(`str`, optional)* - Must be specified when using DataFrame as input. Path to the directory containing the images.

* **text_label** *(`str`, optional)* - Column name to look for instead of the default `"text"`. Note that the change will persist even after the method call terminates. That is, the `args` dictionary of the model itself will be modified.

* **labels_label** *(`str`, optional)* - Column name to look for instead of the default `"labels"`. Note that the change will persist even after the method call terminates. That is, the `args` dictionary of the model itself will be modified.

* **images_label** *(`str`, optional)* - Column name to look for instead of the default `"images"`. Note that the change will persist even after the method call terminates. That is, the `args` dictionary of the model itself will be modified.

* **image_type_extension** *(`str`, optional)* - If given, this will be added to the end of each value in "images".

* **data_type_extension** *(`str`, optional)* - If given, this will be added to the end of each value in "files_list".

* **auto_weights** *(`str`, optional)* - If True, weights will be used to balance the classes. Only implemented for multi label tasks currently.

* **output_dir** *(`bool`, optional)* - The directory where model files will be saved. If not given, `self.args['output_dir']` will be used.

* **show_running_loss** *(`bool`, optional)* - If True, the running loss (training loss at current step) will be logged to the console.

* **args** *(`dict`, optional)* - A dict of configuration options for the `MultiModalClassificationModel`. Any changes made will persist for the model.

* **eval_data** *(optional)* - Evaluation data (same format as train_data) against which evaluation will be performed when evaluate_during_training is enabled. Is required if evaluate_during_training is enabled.

* **kwargs** *(optional)* - Additional metrics that should be calculated. Pass in the metrics as keyword arguments *(name of metric: function to calculate metric)*. Refer to the [additional metrics](/docs/tips-and-tricks/#additional-evaluation-metrics) section.
E.g. `f1=sklearn.metrics.f1_score`.
A metric function should take in two parameters. The first parameter will be the true labels, and the second parameter will be the predictions.
{: .parameter-list}

> Returns
{: .returns}

* **global_step** *(`int`)* - Number of global steps trained

* **training_details** *(`list`)* - Average training loss if `evaluate_during_training` is False or full training progress scores if `evaluate_during_training` is True
{: .return-list}

**Note:** For more details on training models with Simple Transformers, please refer to the [Tips and Tricks](/docs/tips-and-tricks) section.
{: .notice--info}


## Evaluating a `MultiModalClassificationModel`

The `eval_model()`  method is used to evaluate the model.

The following metrics will be calculated by default:

* `mcc` - [Matthews correlation coefficient](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.matthews_corrcoef.html)
* `tp` - True positives
* `tn` - True negatives
* `fp` - False positives
* `fn` - False negatives
* `eval_loss` - Cross Entropy Loss for eval_data


```python
result, model_outputs = model.eval_model(eval_data)
```

> *simpletransformers.classification.MultiModalClassificationModel.eval_model*{: .function-name}(self, eval_data,
files_list=None, image_path=None, text_label=None, labels_label=None, images_label=None, image_type_extension=None, data_type_extension=None, auto_weights=False, output_dir=None, show_running_loss=True, args=None, eval_data=None, verbose=True, **kwargs)

Evaluates the model using 'eval_data'
{: .function-text}

> Parameters
{: .parameter-blockquote}

* **eval_data** - Path to data directory containing text files (JSON) and image files OR a Pandas DataFrame.
                If a DataFrame is given, it should contain the columns [text, labels, images]. When using a DataFrame,
                image_path MUST be specified. The image column of the DataFrame should contain the relative path from
                image_path to the image.
                E.g:
                    For an image file 1.jpeg located in `"data/train/"`;<br>
                        `image_path = "data/train/"`<br>
                        `images = "1.jpeg"`

* **files_list** *(`list` or `str`, optional)* - If given, only the files specified in this list will be taken from data directory. `files_list` can be a Python list or the path (`str`) to a JSON file containing a list of files.

* **image_path** *(`str`, optional)* - Must be specified when using DataFrame as input. Path to the directory containing the images.

* **text_label** *(`str`, optional)* - Column name to look for instead of the default `"text"`. Note that the change will persist even after the method call terminates. That is, the `args` dictionary of the model itself will be modified.

* **labels_label** *(`str`, optional)* - Column name to look for instead of the default `"labels"`. Note that the change will persist even after the method call terminates. That is, the `args` dictionary of the model itself will be modified.

* **images_label** *(`str`, optional)* - Column name to look for instead of the default `"images"`. Note that the change will persist even after the method call terminates. That is, the `args` dictionary of the model itself will be modified.

* **image_type_extension** *(`str`, optional)* - If given, this will be added to the end of each value in "images".

* **data_type_extension** *(`str`, optional)* - If given, this will be added to the end of each value in "files_list".

* **output_dir** *(`str`, optional)* - The directory where model files will be saved. If not given, `self.args['output_dir']` will be used.

* **verbose** *(`bool`, optional)* - If verbose, results will be printed to the console on completion of evaluation.

* **silent** *(`bool`, optional)* - If silent, tqdm progress bars will be hidden.

* **kwargs** *(optional)* - Additional metrics that should be calculated. Pass in the metrics as keyword arguments *(name of metric: function to calculate metric)*. Refer to the [additional metrics](/docs/tips-and-tricks/#additional-evaluation-metrics) section.
E.g. `f1=sklearn.metrics.f1_score`.
A metric function should take in two parameters. The first parameter will be the true labels, and the second parameter will be the predictions.
{: .parameter-list}

> Returns
{: .returns}

* **result** *(`dict`)* -  Dictionary containing evaluation results. (Matthews correlation coefficient, tp, tn, fp, fn)

* **model_outputs** *(`list`)* - List of model outputs for each row in eval_data
{: .return-list}

**Note:** For more details on evaluating models with Simple Transformers, please refer to the [Tips and Tricks](/docs/tips-and-tricks) section.
{: .notice--info}


## Making Predictions With a `MultiModalClassificationModel`

The `predict()`  method is used to make predictions with the model.


> *simpletransformers.classification.MultiModalClassificationModel.predict*{: .function-name}(to_predict, image_path, image_type_extension=None)

Performs predictions on a list of text `to_predict`.
{: .function-text}

> Parameters
{: .parameter-blockquote}

* **to_predict** - A python dictionary to be sent to the model for prediction.
                The dictionary should be of the form `{"text": [<list of sentences>], "images": [<list of images>]}`.

* **image_path** *(`str`)* - Path to the directory containing the image or images.

* **image_type_extension** *(`str`, optional)* - If given, this will be added to the end of each value in "images".

> Returns
{: .returns}

* **preds** *(`list`)* - A python list of the predictions for each text.

* **model_outputs** *(`list`)* - A python list of the raw model outputs for each text.
{: .return-list}
