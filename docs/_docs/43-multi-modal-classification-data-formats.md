---
title: Multi-Modal Classification Data Formats
permalink: /docs/multi-modal-classification-data-formats/
excerpt: "Data Formats for Multi-Modal Classification tasks."
last_modified_at: 2020/12/08 15:29:39
toc: true
---

There are several possible input formats you may use for Multi-Modal Classification tasks. The input formats are inspired by the [MM-IMDb](http://lisi1.unal.edu.co/mmimdb/) format.


## Data Formats

### Directory based

Each subset of data (E.g: train and test) should be in its own directory. The path to the directory can then be given directly to either `train_model()` or `eval_model()`.

Each data sample should have a text portion and an image associated with it (and a label/labels for training and evaluation data). The text for each sample should be in a separate JSON file. The JSON file may contain other fields in addition to the text itself but they will be ignored. The image associated with each sample should be in the same directory and both the text and the image must have the same identifier except for the file extension (E.g: `000001.json` and `000001.jpg`).

### Directory and file list

All data (including both train and test data) should be in the same directory. The path to this directory should be given to both `train_model()` and `eval_model()`. A second argument, `files_list` specifies which files should be taken from the directory. files_list can be a Python list or the path to a JSON file containing the list of files.

Each data sample should have a text portion and an image associated with it (and a label/labels for training and evaluation data). The text for each sample should be in a separate JSON file. The JSON file may contain other fields in addition to the text itself but they will be ignored. The image associated with each sample should be in the same directory and both the text and the image must have the same identifier except for the file extension (E.g: `000001.json` and `000001.jpg`).

### Pandas DataFrame

Data can also be given in a Pandas DataFrame. When using this format, the `image_path` argument must be specified and
it should be a String of the path to the directory containing the images. The DataFrame should contain at least 3
columns as detailed below.

- `text` (str) - The text associated with the sample.
- `images` (str) - The relative path to the image file from `image_path` directory.
- `labels` (str) - The label (or list of labels for multilabel tasks) associated with the sample.


**Note:** Both the training and evaluation data formats follow the specification given above and are identical to each other.
{: .notice--info}