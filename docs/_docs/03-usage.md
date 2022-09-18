---
title: "General Usage"
permalink: /docs/usage/
excerpt: "General usage instructions applicable to most tasks."
last_modified_at: 2021/11/10 21:24:03
toc: true
---

This section contains general usage information and tips applicable to most tasks in the library.

## Task Specific Models

Simple Transformer models are built with a particular Natural Language Processing (NLP) task in mind. Each such model comes equipped with features and functionality designed to best fit the task that they are intended to perform. The high-level process of using Simple Transformers models follows the same pattern.

1. Initialize a task-specific model
2. Train the model with `train_model()`
3. Evaluate the model with `eval_model()`
4. Make predictions on (unlabelled) data with `predict()`

However, there are necessary differences between the different models to ensure that they are well suited for their intended task. The key differences will typically be the differences in input/output data formats and any task specific features/configuration options. These can all be found in the documentation section for each task.

The currently implemented task-specific Simple Transformer models, along with their task, are given below.

| Task                                                      | Model                           |
| --------------------------------------------------------- | ------------------------------- |
| Binary and multi-class text classification                | `ClassificationModel`           |
| Conversational AI (chatbot training)                      | `ConvAIModel`                   |
| Language generation                                       | `LanguageGenerationModel`       |
| Language model training/fine-tuning                       | `LanguageModelingModel`         |
| Multi-label text classification                           | `MultiLabelClassificationModel` |
| Multi-modal classification (text and image data combined) | `MultiModalClassificationModel` |
| Named entity recognition                                  | `NERModel`                      |
| Question answering                                        | `QuestionAnsweringModel`        |
| Regression                                                | `ClassificationModel`           |
| Sentence-pair classification                              | `ClassificationModel`           |
| Text Representation Generation                            | `RepresentationModel`           |
| Document Retrieval                                        | `RetrievalModel`                |



## Creating a Task-Specific Model

To create a task-specific Simple Transformers model, you will typically specify a `model_type` and a `model_name`.
Any deviation from this will be noted in the appropriate model documentation.

- `model_type` should be one of the model types from the supported models (e.g. bert, electra, xlnet)
- `model_name` specifies the exact architecture and trained weights to use. This may be a Hugging Face Transformers compatible pre-trained model, a community model, or the path to a directory containing model files.

    **Note:** For a list of standard pre-trained models, see [here](https://huggingface.co/transformers/pretrained_models.html).
    {: .notice--info}

    **Note:** For a list of community models, see [here](https://huggingface.co/models).
    {: .notice--info}

    You may use any of these models provided the `model_type` is supported.


The code snippets below demonstrate the typical process of creating a Simple Transformers model, using the `ClassificationModel` as an example.


### Importing the task-specific model

```python
from simpletransformers.classification import ClassificationModel
```

### Loading a pre-trained model

```python
model = ClassificationModel(
    "roberta", "roberta-base"
)
```

### Loading a community model

```python
model = ClassificationModel(
    "bert", "KB/bert-base-swedish-cased"
)
```

### Loading a local save

When loading a saved model, the path to the directory containing the model file should be used.

```python
model = ClassificationModel(
    "bert", "outputs/best_model"
)
```


## To CUDA or not to CUDA

Deep Learning (DL) models are typically run on CUDA-enabled GPUs as the performance is far, *far* superior compared to running on a CPU. This is especially true for Transformer models considering that they are quite large even in relation to other DL models.
{: .notice--info}

CUDA is enabled by default on all Simple Transformers models.

### Enabling/Disabling CUDA

All Simple Transformers models have a `use_cuda` parameter to easily flip the switch on CUDA. Attempting to use CUDA when a CUDA device is not available will result in an error.

```python
model = ClassificationModel(
    "roberta", "roberta-base", use_cuda=False
)
```

**Pro tip:** You can use the following code snippet to ensure that your script will use CUDA if it is available, but won't error out if it is not.
{: .notice--info}

```python
import torch


cuda_available = torch.cuda.is_available()

model = ClassificationModel(
    "roberta", "roberta-base", use_cuda=cuda_available
)
```

### Selecting a CUDA device

If your environment has multiple CUDA devices, but you wish to use a particular device, you can specify the device ID (`int` starting from `0`) as shown below.

```python
model = ClassificationModel(
    "roberta", "roberta-base", cuda_device=1
```


## Configuring a Simple Transformers Model

Every task-specific Simple Transformers model comes with tons of configuration options to enable the user to easily tailor the model for their use case. These options can be categorized into two types, options common to all tasks and task-specific options. This section focuses on the common (or global) options. The task-specific options are detailed in the relevant documentation for the task.

Configuration options in Simple Transformers are defined as either dataclasses or as Python dicts. The [`ModelArgs`](https://github.com/ThilinaRajapakse/simpletransformers/blob/master/simpletransformers/config/model_args.py) dataclass contains all the global options set to their default values, as shown below.

| Argument                           | Type  | Default                                                   | Description                                                                                                                                                                                                                                              |
| ---------------------------------- | ----- | --------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| adafactor_beta1                    | float | 0                                                         | Coefficient used for computing running averages of gradient.                                                                                                                                                                                             |
| adafactor_clip_threshold           | float | 1.0                                                       | Threshold of root mean square of final gradient update.                                                                                                                                                                                                  |
| adafactor_decay_rate               | float | -0.8                                                      | Coefficient used to compute running averages of square.                                                                                                                                                                                                  |
| adafactor_eps                      | tuple | (1e-30, 1e-3)                                             | Regularization constants for square gradient and parameter scale respectively.                                                                                                                                                                           |
| adafactor_relative_step            | bool  | True                                                      | If True, time-dependent learning rate is computed instead of external learning rate.                                                                                                                                                                     |
| adafactor_scale_parameter          | bool  | True                                                      | If True, learning rate is scaled by root mean square.                                                                                                                                                                                                    |
| adafactor_warmup_init              | bool  | True                                                      | Time-dependent learning rate computation depends on whether warm-up initialization is being used.                                                                                                                                                        |
| adam_betas                         | tuple | (0.9, 0.999)                                              | coefficients used for computing running averages of gradient and its square with AdamW.                                                                                                                                                                  |
| adam_epsilon                       | float | 1e-8                                                      | Epsilon hyperparameter used in AdamOptimizer.                                                                                                                                                                                                            |
| best_model_dir                     | str   | outputs/best_model                                        | The directory where the best model (model checkpoints) will be saved (based on eval_during_training)                                                                                                                                                     |
| cache_dir                          | str   | cache_dir                                                 | The directory where cached files will be saved.                                                                                                                                                                                                          |
| config                             | dict  | {}                                                        | A dictionary containing configuration options that should be overriden in a model's config.                                                                                                                                                              |
| cosine_schedule_num_cycles         | float | 0.5                                                       | The number of waves in the cosine schedule if `cosine_schedule_With_warmup` is used. The number of hard restarts if `cosine_with_hard_restarts_schedule_with_warmup` is used.                                                                            |
| dataloader_num_workers             | int   | cpu_count ()  -   2   if   cpu_count ()  >   2   else   1 | Number of worker processed to use with the Pytorch dataloader.                                                                                                                                                                                           |
| do_lower_case                      | bool  | False                                                     | Set to True when using uncased models.                                                                                                                                                                                                                   |
| dynamic_quantize                   | bool  | False                                                     | Set to True to use dynamic quantization.                                                                                                                                                                                                                 |
| early_stopping_consider_epochs     | bool  | False                                                     | If True, end of epoch evaluation score will be considered for early stopping.                                                                                                                                                                            |
| early_stopping_delta               | float | 0                                                         | The improvement over best_eval_loss necessary to count as a better checkpoint.                                                                                                                                                                           |
| early_stopping_metric              | str   | eval_loss                                                 | The metric that should be used with early stopping. (Should be computed during eval_during_training).                                                                                                                                                    |
| early_stopping_metric_minimize     | bool  | True                                                      | Whether early_stopping_metric should be minimized (or maximized).                                                                                                                                                                                        |
| early_stopping_patience            | int   | 3                                                         | Terminate training after this many evaluations without an improvement in the evaluation metric greater then early_stopping_delta.                                                                                                                        |
| encoding                           | str   | None                                                      | Specify an encoding to be used when reading text files.                                                                                                                                                                                                  |
| eval_batch_size                    | int   | 8                                                         | The evaluation batch size.                                                                                                                                                                                                                               |
| evaluate_during_training           | bool  | False                                                     | Set to True to perform evaluation while training models. Make sure eval data is passed to the training method if enabled.                                                                                                                                |
| evaluate_during_training_steps     | int   | 2000                                                      | Perform evaluation at every specified number of steps. A checkpoint model and the evaluation results will be saved.                                                                                                                                      |
| evaluate_during_training_verbose   | bool  | False                                                     | Print results from evaluation during training.                                                                                                                                                                                                           |
| fp16                               | bool  | True                                                      | Whether or not fp16 mode should be used. Requires NVidia Apex library.                                                                                                                                                                                   |
| gradient_accumulation_steps        | int   | 1                                                         | The number of training steps to execute before performing a optimizer.step(). Effectively increases the training batch size while sacrificing training time to lower memory consumption.                                                                 |
| learning_rate                      | float | 4e-5                                                      | The learning rate for training.                                                                                                                                                                                                                          |
| logging_steps                      | int   | 50                                                        | Log training loss and learning at every specified number of steps.                                                                                                                                                                                       |
| manual_seed                        | int   | None                                                      | Set a manual seed if necessary for reproducible results.                                                                                                                                                                                                 |
| max_grad_norm                      | float | 1.0                                                       | Maximum gradient clipping.                                                                                                                                                                                                                               |
| max_seq_length                     | int   | 128                                                       | Maximum sequence length the model will support.                                                                                                                                                                                                          |
| multiprocessing_chunksize          | int   | -1                                                        | Number of examples sent to a CPU core at a time when using multiprocessing. If this is set to `-1`, the chunksize will be calculated dynamically as `max(len(data) // (args.process_count * 2), 500)`                                                    |
| n_gpu                              | int   | 1                                                         | Number of GPUs to use.                                                                                                                                                                                                                                   |
| no_cache                           | bool  | False                                                     | Cache features to disk.                                                                                                                                                                                                                                  |
| no_save                            | bool  | False                                                     | If `True`, models will not be saved to disk.                                                                                                                                                                                                             |
| not_saved_args                     | list  | ()                                                        | The `model_args` which should not be saved when the model is saved. If any `model_args` are not JSON serializable, those argument names should be specified here.                                                                                        |
| num_train_epochs                   | int   | 1                                                         | The number of epochs the model will be trained for.                                                                                                                                                                                                      |
| optimizer                          | str   | "AdamW"                                                   | Should be one of (AdamW, Adafactor)                                                                                                                                                                                                                      |
| output_dir                         | str   | "outputs/"                                                | The directory where all outputs will be stored. This includes model checkpoints and evaluation results.                                                                                                                                                  |
| overwrite_output_dir               | bool  | False                                                     | If True, the trained model will be saved to the ouput_dir and will overwrite existing saved models in the same directory.                                                                                                                                |
| polynomial_decay_schedule_lr_end   | float | 1e-7                                                      | The end learning rate.                                                                                                                                                                                                                                   |
| polynomial_decay_schedule_power    | float | 1.0                                                       | Power factor.                                                                                                                                                                                                                                            |
| process_count                      | int   | cpu_count ()  -   2   if   cpu_count ()  >   2   else   1 | Number of cpu cores (processes) to use when converting examples to features. Default is (number of cores - 2) or 1 if (number of cores <= 2)                                                                                                             |
| quantized_model                    | bool  | False                                                     | Set to True if loading a quantized model. Note that this will automatically be set to True if `dynamic_quantize` is enabled.                                                                                                                             |
| reprocess_input_data               | bool  | True                                                      | If True, the input data will be reprocessed even if a cached file of the input data exists in the cache_dir.                                                                                                                                             |
| save_eval_checkpoints              | bool  | True                                                      | Save a model checkpoint for every evaluation performed.                                                                                                                                                                                                  |
| save_model_every_epoch             | bool  | True                                                      | Save a model checkpoint at the end of every epoch.                                                                                                                                                                                                       |
| save_optimizer_and_scheduler       | bool  | True                                                      | Save optimizer and scheduler whenever they are available.                                                                                                                                                                                                |
| save_steps                         | int   | 2000                                                      | Save a model checkpoint at every specified number of steps. Set to -1 to disable.                                                                                                                                                                        |
| scheduler                          | str   | "linear_schedule_with_warmup"                             | The scheduler to use when training. Should be one of (constant_schedule, constant_schedule_with_warmup, linear_schedule_with_warmup, cosine_schedule_with_warmup, cosine_with_hard_restarts_schedule_with_warmup, polynomial_decay_schedule_with_warmup) |
| silent                             | bool  | False                                                     | Disables progress bars.                                                                                                                                                                                                                                  |
| tensorboard_dir                    | str   | None                                                      | The directory where Tensorboard events will be stored during training. By default, Tensorboard events will be saved in a subfolder inside runs/ like runs/Dec02_09-32-58_36d9e58955b0/.                                                                  |
| train_batch_size                   | int   | 8                                                         | The training batch size.                                                                                                                                                                                                                                 |
| use_cached_eval_features           | bool  | False                                                     | Evaluation during training uses cached features. Setting this to False will cause features to be recomputed at every evaluation step.                                                                                                                    |
| use_early_stopping                 | bool  | False                                                     | Use early stopping to stop training when early_stopping_metric doesn't improve (based on early_stopping_patience, and early_stopping_delta)                                                                                                              |
| use_multiprocessing                | bool  | True                                                      | If True, multiprocessing will be used when converting data into features. Enabling can speed up processing, but may be unstable in certain cases. Defaults to True.                                                                                      |
| use_multiprocessing_for_evaluation | bool  | False                                                     | If True, multiprocessing will be used when converting evaluation data into features. Enabling this can sometimes cause issues when evaluating during training. Defaults to False.                                                                        |
| wandb_kwargs                       | dict  | {}                                                        | Dictionary of keyword arguments to be passed to the W&B project.                                                                                                                                                                                         |
| wandb_project                      | str   | None                                                      | Name of W&B project. This will log all hyperparameter values, training losses, and evaluation metrics to the given project.                                                                                                                              |
| warmup_ratio                       | float | 0.06                                                      | Ratio of total training steps where learning rate will "warm up". Overridden if `warmup_steps` is specified.                                                                                                                                             |
| warmup_steps                       | int   | 0                                                         | Number of training steps where learning rate will "warm up". Overrides `warmup_ratio`.                                                                                                                                                                   |
| weight_decay                       | int   | 0                                                         | Adds L2 penalty.                                                                                                                                                                                                                                         |


You can override any of these default values by either editing the dataclass attributes or by passing in a Python dict containing the appropriate key-value pairs when initializing a Simple Transformers model.

### Using the dataclass

```python
from simpletransformers.classification import ClassificationModel, ClassificationArgs

model_args = ClassificationArgs()
model_args.num_train_epochs = 5
model_args.learning_rate = 1e-4

model = ClassificationModel("bert", "bert-base-cased", args=model_args)
```

### Using a python dictionary

```python
from simpletransformers.classification import ClassificationModel


model_args = {
    "num_train_epochs": 5,
    "learning_rate": 1e-4,
}

model = ClassificationModel("bert", "bert-base-cased", args=model_args)
```

**Tip:** Using the dataclass approach has the benefits of IDE auto-completion as well as ensuring that there are no typos in arguments that could lead to unexpected behaviour.
{: .notice--success}

**Tip:** Both the dataclass and the dictionary approaches are interchangeable.
{: .notice--success}
