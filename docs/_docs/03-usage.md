---
title: "General Usage"
permalink: /docs/usage/
excerpt: "General usage instructions applicable to most tasks."
last_modified_at: 2020/07/06 03:17:33
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

| Argument                         | Type  | Default                                                   | Description                                                                                                                                                                              |
| -------------------------------- | ----- | --------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| adam_epsilon                     | float | 1e-8                                                      | Epsilon hyperparameter used in AdamOptimizer.                                                                                                                                            |
| best_model_dir                   | str   | outputs/best_model                                        | The directory where the best model (model checkpoints) will be saved (based on eval_during_training)                                                                                     |
| cache_dir                        | str   | cache_dir                                                 | The directory where cached files will be saved.                                                                                                                                          |
| config                           | dict  | {}                                                        | A dictionary containing configuration options that should be overriden in a model's config.                                                                                              |
| do_lower_case                    | bool  | False                                                     | Set to True when using uncased models.                                                                                                                                                   |
| early_stopping_consider_epochs   | bool  | False                                                     | If True, end of epoch evaluation score will be considered for early stopping.                                                                                                            |
| early_stopping_delta             | float | 0                                                         | The improvement over best_eval_loss necessary to count as a better checkpoint.                                                                                                           |
| early_stopping_metric            | str   | eval_loss                                                 | The metric that should be used with early stopping. (Should be computed during eval_during_training).                                                                                    |
| early_stopping_metric_minimize   | bool  | True                                                      | Whether early_stopping_metric should be minimized (or maximized).                                                                                                                        |
| early_stopping_patience          | int   | 3                                                         | Terminate training after this many evaluations without an improvement in the evaluation metric greater then early_stopping_delta.                                                        |
| encoding                         | str   | None                                                      | Specify an encoding to be used when reading text files.                                                                                                                                  |
| eval_batch_size                  | int   | 8                                                         | The evaluation batch size.                                                                                                                                                               |
| evaluate_during_training         | bool  | False                                                     | Set to True to perform evaluation while training models. Make sure eval data is passed to the training method if enabled.                                                                |
| evaluate_during_training_steps   | int   | 2000                                                      | Perform evaluation at every specified number of steps. A checkpoint model and the evaluation results will be saved.                                                                      |
| evaluate_during_training_verbose | bool  | False                                                     | Print results from evaluation during training.                                                                                                                                           |
| fp16                             | bool  | True                                                      | Whether or not fp16 mode should be used. Requires NVidia Apex library.                                                                                                                   |
| fp16_opt_level                   | str   | O1                                                        | Can be '01', '02', '03'. See the Apex docs for an explanation of the different optimization levels (opt_levels).                                                                         |
| gradient_accumulation_steps      | int   | 1                                                         | The number of training steps to execute before performing a optimizer.step(). Effectively increases the training batch size while sacrificing training time to lower memory consumption. |
| learning_rate                    | float | 4e-5                                                      | The learning rate for training.                                                                                                                                                          |
| logging_steps                    | int   | 50                                                        | Log training loss and learning at every specified number of steps.                                                                                                                       |
| manual_seed                      | int   | None                                                      | Set a manual seed if necessary for reproducible results.                                                                                                                                 |
| max_grad_norm                    | float | 1.0                                                       | Maximum gradient clipping.                                                                                                                                                               |
| max_seq_length                   | int   | 128                                                       | Maximum sequence length the model will support.                                                                                                                                          |
| multiprocessing_chunksize        | int   | 500                                                       | Number of examples sent to a CPU core at a time when using multiprocessing. Usually, the optimal value will be (roughly) `number of examples / process count`.                           |
| n_gpu                            | int   | 1                                                         | Number of GPUs to use.                                                                                                                                                                   |
| no_cache                         | bool  | False                                                     | Cache features to disk.                                                                                                                                                                  |
| no_save                          | bool  | False                                                     | If `True`, models will not be saved to disk.                                                                                                                                             |
| num_train_epochs                 | int   | 1                                                         | The number of epochs the model will be trained for.                                                                                                                                      |
| output_dir                       | str   | "outputs/"                                                | The directory where all outputs will be stored. This includes model checkpoints and evaluation results.                                                                                  |
| overwrite_output_dir             | bool  | False                                                     | If True, the trained model will be saved to the ouput_dir and will overwrite existing saved models in the same directory.                                                                |
| process_count                    | int   | cpu_count ()  -   2   if   cpu_count ()  >   2   else   1 | Number of cpu cores (processes) to use when converting examples to features. Default is (number of cores - 2) or 1 if (number of cores <= 2)                                             |
| reprocess_input_data             | bool  | True                                                      | If True, the input data will be reprocessed even if a cached file of the input data exists in the cache_dir.                                                                             |
| save_eval_checkpoints            | bool  | True                                                      | Save a model checkpoint for every evaluation performed.                                                                                                                                  |
| save_model_every_epoch           | bool  | True                                                      | Save a model checkpoint at the end of every epoch.                                                                                                                                       |
| save_steps                       | int   | 2000                                                      | Save a model checkpoint at every specified number of steps.                                                                                                                              |
| save_optimizer_and_scheduler     | bool  | True                                                      | Save optimizer and scheduler whenever they are available.                                                                                                                                  |
| silent                           | bool  | False                                                     | Disables progress bars.                                                                                                                                                                  |
| tensorboard_dir                  | str   | None                                                      | The directory where Tensorboard events will be stored during training. By default, Tensorboard events will be saved in a subfolder inside runs/ like runs/Dec02_09-32-58_36d9e58955b0/.  |
| train_batch_size                 | int   | 8                                                         | The training batch size.                                                                                                                                                                 |
| use_cached_eval_features         | bool  | False                                                     | Evaluation during training uses cached features. Setting this to False will cause features to be recomputed at every evaluation step.                                                    |
| use_early_stopping               | bool  | False                                                     | Use early stopping to stop training when early_stopping_metric doesn't improve (based on early_stopping_patience, and early_stopping_delta)                                              |
| use_multiprocessing              | bool  | True                                                      | If True, multiprocessing will be used when converting data into features. Disabling can reduce memory usage, but may substantially slow down processing.                                 |
| wandb_kwargs                     | dict  | {}                                                        | Dictionary of keyword arguments to be passed to the W&B project.                                                                                                                         |
| wandb_project                    | str   | None                                                      | Name of W&B project. This will log all hyperparameter values, training losses, and evaluation metrics to the given project.                                                              |
| warmup_ratio                     | float | 0.06                                                      | Ratio of total training steps where learning rate will "warm up". Overridden if `warmup_steps` is specified.                                                                             |
| warmup_steps                     | int   | 0                                                         | Number of training steps where learning rate will "warm up". Overrides `warmup_ratio`.                                                                                                   |
| weight_decay                     | int   | 0                                                         | Adds L2 penalty.                                                                                                                                                                         |


You can override any of these default values by either editing the dataclass attributes or by passing in a Python dict containing the appropriate key-value pairs when initializing a Simple Transformers model.

### Using the dataclass

```python
from simpletransformers.classification import ClassificationModel, ClassificationArgs

model_args = ClassificationArgs()
model_args.num_train_epochs = 5
model_args.learning_rate = 1e-4

model = ClassficationModel("bert", "bert-base-cased", args=model_args)
```

### Using a python dictionary

```python
from simpletransformers.classification import ClassificationModel


model_args = {
    "num_train_epochs": 5,
    "learning_rate": 1e-4,
}

model = ClassficationModel("bert", "bert-base-cased", args=model_args)
```

**Tip:** Using the dataclass approach has the benefits of IDE auto-completion as well as ensuring that there are no typos in arguments that could lead to unexpected behaviour.
{: .notice--success}

**Tip:** Both the dataclass and the dictionary approaches are interchangeable.
{: .notice--success}

## Tips and Tricks

### Using early stopping

Early stopping is a technique used to prevent model overfitting. In a nutshell, the idea is to periodically evaluate the performance of a model against a test dataset and terminate the training once the model stops improving on the test data.

The exact conditions for early stopping can be adjusted as needed using a model's configuration options.

**Note:** Refer the configuration options table for more details. (`early_stopping_consider_epochs`, `early_stopping_delta`, `early_stopping_metric`, `early_stopping_metric_minimize`, `early_stopping_patience`)
{: .notice--info}

You must set `use_early_stopping` to `True` in order to use early stopping.

```python
from simpletransformers.classification import ClassificationModel, ClassificationArgs


model_args = ClassificationArgs()
model_args.use_early_stopping = True
model_args.early_stopping_delta = 0.01
model_args.early_stopping_metric = "mcc"
model_args.early_stopping_metric_minimize = False
model_args.early_stopping_patience = 5
model_args.evaluate_during_training_steps = 1000

model = ClassficationModel("bert", "bert-base-cased", args=model_args)
```

With this configuration, the training will terminate if the `mcc` score of the model on the test data does not improve upon the best `mcc` score by at least `0.01` for 5 consecutive evaluations. An evaluation will occur once for every `1000` training steps.

**Pro tip:** You can use the evaluation during training functionality without invoking early stopping by setting `evaluate_during_training` to `True` while keeping `use_early_stopping` as `False`.
{: .notice--success}


### Additional evaluation metrics

Task-specific Simple Transformers models each have their own default metrics that will be calculated when a model is evaluated
on a dataset. The default metrics have been chosen according to the task, usually by looking at the metrics used in standard benchmarks for that task.

However, it is likely that you will wish to calculate your own metrics depending on your particular use case. To facilitate this, all `eval_model()` and `train_model()` methods in Simple Transformers accepts keyword-arguments consisting of the name of the metric (str), and the metric function itself. The metric function should accept two inputs, the true labels and the model predictions (sklearn format).


```python
from simpletransformers.classification import ClassificationModel
import sklearn


model = ClassficationModel("bert", "bert-base-cased")

model.train_model(train_df, acc=sklearn.metrics.accuracy_score)

model.eval_model(eval_df, acc=sklearn.metrics.accuracy_score)
```

**Pro tip:** You can combine the additional evaluation metrics functionality with early stopping by setting the name of your metrics function as the `early_stopping_metric`.
{: .notice--success}


### Hyperparameter optimization

Machine learning models can be very sensitive to the hyperparameters used to train them. While large models like Transformers can perform well across a relatively wider hyperparameter range, they can also break completely under certain conditions (like training with large learning rates for many iterations).

**Hint:** We can define two kinds of parameters used to train Transformer models. The first is the learned parameters (like the model weights) and the second is hyperparameters. To give a high-level description of the two kinds of parameters, the hyperparameters (learning rate, batch sizes, etc.) are used to control the process of *learning* learned parameters.
{: .notice--success}

Choosing a good set of hyperparameter values plays a huge role in developing a state-of-the-art model. Because of this, Simple Transformers has native support for the excellent [W&B Sweeps](https://docs.wandb.com/sweeps) feature for autometed hyperparameter optimization.

How to perform hyperparameter optimization with Simple Transformers and W&B Sweeps (Adapted from W&B [docs](https://docs.wandb.com/sweeps)):

#### 1. Setup the sweep

The sweep can be configured through a Python dictionary (`sweep_config`). The dictionary contains at least 3 keys;

1. `method` -- Specifies the search strategy

    | `method` | Meaning                                                                                                                                                                                      |
    |--------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
    | grid   | Grid search iterates over all possible combinations of parameter values.                                                                                                                     |
    | random | Random search chooses random sets of values.                                                                                                                                                 |
    | bayes  | Bayesian Optimization uses a gaussian process to model the function and then chooses parameters to optimize probability of improvement. This strategy requires a metric key to be specified. |

2. `metric` -- Specifies the metric to be optimized

    *This should be a metric that is logged to W&B by the training script*

    The `metric` key of the `sweep_config` points to another Python dictionary containing the `name`, `goal`, and (optionally) `target`.

    | sub-key | Meaning                                                                                                                                                                                                                                                                             |
    |---------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
    | name    | Name of the metric to optimize                                                                                                                                                                                                                                                      |
    | goal    | `"minimize"` or `"maximize"` (Default is `"minimize"`)                                                                                                                                                                                                                                          |
    | target  | Value that you'd like to achieve for the metric you're optimizing. When any run in the sweep achieves that target value, the sweep's state will be set to "Finished." This means all agents with active runs will finish those jobs, but no new runs will be launched in the sweep. |

3. `parameters` -- Specifies the hyperparameters and their values to explore

    The `parameters` key of the `sweep_config` points to another Python dictionary which contains all the hyperparameters to be optimized and their possible values. Generally, these will be any combination of the `model_args` for the particular Simple Transformers model.

    W&B offers a variety of ways to define the possible values for each parameter, all of which can be found in the [W&B docs](https://docs.wandb.com/sweeps/configuration#parameters). The possible values are also represented using a Python dictionary. Two common methods are given below.

    1. Discrete values

        A dictionary with the key `values` pointing to a Python list of discrete values.

    2. Range of values

        A dictionary with the two keys `min` and `max` which specifies the minimum and maximum values of the range. *The range is continuous if `min` and `max` are floats and discrete if `min` and `max` are ints.*

Example `sweep_config`:

```python
sweep_config = {
    "method": "bayes",  # grid, random
    "metric": {"name": "train_loss", "goal": "minimize"},
    "parameters": {
        "num_train_epochs": {"values": [2, 3, 5]},
        "learning_rate": {"min": 5e-5, "max": 4e-4},
    },
}
```

#### 2. Initialize the sweep

Initialize a W&B sweep with the config defined earlier.

```python
sweep_id = wandb.sweep(sweep_config, project="Simple Sweep")
```

#### 3. Prepare the data and default model configuration

In order to run our sweep, we must get our data ready. This is identical to how you would normally set up datasets for training a Simple Transformers model.

For example;

```python
# Preparing train data
train_data = [
    ["Aragorn was the heir of Isildur", "true"],
    ["Frodo was the heir of Isildur", "false"],
]
train_df = pd.DataFrame(train_data)
train_df.columns = ["text", "labels"]

# Preparing eval data
eval_data = [
    ["Theoden was the king of Rohan", "true"],
    ["Merry was the king of Rohan", "false"],
]
eval_df = pd.DataFrame(eval_data)
eval_df.columns = ["text", "labels"]
```

Next, we can set up the default configuration for the Simple Transformers model. This would include any `args` that are not being optimized through the sweep.

**Hint:** As a rule of thumb, it might be a good idea to set all of `reprocess_input_data`, `overwrite_output_dir`, and `no_save` to `True` when running sweeps.
{: .notice--success}

```python
model_args = ClassificationArgs()
model_args.reprocess_input_data = True
model_args.overwrite_output_dir = True
model_args.evaluate_during_training = True
model_args.manual_seed = 4
model_args.use_multiprocessing = True
model_args.train_batch_size = 16
model_args.eval_batch_size = 8
model_args.labels_list = ["true", "false"]
```

#### 4. Set up the training function

W&B will call this function to run the training for a particular sweep run. This function must perform 3 critical tasks.

1. Initialize the `wandb` run
2. Initialize a Simple Transformers model and pass in `sweep_config=wandb.config` as a `kwarg`.
3. Run the training for the Simple Transformers model.

*`wandb.config` contains the hyperparameter values for the current sweeps run. Simple Transformers will update the model `args` accordingly.*

An example training function is shown below.

```python
def train():
    # Initialize a new wandb run
    wandb.init()

    # Create a TransformerModel
    model = ClassificationModel(
        "roberta",
        "roberta-base",
        use_cuda=True,
        args=model_args,
        sweep_config=wandb.config,
    )

    # Train the model
    model.train_model(train_df, eval_df=eval_df)

    # Evaluate the model
    model.eval_model(eval_df)

    # Sync wandb
    wandb.join()

```

In addition to the 3 tasks outlined earlier, the function also performs an evaluation and manually syncs the W&B run.

**Hint:** This function can be reused across any Simple Transformers task by simply replacing `ClassificationModel` with the appropriate model class.
{: .notice--success}


#### 5. Run the sweeps

The following line will execute the sweeps.

```python
wandb.agent(sweep_id, train)
```

#### Putting it all together

```python
import logging

import pandas as pd
import sklearn

import wandb
from simpletransformers.classification import (
    ClassificationArgs,
    ClassificationModel,
)

sweep_config = {
    "method": "bayes",  # grid, random
    "metric": {"name": "train_loss", "goal": "minimize"},
    "parameters": {
        "num_train_epochs": {"values": [2, 3, 5]},
        "learning_rate": {"min": 5e-5, "max": 4e-4},
    },
}

sweep_id = wandb.sweep(sweep_config, project="Simple Sweep")

logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

# Preparing train data
train_data = [
    ["Aragorn was the heir of Isildur", "true"],
    ["Frodo was the heir of Isildur", "false"],
]
train_df = pd.DataFrame(train_data)
train_df.columns = ["text", "labels"]

# Preparing eval data
eval_data = [
    ["Theoden was the king of Rohan", "true"],
    ["Merry was the king of Rohan", "false"],
]
eval_df = pd.DataFrame(eval_data)
eval_df.columns = ["text", "labels"]

model_args = ClassificationArgs()
model_args.reprocess_input_data = True
model_args.overwrite_output_dir = True
model_args.evaluate_during_training = True
model_args.manual_seed = 4
model_args.use_multiprocessing = True
model_args.train_batch_size = 16
model_args.eval_batch_size = 8
model_args.labels_list = ["true", "false"]


def train():
    # Initialize a new wandb run
    wandb.init()

    # Create a TransformerModel
    model = ClassificationModel(
        "roberta",
        "roberta-base",
        use_cuda=True,
        args=model_args,
        sweep_config=wandb.config,
    )

    # Train the model
    model.train_model(train_df, eval_df=eval_df)

    # Evaluate the model
    model.eval_model(eval_df)

    # Sync wandb
    wandb.join()


wandb.agent(sweep_id, train)

```

**Hint:** This script can also be found in the `examples` directory of the Github repo.
{: .notice--success}

To visualize your sweep results, open the project on W&B. Please refer to [W&B docs](https://docs.wandb.com/sweeps/visualize-sweep-results) for more details on understanding the results.


### Custom parameter groups (freezing layers)

Simple Transformers supports custom parameter groups which can be used to set different learning rates for different layers in a model, freeze layers, train only the final layer, etc.

All Simple Transformers models supports the following three configuration options for setting up custom parameter groups.

#### `custom_parameter_groups`

`custom_parameter_groups` offers the most granular configuration option. This should be a list of Python dicts where each dict contains a `params` key and any other optional keys matching the keyword arguments accepted by the optimizer (e.g. `lr`, `weight_decay`). The value for the `params` key should be a list of named parameters (e.g. `["classifier.weight", "bert.encoder.layer.10.output.dense.weight"]`)

**Hint:** All Simple Transformers models have a `get_named_parameters()` method that returns a list of all parameter names in the model.
{: .notice--success}

```python
model_args = ClassificationArgs()
model_args.custom_parameter_groups = [
    {
        "params": ["classifier.weight", "bert.encoder.layer.10.output.dense.weight"],
        "lr": 1e-2,
    }
]
```

#### `custom_layer_parameters`

`custom_layer_parameters` makes it more convenient to set the optimizer options for a given layer or set of layers. This should be a list of Python dicts where each dict contains a `layer` key and any other optional keys matching the keyword arguments accepted by the optimizer (e.g. `lr`, `weight_decay`). The value for the `layer` key should be an `int` (must be numeric) which specifies the layer (e.g. `0`, `1`, `11`).

```python
model_args = ClassificationArgs()
model_args.custom_layer_parameters = [
    {
        "layer": 10,
        "lr": 1e-3,
    },
    {
        "layer": 0,
        "lr": 1e-5,
    },
]
```

**Note:** Any named parameters specified through `custom_layer_parameters` with `bias` or `LayerNorm.weight` in the name will have their `weight_decay` set to `0.0`. This also happens for any parameters **not specified** in either `custom_parameter_groups` or in `custom_layer_parameters` but **does not happen** for parameters specified through `custom_parameter_groups`.
{: .notice--info}

{% capture notice-text %}

Note that `custom_parameter_groups` has *higher priority* than `custom_layer_parameters` as `custom_parameter_groups` is more specific. If a parameter specificed in `custom_parameter_groups` also happens to be in a layer specified in `custom_layer_parameters`, that particular parameter will be assigned to the parameter group specified in `custom_parameter_groups`.

For example:

```python
model_args = ClassificationArgs()
model_args.custom_layer_parameters = [
    {
        "layer": 10,
        "lr": 1e-3,
    },
    {
        "layer": 0,
        "lr": 1e-5,
    },
]
model_args.custom_parameter_groups = [
    {
        "params": ["classifier.weight", "bert.encoder.layer.10.output.dense.weight"],
        "lr": 1e-2,
    }
]
```

Here, `"bert.encoder.layer.10.output.dense.weight"` is specified in both the `custom_parameter_groups` and the `custom_layer_parameters`. However, `"bert.encoder.layer.10.output.dense.weight"` will have a `lr` of `1e-2` due to the higher precedence of `custom_parameter_groups`.

{% endcapture %}

<div class="notice--success">
  <h4>Multi-label vs Multi-class:</h4>
  {{ notice-text | markdownify }}
</div>

**Hint:** Any parameters not specified in either `custom_parameter_groups` or in `custom_layer_parameters` will be assigned the general values from the model args.
{: .notice--success}

#### `train_custom_parameters_only`

The `train_custom_parameters_only` option is used to facilitate the training of specific parameters only. If `train_custom_parameters_only` is set to `True`, only the parameters specified in either `custom_parameter_groups` or in `custom_layer_parameters` will be trained.

For example, to train only the Classification layers of a `ClassificationModel`:

```python
from simpletransformers.classification import ClassificationModel, ClassificationArgs
import pandas as pd
import logging


logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

# Preparing train data
train_data = [
    ["Aragorn was the heir of Isildur", 1],
    ["Frodo was the heir of Isildur", 0],
]
train_df = pd.DataFrame(train_data)
train_df.columns = ["text", "labels"]

# Preparing eval data
eval_data = [
    ["Theoden was the king of Rohan", 1],
    ["Merry was the king of Rohan", 0],
]
eval_df = pd.DataFrame(eval_data)
eval_df.columns = ["text", "labels"]

# Train only the classifier layers
model_args = ClassificationArgs()
model_args.train_custom_parameters_only = True
model_args.custom_parameter_groups = [
    {
        "params": ["classifier.weight"],
        "lr": 1e-3,
    },
    {
        "params": ["classifier.bias"],
        "lr": 1e-3,
        "weight_decay": 0.0,
    },
]
# Create a ClassificationModel
model = ClassificationModel(
    "bert", "bert-base-cased", args=model_args
)

# Train the model
model.train_model(train_df)

```

## Options For Downloading Pre-Trained Models

Most Simple Transformers models will use the `from_pretrained()` [method](https://huggingface.co/transformers/main_classes/model.html#transformers.PreTrainedModel.from_pretrained) from the Hugging Face Transformers library to download pre-trained models. You can pass `kwargs` to this method to configure things like proxies and force downloading (refer to method link above).

You can pass these `kwargs` when initializing a Simple Transformers task-specific model to access the same functionality. For example, if you are behind a firewall and need to set the proxy settings;

```python
model = ClassficationModel(
    "bert",
    "bert-base-cased",
    proxies={"http": "foo.bar:3128", "http://hostname": "foo.bar:4012"}
)
```
