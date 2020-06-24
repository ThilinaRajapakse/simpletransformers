---
title: Language Modeling  Minimal Start
permalink: /docs/lm-minimal-start/
excerpt: "Minimal start for Language Modeling tasks."
last_modified_at: 2020-05-02 17:58:53
toc: True
---

Simple Transformers currently supports 3 *pre-training objectives*.

- Masked Language Modeling (MLM) - Used with `bert, camembert, distilbert, roberta`
- Causal Language Modeling (CLM) - Used with `gpt2, openai-gpt`
- ELECTRA - Used with `electra`

Because of this, you need to specify the *pre-training objective* when training or fine-tuning a Language Model. By default, MLM is used. Setting `mlm: False` in the model args dict will switch the *pre-training objective* to CLM. Although ELECTRA used its own unique *pre-training objective*, the inputs to the generator model are masked in the same way as with the other MLM models. Therefore, `mlm` can be set to `True` (done by default) in the `args` dict for ELECTRA models.

## Language Model Fine-Tuning

Refer to [Language Model Fine-Tuning](/docs/lm-specifics/#language-model-fine-tuning) section in the Language Model Specifics section for details.

Refer to [Language Model Data Formats](/docs/lm-data-formats/) for the correct input data formats.

### Fine-Tuning a BERT model (MLM)

```python
import logging

from simpletransformers.language_modeling import (
    LanguageModelingModel,
    LanguageModelingArgs,
)


logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

model_args = LanguageModelingArgs()
model_args.reprocess_input_data = True
model_args.overwrite_output_dir = True
model_args.num_train_epochs = 1
model_args.dataset_type = "simple"

train_file = "data/train.txt"
test_file = "data/test.txt"

model = LanguageModelingModel(
    "bert", "bert-base-cased", args=model_args
)

# Train the model
model.train_model(train_file, eval_file=test_file)

# Evaluate the model
result = model.eval_model(test_file)

```

### Fine-Tuning a GPT-2 model (CLM)

```python
import logging

from simpletransformers.language_modeling import (
    LanguageModelingModel,
    LanguageModelingArgs,
)


logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)


model_args.reprocess_input_data = True
model_args.overwrite_output_dir = True
model_args.num_train_epochs = 1
model_args.dataset_type = "simple"
model_args.mlm = False  # mlm must be False for CLM

train_file = "data/train.txt"
test_file = "data/test.txt"

model = LanguageModelingModel(
    "gpt2", "gpt2-medium", args=model_args
)

# Train the model
model.train_model(train_file, eval_file=test_file)

# Evaluate the model
result = model.eval_model(test_file)

```


### Fine-Tuning an ELECTRA model

```python
import logging

from simpletransformers.language_modeling import (
    LanguageModelingModel,
    LanguageModelingArgs,
)


logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)


model_args.reprocess_input_data = True
model_args.overwrite_output_dir = True
model_args.num_train_epochs = 1
model_args.dataset_type = "simple"

train_file = "data/train.txt"
test_file = "data/test.txt"

# Google released separate generator/discriminator models
model = LanguageModelingModel(
    "electra",
    "electra",
    generator_name="google/electra-small-generator",
    discriminator_name="google/electra-large-discriminator",
    args=model_args,
)

# Train the model
model.train_model(train_file, eval_file=test_file)

# Evaluate the model
result = model.eval_model(test_file)

```


## Language Model Training From Scratch

Refer to [Training a Language Model From Scratch](/docs/lm-specifics/#training-a-language-model-from-scratch) section in the Language Model Specifics section for details.

Refer to [Language Model Data Formats](/docs/lm-data-formats/) for the correct input data formats.

When training a Language Model from scratch, the `model_name` parameter is set to `None`. In addition, the `train_files` argument is required (see [here](/docs/lm-data-formats/)).

### Training a BERT model (MLM) from scratch

```python
import logging

from simpletransformers.language_modeling import (
    LanguageModelingModel,
    LanguageModelingArgs,
)


logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)


model_args.reprocess_input_data = True
model_args.overwrite_output_dir = True
model_args.num_train_epochs = 1
model_args.dataset_type = "simple"
model_args.vocab_size = 30000

train_file = "data/train.txt"
test_file = "data/test.txt"

model = LanguageModelingModel(
    "bert", None, args=model_args, train_files=train_file
)

# Train the model
model.train_model(train_file, eval_file=test_file)

# Evaluate the model
result = model.eval_model(test_file)

```

### Training a GPT-2 model (CLM) from scratch

```python
import logging

from simpletransformers.language_modeling import (
    LanguageModelingModel,
    LanguageModelingArgs,
)


logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)


model_args.reprocess_input_data = True
model_args.overwrite_output_dir = True
model_args.num_train_epochs = 1
model_args.dataset_type = "simple"
model_args.mlm = False  # mlm must be False for CLM
model_args.vocab_size = 30000

train_file = "data/train.txt"
test_file = "data/test.txt"

model = LanguageModelingModel(
    "gpt2", None, args=model_args, train_files=train_file
)

# Train the model
model.train_model(train_file, eval_file=test_file)

# Evaluate the model
result = model.eval_model(test_file)

```

### Training an ELECTRA model from scratch

```python
import logging

from simpletransformers.language_modeling import (
    LanguageModelingModel,
    LanguageModelingArgs,
)


logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)


model_args.reprocess_input_data = True
model_args.overwrite_output_dir = True
model_args.num_train_epochs = 1
model_args.dataset_type = "simple"
model_args.vocab_size = 30000

train_file = "data/train.txt"
test_file = "data/test.txt"

model = LanguageModelingModel(
    "electra",
    None,
    args=model_args,
    train_files=train_file
)

# Train the model
model.train_model(train_file, eval_file=test_file)

# Evaluate the model
result = model.eval_model(test_file)

```


## Guides

- [A Basic Overview of Language Model Fine-Tuning](https://medium.com/skilai/language-model-fine-tuning-for-pre-trained-transformers-b7262774a7ee?source=friends_link&sk=1f9f834447db7e748ae333c6490064fa)
- [Fine-Tuning a GPT-2 Model For Domain-Specific Language Generation](https://medium.com/swlh/learning-to-write-language-generation-with-gpt-2-2a13fa249024?source=friends_link&sk=97192355cd3d8ba6cfd8b782d7380d86)
- [Training a Language Model From Scratch](https://towardsdatascience.com/understanding-electra-and-training-an-electra-language-model-3d33e3a9660d?source=friends_link&sk=2b4b4a79954e3d7c84ab863efaea8c65)