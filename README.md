# Simple Transformers


This library is based on the [Pytorch-Transformers](https://github.com/huggingface/pytorch-transformers) library by HuggingFace. Using this library, you can quickly train and evaluate Transformer models. Only 3 lines of code are needed to initialize a model, train the model, and evaluate the model.


Table of contents
=================

<!--ts-->
   * [Setup](#Setup)
      * [With Conda](#with-conda)
   * [Usage](#usage)
      * [Minimal Start](#minimal-start)
      * [Default Settings](#default-settings)
      * [TransformerModel](#transformermodel)
      * [Current Pretrained Models](#current-pretrained-models)
   * [Acknowledgements](#acknowledgements)
<!--te-->

## Setup

### With Conda

1. Install Anaconda or Miniconda Package Manager from [here](https://www.anaconda.com/distribution/)
2. Create a new virtual environment and install packages.  
`conda create -n transformers python pandas tqdm`  
`conda activate transformers`  
If using cuda:  
&nbsp;&nbsp;&nbsp;&nbsp;`conda install pytorch cudatoolkit=10.0 -c pytorch`  
else:  
&nbsp;&nbsp;&nbsp;&nbsp;`conda install pytorch cpuonly -c pytorch`  
`conda install -c anaconda scipy`  
`conda install -c anaconda scikit-learn`  
`pip install transformers`  
`pip install tensorboardx`  

3. Install simpletransformers.  
`pip install simpletransformers`  

## Usage

### Minimal Start

```
from simpletransformers.model import TransformerModel
import pandas as pd


# Train and Evaluation data needs to be in a Pandas Dataframe of two columns. The first column is the text with type str, and the second column is the label with type int.
train_data = [['Example sentence belonging to class 1', 1], ['Example sentence belonging to class 0', 0]]
train_df = pd.DataFrame(train_data)

eval_data = [['Example eval sentence belonging to class 1', 1], ['Example eval sentence belonging to class 0', 0]]
eval_df = pd.DataFrame(eval_data)

# Create a TransformerModel
model = TransformerModel('roberta', 'roberta-base')

# Train the model
model.train_model(train_df)

# Evaluate the model
result, model_outputs, wrong_predictions = model.eval_model(eval_df)
```

To make predictions on arbitary data, the `predict(to_predict)` function can be used. For a list of text, it returns the model predictions and the raw model outputs.

```
predictions = model.predict(['Some arbitary sentence'])
```

Please refer to [this Medium article](https://towardsdatascience.com/simple-transformers-introducing-the-easiest-bert-roberta-xlnet-and-xlm-library-58bf8c59b2a3?source=friends_link&sk=40726ceeadf99e1120abc9521a10a55c) for an example of using the library on the Yelp Reviews Dataset.

### Default Settings


The default args used are given below. Any of these can be overridden by passing a dict containing the corresponding key: value pairs to the the init method of TransformerModel.

```
self.args = {
   'model_type':  'roberta',
   'model_name': 'roberta-base',
   'output_dir': 'outputs/',
   'cache_dir': 'cache/',

   'fp16': True,
   'fp16_opt_level': 'O1',
   'max_seq_length': 128,
   'train_batch_size': 8,
   'eval_batch_size': 8,
   'gradient_accumulation_steps': 1,
   'num_train_epochs': 1,
   'weight_decay': 0,
   'learning_rate': 4e-5,
   'adam_epsilon': 1e-8,
   'warmup_ratio': 0.06,
   'warmup_steps': 0,
   'max_grad_norm': 1.0,

   'logging_steps': 50,
   'evaluate_during_training': False,
   'save_steps': 2000,
   'eval_all_checkpoints': True,
   'use_tensorboard': True,

   'overwrite_output_dir': False,
   'reprocess_input_data': False,
}
```

### Args Explained

#### *output_dir: str*
The directory where all outputs will be stored. This includes model checkpoints and evaluation results.

#### *cache_dir: str*
The directory where cached files will be saved.

#### *fp16: bool*
Whether or not fp16 mode should be used. Requires NVidia Apex library.

#### *fp16_opt_level: str*
Can be '01', '02', '03'. See the [Apex docs](https://nvidia.github.io/apex/amp.html) for an explanation of the different optimization levels (opt_levels).

#### *max_seq_length: int*
Maximum sequence level the model will support.

#### *train_batch_size: int*
The training batch size.

#### *gradient_accumulation_steps: int*
The number of training steps to execute before performing a `optimizer.step()`. Effectively increases the training batch size while sacrificing training time to lower memory consumption.

#### *eval_batch_size: int*
The evaluation batch size.

#### *num_train_epochs: int*
The number of epochs the model will be trained for.

#### *weight_decay: float*
Adds L2 penalty.

#### *learning_rate: float*
The learning rate for training.

#### *adam_epsilon: float*
Epsilon hyperparameter used in AdamOptimizer.

#### *max_grad_norm: float*
Maximum gradient clipping.

#### *logging_steps: int*
Log training loss and learning at every specified number of steps.

#### *save_steps: int*
Save a model checkpoint at every specified number of steps.

#### *overwrite_output_dir: bool*
If True, the trained model will be saved to the ouput_dir and will overwrite existing saved models in the same directory.

#### *reprocess_input_data: bool*
If True, the input data will be reprocessed even if a cached file of the input data exists in the cache_dir.

#### *process_count: int*
Number of cpu cores (processes) to use when converting examples to features. Default is (number of cores - 2) or 1 if (number of cores <= 2)

### TransformerModel

`class simpletransformers.model.TransformerModel (model_type, model_name, args=None, use_cuda=True)`  
This is the main class of this library. All configuration, training, and evaluation is performed using this class.

`Class attributes`
* `tokenizer`: The tokenizer to be used.
* `model`: The model to be used.
* `device`: The device on which the model will be trained and evaluated.
* `results`: A python dict of past evaluation results for the TransformerModel object.
* `args`: A python dict of arguments used for training and evaluation.

`Parameters`
* `model_type`: (required) str - The type of model to use. Currently, BERT, XLNet, XLM, and RoBERTa models are available.
* `model_name`: (required) str - The exact model to use. See [Current Pretrained Models](#current-pretrained-models) for all available models.
* `args`: (optional) python dict - A dictionary containing any settings that should be overwritten from the default values.
* `use_cuda`: (optional) bool - Default = True. Flag used to indicate whether CUDA should be used.

`class methods`  
**`train_model(self, train_df, output_dir=None)`**

Trains the model using 'train_df'

Args:  
>train_df: Pandas Dataframe (no header) of two columns, first column containing the text, and the second column containing the label. The model will be trained on this Dataframe.

>output_dir: The directory where model files will be saved. If not given, self.args['output_dir'] will be used.

Returns:  
>None

**`eval_model(self, eval_df, output_dir=None, verbose=False)`**

Evaluates the model on eval_df. Saves results to output_dir.

Args:  
>eval_df: Pandas Dataframe (no header) of two columns, first column containing the text, and the second column containing the label. The model will be evaluated on this Dataframe.

>output_dir: The directory where model files will be saved. If not given, self.args['output_dir'] will be used.  

>verbose: If verbose, results will be printed to the console on completion of evaluation.  

Returns:  
>result: Dictionary containing evaluation results. (Matthews correlation coefficient, tp, tn, fp, fn)  

>model_outputs: List of model outputs for each row in eval_df  

>wrong_preds: List of InputExample objects corresponding to each incorrect prediction by the model  

**`predict(self, to_predict)`**

Performs predictions on a list of text.

Args:
>to_predict: A python list of text (str) to be sent to the model for prediction.

Returns:
>preds: A python list of the predictions (0 or 1) for each text.
>model_outputs: A python list of the raw model outputs for each text.


**`train(self, train_dataset, output_dir)`**

Trains the model on train_dataset.
*Utility function to be used by the train_model() method. Not intended to be used directly.*

**`evaluate(self, eval_df, output_dir, prefix="")`**

Evaluates the model on eval_df.
*Utility function to be used by the eval_model() method. Not intended to be used directly*

**`load_and_cache_examples(self, examples, evaluate=False)`**

Converts a list of InputExample objects to a TensorDataset containing InputFeatures. Caches the InputFeatures.
*Utility function for train() and eval() methods. Not intended to be used directly*

**`List of InputExample objects corresponding to each incorrect prediction by the model`**

Computes the evaluation metrics for the model predictions.

Args:
>preds: Model predictions  

>labels: Ground truth labels  

>eval_examples: List of examples on which evaluation was performed  

Returns:
>result: Dictionary containing evaluation results. (Matthews correlation coefficient, tp, tn, fp, fn)  

>wrong: List of InputExample objects corresponding to each incorrect prediction by the model  


### Current Pretrained Models

The table below shows the currently available model types and their models. You can use any of these by setting the `model_type` and `model_name` in the `args` dictionary. For more information about pretrained models, see [HuggingFace docs](https://huggingface.co/pytorch-transformers/pretrained_models.html).

| Architecture        | Model Type           | Model Name  | Details  |
| :------------- |:----------| :-------------| :-----------------------------|
| BERT      | bert | bert-base-uncased | 12-layer, 768-hidden, 12-heads, 110M parameters.<br>Trained on lower-cased English text. |
| BERT      | bert | bert-large-uncased | 24-layer, 1024-hidden, 16-heads, 340M parameters.<br>Trained on lower-cased English text. |
| BERT      | bert | bert-base-cased | 12-layer, 768-hidden, 12-heads, 110M parameters.<br>Trained on cased English text. |
| BERT      | bert | bert-large-cased | 24-layer, 1024-hidden, 16-heads, 340M parameters.<br>Trained on cased English text. |
| BERT      | bert | bert-base-multilingual-uncased | (Original, not recommended) 12-layer, 768-hidden, 12-heads, 110M parameters. <br>Trained on lower-cased text in the top 102 languages with the largest Wikipedias |
| BERT      | bert | bert-base-multilingual-cased | (New, recommended) 12-layer, 768-hidden, 12-heads, 110M parameters.<br>Trained on cased text in the top 104 languages with the largest Wikipedias |
| BERT      | bert | bert-base-chinese | 12-layer, 768-hidden, 12-heads, 110M parameters. <br>Trained on cased Chinese Simplified and Traditional text. |
| BERT      | bert | bert-base-german-cased | 12-layer, 768-hidden, 12-heads, 110M parameters. <br>Trained on cased German text by Deepset.ai |
| BERT      | bert | bert-large-uncased-whole-word-masking | 24-layer, 1024-hidden, 16-heads, 340M parameters. <br>Trained on lower-cased English text using Whole-Word-Masking |
| BERT      | bert | bert-large-cased-whole-word-masking | 24-layer, 1024-hidden, 16-heads, 340M parameters. <br>Trained on cased English text using Whole-Word-Masking |
| BERT      | bert | bert-large-uncased-whole-word-masking-finetuned-squad | 24-layer, 1024-hidden, 16-heads, 340M parameters. <br>The bert-large-uncased-whole-word-masking model fine-tuned on SQuAD |
| BERT      | bert | bert-large-cased-whole-word-masking-finetuned-squad | 24-layer, 1024-hidden, 16-heads, 340M parameters <br>The bert-large-cased-whole-word-masking model fine-tuned on SQuAD |
| BERT      | bert | bert-base-cased-finetuned-mrpc | 12-layer, 768-hidden, 12-heads, 110M parameters. <br>The bert-base-cased model fine-tuned on MRPC |
| XLNet      | xlnet | xlnet-base-cased | 12-layer, 768-hidden, 12-heads, 110M parameters. <br>XLNet English model |
| XLNet      | xlnet | xlnet-large-cased | 24-layer, 1024-hidden, 16-heads, 340M parameters. <br>XLNet Large English model |
| XLM      | xlm | xlm-mlm-en-2048 | 12-layer, 2048-hidden, 16-heads <br>XLM English model |
| XLM      | xlm | xlm-mlm-ende-1024 | 6-layer, 1024-hidden, 8-heads <br>XLM English-German Multi-language model |
| XLM      | xlm | xlm-mlm-enfr-1024 | 6-layer, 1024-hidden, 8-heads <br>XLM English-French Multi-language model |
| XLM      | xlm | xlm-mlm-enro-1024 | 6-layer, 1024-hidden, 8-heads <br>XLM English-Romanian Multi-language model |
| XLM      | xlm | xlm-mlm-xnli15-1024 | 12-layer, 1024-hidden, 8-heads <br>XLM Model pre-trained with MLM on the 15 XNLI languages |
| XLM      | xlm | xlm-mlm-tlm-xnli15-1024 | 12-layer, 1024-hidden, 8-heads <br>XLM Model pre-trained with MLM + TLM on the 15 XNLI languages |
| XLM      | xlm | xlm-clm-enfr-1024 | 12-layer, 1024-hidden, 8-heads <br>XLM English model trained with CLM (Causal Language Modeling) |
| XLM      | xlm | xlm-clm-ende-1024 | 6-layer, 1024-hidden, 8-heads <br>XLM English-German Multi-language model trained with CLM (Causal Language Modeling) |
| RoBERTa      | roberta | roberta-base | 125M parameters <br>RoBERTa using the BERT-base architecture |
| RoBERTa      | roberta | roberta-large | 24-layer, 1024-hidden, 16-heads, 355M parameters <br>RoBERTa using the BERT-large architecture |
| RoBERTa      | roberta | roberta-large-mnli | 24-layer, 1024-hidden, 16-heads, 355M parameters <br>roberta-large fine-tuned on MNLI. |

## Acknowledgements

None of this would have been possible without the hard work by the HuggingFace team in developing the [Pytorch-Transformers](https://github.com/huggingface/pytorch-transformers) library.
