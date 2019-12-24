[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0) [![Downloads](https://pepy.tech/badge/simpletransformers)](https://pepy.tech/project/simpletransformers)

# Simple Transformers

This library is based on the [Transformers](https://github.com/huggingface/transformers) library by HuggingFace. Simple Transformers lets you quickly train and evaluate Transformer models. Only 3 lines of code are needed to initialize a model, train the model, and evaluate a model. Currently supports Sequence Classification, Token Classification (NER), and Question Answering.

# Table of contents

<!--ts-->
* [Setup](#Setup)
    * [With Conda](#with-conda)
* [Usage](#usage)
* [Text Classification](#text-classification)
    * [Minimal Start for Binary Classification](#minimal-start-for-binary-classification)
    * [Minimal Start for Multiclass Classification](#minimal-start-for-multiclass-classification)
    * [Minimal Start for Multilabel Classification](#minimal-start-for-multilabel-classification)
    * [Real Dataset Examples](#real-dataset-examples)
    * [ClassificationModel](#classificationmodel)
* [Named Entity Recognition](#named-entity-recognition)
    * [Minimal Start](#minimal-start)
    * [Real Dataset Examples](#real-dataset-examples-1)
    * [NERModel](#nermodel)
* [Question Answering](#question-answering)
    * [Data Format](#data-format)
    * [Minimal Start](#minimal-example)
    * [Real Dataset Examples](#real-dataset-examples-2)
    * [QuestionAnsweringModel](#questionansweringmodel)
* [Experimental Features](#experimental-features)
    * [Sliding Window For Long Sequences](#sliding-window-for-long-sequences)
* [Loading Saved Models](#loading-saved-models)
* [Default Settings](#default-settings)
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
&nbsp;&nbsp;&nbsp;&nbsp;`conda install pytorch cudatoolkit=10.1 -c pytorch`  
else:  
&nbsp;&nbsp;&nbsp;&nbsp;`conda install pytorch cpuonly -c pytorch`  

3. Install Apex if you are using fp16 training. Please follow the instructions [here](https://github.com/NVIDIA/apex). (Installing Apex from pip has caused issues for several people.)

4. Install simpletransformers.  
`pip install simpletransformers`  

## Usage

Most available hyperparameters are common for all tasks. Any special hyperparameters will be listed in the docs section for the corresponding class. See [Default Settings](#default-settings) and [Args Explained](#args-explained) sections for more information.

Example scripts can be found in the `examples` directory.

See the [Changelog](https://github.com/ThilinaRajapakse/simpletransformers/blob/master/CHANGELOG.md) for up-to-date changes to the project.

### Structure

_The file structure has been updated starting with version 0.6.0. This should only affect import statements. The old import paths should still be functional although it is recommended to use the updated paths given below and in the minimal start examples_.

* `simpletransformers.classification` - Includes all Classification models.
  * `ClassificationModel`
  * `MultiLabelClassificationModel`
* `simpletransformers.ner` - Includes all Named Entity Recognition models.
  * `NERModel`
* `simpletransformers.question_answering` - Includes all Question Answering models.
  * `QuestionAnsweringModel`

---

## Text Classification

Supports Binary Classification, Multiclass Classification, and Multilabel Classification.

Supported model types:

* BERT
* RoBERTa
* XLNet
* XLM
* DistilBERT
* ALBERT
* CamemBERT @[manueltonneau](https://github.com/manueltonneau)

### Task Specific Notes

* Set `'sliding_window': True` in `args` to prevent text being truncated. The default *stride* is `'stride': 0.8` which is `0.8 * max_seq_length`. Training text will be split using a sliding window and each window will be assigned the label from the orignal text. During evaluation and prediction, the mode of the predictions for each window will be the final prediction on each sample. The `tie_value` (default `1`) will be used in the case of a tie.  
*Currently not available for Multilabel Classification*

#### Minimal Start for Binary Classification

```
from simpletransformers.classification import ClassificationModel
import pandas as pd


# Train and Evaluation data needs to be in a Pandas Dataframe of two columns. The first column is the text with type str, and the second column is the label with type int.
train_data = [['Example sentence belonging to class 1', 1], ['Example sentence belonging to class 0', 0]]
train_df = pd.DataFrame(train_data)

eval_data = [['Example eval sentence belonging to class 1', 1], ['Example eval sentence belonging to class 0', 0]]
eval_df = pd.DataFrame(eval_data)

# Create a ClassificationModel
model = ClassificationModel('roberta', 'roberta-base') # You can set class weights by using the optional weight argument

# Train the model
model.train_model(train_df)

# Evaluate the model
result, model_outputs, wrong_predictions = model.eval_model(eval_df)
```

If you wish to add any custom metrics, simply pass them as additional keyword arguments. The keyword is the name to be given to the metric, and the value is the function that will calculate the metric. Make sure that the function expects two parameters with the first one being the true label, and the second being the predictions. (This is the default for sklearn metrics)

```
import sklearn


result, model_outputs, wrong_predictions = model.eval_model(eval_df, acc=sklearn.metrics.accuracy_score)
```


To make predictions on arbitary data, the `predict(to_predict)` function can be used. For a list of text, it returns the model predictions and the raw model outputs.

```
predictions, raw_outputs = model.predict(['Some arbitary sentence'])
```

#### Minimal Start for Multiclass Classification

For multiclass classification, simply pass in the number of classes to the `num_labels` optional parameter of `ClassificationModel`.

```
from simpletransformers.classification import ClassificationModel
import pandas as pd


# Train and Evaluation data needs to be in a Pandas Dataframe containing at least two columns. If the Dataframe has a header, it should contain a 'text' and a 'labels' column. If no header is present, the Dataframe should contain at least two columns, with the first column is the text with type str, and the second column in the label with type int.
train_data = [['Example sentence belonging to class 1', 1], ['Example sentence belonging to class 0', 0], ['Example eval senntence belonging to class 2', 2]]
train_df = pd.DataFrame(train_data)

eval_data = [['Example eval sentence belonging to class 1', 1], ['Example eval sentence belonging to class 0', 0], ['Example eval senntence belonging to class 2', 2]]
eval_df = pd.DataFrame(eval_data)

# Create a ClassificationModel
model = ClassificationModel('bert', 'bert-base-cased', num_labels=3, args={'reprocess_input_data': True, 'overwrite_output_dir': True}) 
# You can set class weights by using the optional weight argument

# Train the model
model.train_model(train_df)

# Evaluate the model
result, model_outputs, wrong_predictions = model.eval_model(eval_df)

predictions, raw_outputs = model.predict(["Some arbitary sentence"])
```

#### Minimal Start for Multilabel Classification

For Multi-Label Classification, the labels should be multi-hot encoded. The number of classes can be specified (default is 2) by passing it to the `num_labels` optional parameter of `MultiLabelClassificationModel`.

_Warning: Pandas can cause issues when saving and loading lists stored in a column. Check whether your list has been converted to a String!_

The default evaluation metric used is Label Ranking Average Precision ([LRAP](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.label_ranking_average_precision_score.html)) Score.

```
from simpletransformers.classification import MultiLabelClassificationModel
import pandas as pd


# Train and Evaluation data needs to be in a Pandas Dataframe containing at least two columns, a 'text' and a 'labels' column. The `labels` column should contain multi-hot encoded lists.
train_data = [['Example sentence 1 for multilabel classification.', [1, 1, 1, 1, 0, 1]]] + [['This is another example sentence. ', [0, 1, 1, 0, 0, 0]]]
train_df = pd.DataFrame(train_data, columns=['text', 'labels'])
train_df = pd.DataFrame(train_data)

eval_data = [['Example eval sentence for multilabel classification.', [1, 1, 1, 1, 0, 1]], ['Example eval senntence belonging to class 2', [0, 1, 1, 0, 0, 0]]]
eval_df = pd.DataFrame(eval_data)

# Create a MultiLabelClassificationModel
model = MultiLabelClassificationModel('roberta', 'roberta-base', num_labels=6, args={'reprocess_input_data': True, 'overwrite_output_dir': True, 'num_train_epochs': 5})
# You can set class weights by using the optional weight argument
print(train_df.head())

# Train the model
model.train_model(train_df)

# Evaluate the model
result, model_outputs, wrong_predictions = model.eval_model(eval_df)
print(result)
print(model_outputs)

predictions, raw_outputs = model.predict(['This thing is entirely different from the other thing. '])
print(predictions)
print(raw_outputs)
```

##### Special Attributes

* The args dict of `MultiLabelClassificationModel` has an additional `threshold` parameter with default value 0.5. The threshold is the value at which a given label flips from 0 to 1 when predicting. The `threshold` may be a single value or a list of value with the same length as the number of labels. This enables the use of seperate threshold values for each label.
* `MultiLabelClassificationModel` takes in an additional optional argument `pos_weight`. This should be a list with the same length as the number of labels. This enables using different weights for each label when calculating loss during training and evaluation.

#### Real Dataset Examples

* [Yelp Reviews Dataset - Binary Classification](https://towardsdatascience.com/simple-transformers-introducing-the-easiest-bert-roberta-xlnet-and-xlm-library-58bf8c59b2a3?source=friends_link&sk=40726ceeadf99e1120abc9521a10a55c)
* [AG News Dataset - Multiclass Classification](https://medium.com/swlh/simple-transformers-multi-class-text-classification-with-bert-roberta-xlnet-xlm-and-8b585000ce3a)
* [Toxic Comments Dataset - Multilabel Classification](https://medium.com/@chaturangarajapakshe/multi-label-classification-using-bert-roberta-xlnet-xlm-and-distilbert-with-simple-transformers-b3e0cda12ce5?sk=354e688fe238bfb43e9a575216816219)


#### ClassificationModel

`class simpletransformers.classification.ClassificationModel (model_type, model_name, args=None, use_cuda=True)`  
This class  is used for Text Classification tasks.

`Class attributes`
* `tokenizer`: The tokenizer to be used.
* `model`: The model to be used.
* `model_name`: model_name: Default Transformer model name or path to Transformer model file (pytorch_nodel.bin).
* `device`: The device on which the model will be trained and evaluated.
* `results`: A python dict of past evaluation results for the TransformerModel object.
* `args`: A python dict of arguments used for training and evaluation.

`Parameters`
* `model_type`: (required) str - The type of model to use. Currently, BERT, XLNet, XLM, and RoBERTa models are available.
* `model_name`: (required) str - The exact model to use. Could be a pretrained model name or path to a directory containing a model. See [Current Pretrained Models](#current-pretrained-models) for all available models.
* `num_labels` (optional): The number of labels or classes in the dataset.
* `weight` (optional): A list of length num_labels containing the weights to assign to each label for loss calculation.
* `args`: (optional) python dict - A dictionary containing any settings that should be overwritten from the default values.
* `use_cuda`: (optional) bool - Default = True. Flag used to indicate whether CUDA should be used.

`class methods`  
**`train_model(self, train_df, output_dir=None, show_running_loss=True, args=None, eval_df=None)`**

Trains the model using 'train_df'

Args:  
* `train_df`: Pandas Dataframe containing at least two columns. If the Dataframe has a header, it should contain a 'text' and a 'labels' column. If no header is present, the Dataframe should contain at least two columns, with the first column containing the text, and the second column containing the label. The model will be trained on this Dataframe.

* `output_dir` (optional): The directory where model files will be saved. If not given, self.args['output_dir'] will be used.

>`args` (optional): Optional changes to the args dict of the model. Any changes made will persist for the model.

* show_running_loss (optional): Set to False to disable printing running training loss to the terminal.

* `eval_df` (optional): A DataFrame against which evaluation will be performed when `evaluate_during_training` is enabled. Is required if `evaluate_during_training` is enabled.

Returns:  
* None

**`eval_model(self, eval_df, output_dir=None, verbose=False)`**

Evaluates the model on eval_df. Saves results to output_dir.

Args:  
* eval_df: Pandas Dataframe containing at least two columns. If the Dataframe has a header, it should contain a 'text' and a 'labels' column. If no header is present, the Dataframe should contain at least two columns, with the first column containing the text, and the second column containing the label. The model will be evaluated on this Dataframe.

* output_dir: The directory where model files will be saved. If not given, self.args['output_dir'] will be used.  

* verbose: If verbose, results will be printed to the console on completion of evaluation.  

Returns:  
* result: Dictionary containing evaluation results. (Matthews correlation coefficient, tp, tn, fp, fn)  

>model_outputs: List of model outputs for each row in eval_df  

* wrong_preds: List of InputExample objects corresponding to each incorrect prediction by the model  

**`predict(self, to_predict)`**

Performs predictions on a list of text.

Args:
* to_predict: A python list of text (str) to be sent to the model for prediction.

Returns:
* preds: A python list of the predictions (0 or 1) for each text.  
* model_outputs: A python list of the raw model outputs for each text.


**`train(self, train_dataset, output_dir)`**

Trains the model on train_dataset.
*Utility function to be used by the train_model() method. Not intended to be used directly.*

**`evaluate(self, eval_df, output_dir, prefix="")`**

Evaluates the model on eval_df.
*Utility function to be used by the eval_model() method. Not intended to be used directly*

**`load_and_cache_examples(self, examples, evaluate=False)`**

Converts a list of InputExample objects to a TensorDataset containing InputFeatures. Caches the InputFeatures.
*Utility function for train() and eval() methods. Not intended to be used directly*

**`compute_metrics(self, preds, labels, eval_examples, **kwargs):`**

Computes the evaluation metrics for the model predictions.

Args:
* preds: Model predictions  

* labels: Ground truth labels  

* eval_examples: List of examples on which evaluation was performed  

Returns:
* result: Dictionary containing evaluation results. (Matthews correlation coefficient, tp, tn, fp, fn)  

* wrong: List of InputExample objects corresponding to each incorrect prediction by the model  

---

## Named Entity Recognition

This section describes how to use Simple Transformers for Named Entity Recognition. (If you are updating from a Simple Transformers before 0.5.0, note that `seqeval` needs to be installed to perform NER.)

*This model can also be used for any other NLP task involving token level classification. Make sure you pass in your list of labels to the model if they are different from the defaults.*

Supported model types:
* BERT
* RoBERTa
* DistilBERT
* CamemBERT

```
model = NERModel('bert', 'bert-base-cased', labels=["LABEL_1", "LABEL_2", "LABEL_3"])
```

#### Minimal Start

```
from simpletransformers.ner import NERModel
import pandas as pd


# Creating train_df  and eval_df for demonstration
train_data = [
    [0, 'Simple', 'B-MISC'], [0, 'Transformers', 'I-MISC'], [0, 'started', 'O'], [1, 'with', 'O'], [0, 'text', 'O'], [0, 'classification', 'B-MISC'],
    [1, 'Simple', 'B-MISC'], [1, 'Transformers', 'I-MISC'], [1, 'can', 'O'], [1, 'now', 'O'], [1, 'perform', 'O'], [1, 'NER', 'B-MISC']
]
train_df = pd.DataFrame(train_data, columns=['sentence_id', 'words', 'labels'])

eval_data = [
    [0, 'Simple', 'B-MISC'], [0, 'Transformers', 'I-MISC'], [0, 'was', 'O'], [1, 'built', 'O'], [1, 'for', 'O'], [0, 'text', 'O'], [0, 'classification', 'B-MISC'],
    [1, 'Simple', 'B-MISC'], [1, 'Transformers', 'I-MISC'], [1, 'then', 'O'], [1, 'expanded', 'O'], [1, 'to', 'O'], [1, 'perform', 'O'], [1, 'NER', 'B-MISC']
]
eval_df = pd.DataFrame(eval_data, columns=['sentence_id', 'words', 'labels'])

# Create a NERModel
model = NERModel('bert', 'bert-base-cased', args={'overwrite_output_dir': True, 'reprocess_input_data': True})

# Train the model
model.train_model(train_df)

# Evaluate the model
result, model_outputs, predictions = model.eval_model(eval_df)

# Predictions on arbitary text strings
predictions, raw_outputs = model.predict(["Some arbitary sentence"])

print(predictions)
```

#### Real Dataset Examples

* [CoNLL Dataset Example](https://medium.com/@chaturangarajapakshe/simple-transformers-named-entity-recognition-with-transformer-models-c04b9242a2a0?sk=e8b98c994173cd5219f01e075727b096)

#### NERModel

`class simpletransformers.ner.ner_model.NERModel (model_type, model_name, labels=None, args=None, use_cuda=True)`  
This class  is used for Named Entity Recognition.

`Class attributes`
* `tokenizer`: The tokenizer to be used.
* `model`: The model to be used.
            model_name: Default Transformer model name or path to Transformer model file (pytorch_nodel.bin).
* `device`: The device on which the model will be trained and evaluated.
* `results`: A python dict of past evaluation results for the TransformerModel object.
* `args`: A python dict of arguments used for training and evaluation.

`Parameters`
* `model_type`: (required) str - The type of model to use. Currently, BERT, XLNet, XLM, and RoBERTa models are available.
* `model_name`: (required) str - The exact model to use. Could be a pretrained model name or path to a directory containing a model. See [Current Pretrained Models](#current-pretrained-models) for all available models.
* `labels` (optional): A list of all Named Entity labels.  If not given, ["O", "B-MISC", "I-MISC",  "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"] will be used.
* `args`: (optional) python dict - A dictionary containing any settings that should be overwritten from the default values.
* `use_cuda`: (optional) bool - Default = True. Flag used to indicate whether CUDA should be used.

`class methods`  
**`train_model(self, train_data, output_dir=None, args=None, eval_df=None)`**

Trains the model using 'train_data'

Args:  
* train_data: train_data should be the path to a .txt file containing the training data OR a pandas DataFrame with 3 columns.
If a text file is used the data should be in the CoNLL format. i.e. One word per line, with sentences seperated by an empty line. 
The first word of the line should be a word, and the last should be a Name Entity Tag.
If a DataFrame is given, each sentence should be split into words, with each word assigned a tag, and with all words from the same sentence given the same sentence_id.

* output_dir: The directory where model files will be saved. If not given, self.args['output_dir'] will be used.

* show_running_loss (optional): Set to False to prevent running loss from being printed to console. Defaults to True.

* args (optional): Optional changes to the args dict of the model. Any changes made will persist for the model.

* eval_df (optional): A DataFrame against which evaluation will be performed when `evaluate_during_training` is enabled. Is required if `evaluate_during_training` is enabled.


Returns:  
* None

**`eval_model(self, eval_data, output_dir=None, verbose=True)`**

Evaluates the model on eval_data. Saves results to output_dir.

Args:  
* eval_data: Pandas Dataframe containing at least two columns. If the Dataframe has a header, it should contain a 'text' and a 'labels' column. If no header is present, the Dataframe should contain at least two columns, with the first column containing the text, and the second column containing the label. The model will be evaluated on this Dataframe.

* output_dir: The directory where model files will be saved. If not given, self.args['output_dir'] will be used.  

* verbose: If verbose, results will be printed to the console on completion of evaluation.  

Returns:  
* result: Dictionary containing evaluation results. (eval_loss, precision, recall, f1_score)

* model_outputs: List of raw model outputs

* preds_list: List of predicted tags

**`predict(self, to_predict)`**

Performs predictions on a list of text.

Args:
* to_predict: A python list of text (str) to be sent to the model for prediction.

Returns:
* preds: A Python list of lists with dicts containg each word mapped to its NER tag.
* model_outputs: A python list of the raw model outputs for each text.


**`train(self, train_dataset, output_dir)`**

Trains the model on train_dataset.
*Utility function to be used by the train_model() method. Not intended to be used directly.*

**`evaluate(self, eval_dataset, output_dir, prefix="")`**

Evaluates the model on eval_dataset.
*Utility function to be used by the eval_model() method. Not intended to be used directly*

**`load_and_cache_examples(self, data, evaluate=False, no_cache=False, to_predict=None)`**

Converts a list of InputExample objects to a TensorDataset containing InputFeatures. Caches the InputFeatures.
*Utility function for train() and eval() methods. Not intended to be used directly*

___

## Question Answering

Supported model types:

* BERT
* XLNet
* XLM
* DistilBERT
* ALBERT

### Data format

For question answering tasks, the input data can be in JSON files or in a Python list of dicts in the correct format.

The file should contain a single list of dictionaries. A dictionary represents a single context and its associated questions.

Each such dictionary contains two attributes, the `"context"` and `"qas"`.
* `context`: The paragraph or text from which the question is asked.
* `qas`: A list of questions and answers.

Questions and answers are represented as dictionaries. Each dictionary in `qas` has the following format.
* `id`: (string) A unique ID for the question. Should be unique across the entire dataset.
* `question`: (string) A question. 
* `is_impossible`: (bool) Indicates whether the question can be answered correctly from the context. 
* `answers`: (list) The list of correct answers to the question.

A single answer is represented by a dictionary with the following attributes.
* `answer`: (string) The answer to the question. Must be a substring of the context.
* `answer_start`: (int) Starting index of the answer in the context.

### Minimal Example

```
from simpletransformers.question_answering import QuestionAnsweringModel
import json
import os


# Create dummy data to use for training.
train_data = [
    {
        'context': "This is the first context",
        'qas': [
            {
                'id': "00001",
                'is_impossible': False,
                'question': "Which context is this?",
                'answers': [
                    {
                        'text': "the first",
                        'answer_start': 8
                    }
                ]
            }
        ]
    },
    {
        'context': "Other legislation followed, including the Migratory Bird Conservation Act of 1929, a 1937 treaty prohibiting the hunting of right and gray whales,
            and the Bald Eagle Protection Act of 1940. These later laws had a low cost to society—the species were relatively rare—and little opposition was raised",
        'qas': [
            {
                'id': "00002",
                'is_impossible': False,
                'question': "What was the cost to society?",
                'answers': [
                    {
                        'text': "low cost",
                        'answer_start': 225
                    }
                ]
            },
            {
                'id': "00003",
                'is_impossible': False,
                'question': "What was the name of the 1937 treaty?",
                'answers': [
                    {
                        'text': "Bald Eagle Protection Act",
                        'answer_start': 167
                    }
                ]
            }
        ]
    }
]

# Save as a JSON file
os.makedirs('data', exist_ok=True)
with open('data/train.json', 'w') as f:
    json.dump(train_data, f)


# Create the QuestionAnsweringModel
model = QuestionAnsweringModel('distilbert', 'distilbert-base-uncased-distilled-squad', args={'reprocess_input_data': True, 'overwrite_output_dir': True})

# Train the model with JSON file
model.train_model('data/train.json')

# The list can also be used directly
# model.train_model(train_data)

# Evaluate the model. (Being lazy and evaluating on the train data itself)
result, text = model.eval_model('data/train.json')

print(result)
print(text)

print('-------------------')

# Making predictions using the model.
to_predict = [{'context': 'This is the context used for demonstrating predictions.', 'qas': [{'question': 'What is this context?', 'id': '0'}]}]

print(model.predict(to_predict))
```

#### Real Dataset Examples

* [SQuAD 2.0 - Question Answering](https://medium.com/@chaturangarajapakshe/question-answering-with-bert-xlnet-xlm-and-distilbert-using-simple-transformers-4d8785ee762a?sk=e8e6f9a39f20b5aaf08bbcf8b0a0e1c2)

### QuestionAnsweringModel

`class simpletransformers.question_answering.QuestionAnsweringModel (model_type, model_name, args=None, use_cuda=True)`  
This class  is used for Question Answering tasks.

`Class attributes`
* `tokenizer`: The tokenizer to be used.
* `model`: The model to be used.
            model_name: Default Transformer model name or path to Transformer model file (pytorch_nodel.bin).
* `device`: The device on which the model will be trained and evaluated.
* `results`: A python dict of past evaluation results for the TransformerModel object.
* `args`: A python dict of arguments used for training and evaluation.

`Parameters`
* `model_type`: (required) str - The type of model to use.
* `model_name`: (required) str - The exact model to use. Could be a pretrained model name or path to a directory containing a model. See [Current Pretrained Models](#current-pretrained-models) for all available models.
* `args`: (optional) python dict - A dictionary containing any settings that should be overwritten from the default values.
* `use_cuda`: (optional) bool - Default = True. Flag used to indicate whether CUDA should be used.

`class methods`  
**`train_model(self, train_df, output_dir=None, args=None, eval_df=None)`**

Trains the model using 'train_file'

Args:  
* train_df: ath to JSON file containing training data. The model will be trained on this file.
            output_dir: The directory where model files will be saved. If not given, self.args['output_dir'] will be used.

* output_dir (optional): The directory where model files will be saved. If not given, self.args['output_dir'] will be used.

* show_running_loss (Optional): Set to False to prevent training loss being printed.

* args (optional): Optional changes to the args dict of the model. Any changes made will persist for the model.

* eval_file (optional): Path to JSON file containing evaluation data against which evaluation will be performed when evaluate_during_training is enabled. Is required if evaluate_during_training is enabled.

Returns:  
* None

**`eval_model(self, eval_df, output_dir=None, verbose=False)`**

Evaluates the model on eval_file. Saves results to output_dir.

Args:  
* eval_file: Path to JSON file containing evaluation data. The model will be evaluated on this file.

* output_dir: The directory where model files will be saved. If not given, self.args['output_dir'] will be used.  

* verbose: If verbose, results will be printed to the console on completion of evaluation.  

Returns:  
* result: Dictionary containing evaluation results. (correct, similar, incorrect)

* text: A dictionary containing the 3 dictionaries correct_text, similar_text (the predicted answer is a substring of the correct answer or vise versa), incorrect_text. 


**`predict(self, to_predict)`**

Performs predictions on a list of text.

Args:
* to_predict: A python list of python dicts containing contexts and questions to be sent to the model for prediction.
```
E.g: predict([
    {
        'context': "Some context as a demo",
        'qas': [
            {'id': '0', 'question': 'What is the context here?'},
            {'id': '1', 'question': 'What is this for?'}
        ]
    }
])
```
* n_best_size (Optional): Number of predictions to return. args['n_best_size'] will be used if not specified.

Returns:
* preds: A python list containg the predicted answer, and id for each question in to_predict.


**`train(self, train_dataset, output_dir, show_running_loss=True, eval_file=None)`**

Trains the model on train_dataset.
*Utility function to be used by the train_model() method. Not intended to be used directly.*

**`evaluate(self, eval_df, output_dir, , verbose=False)`**

Evaluates the model on eval_df.
*Utility function to be used by the eval_model() method. Not intended to be used directly*

**`load_and_cache_examples(self, examples, evaluate=False, no_cache=False, output_examples=False)`**

Converts a list of InputExample objects to a TensorDataset containing InputFeatures. Caches the InputFeatures.
*Utility function for train() and eval() methods. Not intended to be used directly*

### Additional attributes for Question Answering tasks

QuestionAnsweringModel has a few additional attributes in its `args` dictionary, given below with their default values.

```
  'doc_stride': 384,
  'max_query_length': 64,
  'n_best_size': 20,
  'max_answer_length': 100,
  'null_score_diff_threshold': 0.0
```

#### *doc_stride: int*
When splitting up a long document into chunks, how much stride to take between chunks.

#### *max_query_length: int*
Maximum token length for questions. Any questions longer than this will be truncated to this length.

#### *n_best_size: int*
The number of predictions given per question.

#### *max_answer_length: int*
The maximum token length of an answer that can be generated.

#### *null_score_diff_threshold: float*
If null_score - best_non_null is greater than the threshold predict null.

---

## Experimental Features

To use experimental features, import from `simpletransformers.experimental.X`

```
from simpletransformers.experimental.classification import ClassificationModel
```

### Sliding Window For Long Sequences

Normally, sequences longer than `max_seq_length` are unceremoniously truncated.

This experimental feature moves a sliding window over each sequence and generates sub-sequences with length `max_seq_length`. The model output for each sub-sequence is averaged into a single output before being sent to the linear classifier.

Currently avaiable on binary and multiclass classification models of the following types.

* BERT
* RoBERTa
* AlBERT
* XLNet
* CamemBERT

Set `sliding_window=True` for the ClassificationModel to enable this feature.

```
from simpletransformers.experimental.classification import ClassificationModel
import pandas as pd
import sklearn

# Train and Evaluation data needs to be in a Pandas Dataframe of two columns. The first column is the text with type str, and the second column in the label with type int.
train_data = [['Example sentence belonging to class 1' * 50, 1], ['Example sentence belonging to class 0', 0], ['Example  2 sentence belonging to class 0', 0]] + [['Example sentence belonging to class 0', 0] for i in range(12)]
train_df = pd.DataFrame(train_data, columns=['text', 'labels'])


eval_data = [['Example eval sentence belonging to class 1', 1], ['Example eval sentence belonging to class 0', 0]]
eval_df = pd.DataFrame(eval_data)

train_args={
    'reprocess_input_data': True,
    'overwrite_output_dir': True,
    'evaluate_during_training': True,
    'logging_steps': 5,
    'stride': 0.8,
    'max_seq_length': 128
}

# Create a TransformerModel
model = ClassificationModel('camembert', 'camembert-base', sliding_window=True, args=train_args, use_cuda=False)
print(train_df.head())

# Train the model
model.train_model(train_df, eval_df=eval_df)

# Evaluate the model
result, model_outputs, wrong_predictions = model.eval_model(eval_df, acc=sklearn.metrics.accuracy_score)

predictions, raw_outputs = model.predict(["I'd like to puts some CD-ROMS on my iPad, is that possible?' — Yes, but wouldn't that block the screen?" * 25])
print(predictions)
print(raw_outputs)
```

---
## Loading Saved Models

To load a saved model, provide the path to the directory containing the saved model as the `model_name`.

```
model = ClassificationModel('roberta', 'outputs/')
```

```
model = NERModel('bert', 'outputs/')
```

---


## Default Settings

The default args used are given below. Any of these can be overridden by passing a dict containing the corresponding key: value pairs to the the init method of a Model class.

```
self.args = {
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
  'do_lower_case': False,

  'logging_steps': 50,
  'evaluate_during_training': False,
  'evaluate_during_training_steps': 2000,
  'save_steps': 2000,

  'overwrite_output_dir': False,
  'reprocess_input_data': False,
  
  'process_count': cpu_count() - 2 if cpu_count() > 2 else 1
  'n_gpu': 1,
  'silent': False,
  'use_multiprocessing': True,
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

#### *do_lower_case: bool*
Set to True when using uncased models.

#### *evaluate_during_training*
Set to True to perform evaluation while training models. Make sure `eval_df` is passed to the training method if enabled.

#### *evaluate_during_training_steps*
Perform evaluation at every specified number of steps. A checkpoint model and the evaluation results will be saved.

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

#### *n_gpu: int*
Number of GPUs to use.

#### *silent: bool*
Disables progress bars.

#### *use_multiprocessing: bool*
If True, multiprocessing will be used when converting data into features. Disabling can reduce memory usage, but may substantially slow down processing.

---

## Current Pretrained Models

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
| BERT      | bert | bert-base-german-dbmdz-cased | 12-layer, 768-hidden, 12-heads, 110M parameters. Trained on cased German text by DBMDZ |
| BERT      | bert | bert-base-german-dbmdz-uncased | 12-layer, 768-hidden, 12-heads, 110M parameters. Trained on uncased German text by DBMDZ |
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
| DistilBERT   | distilbert | distilbert-base-uncased| 6-layer, 768-hidden, 12-heads, 66M parameters <br>The DistilBERT model distilled from the BERT model bert-base-uncased checkpoint |
| DistilBERT   | distilbert | distilbert-base-uncased-distilled-squad | 6-layer, 768-hidden, 12-heads, 66M parameters <br>The DistilBERT model distilled from the BERT model bert-base-uncased checkpoint, with an additional linear layer.|
| DistilBERT German   | distilbert | distilbert-base-german-cased | 6-layer, 768-hidden, 12-heads, 66M parameters <br>The DistilBERT model distilled from the BERT model bert-base-cased checkpoint on German data.|
| DistilBERT Multilingual   | distilbert | distilbert-base-multilingual-cased | 6-layer, 768-hidden, 12-heads, 66M parameters <br>The DistilBERT model distilled from the BERT model bert-base-cased checkpoint on multilingual data.|
| ALBERT      | albert | albert-base-v1 | 12 repeating layers, 128 embedding, 768-hidden, 12-heads, 11M parameters; ALBERT base model. |
| ALBERT      | albert | albert-large-v1 | 24 repeating layers, 128 embedding, 1024-hidden, 16-heads, 17M parameters; ALBERT large model |
| ALBERT      | albert | albert-xlarge-v1 | 24 repeating layers, 128 embedding, 2048-hidden, 16-heads, 58M parameters; ALBERT xlarge model |
| ALBERT      | albert | albert-xxlarge-v1 | 12 repeating layers, 128 embedding, 4096-hidden, 64-heads, 223M parameters; ALBERT xxlarge model |
| ALBERT      | albert | albert-base-v2 | 12 repeating layers, 128 embedding, 768-hidden, 12-heads, 11M parameters; ALBERT base model with no dropout, additional training data and longer training|
| ALBERT      | albert | albert-large-v2 | 24 repeating layers, 128 embedding, 1024-hidden, 16-heads, 17M parameters; ALBERT large model with no dropout, additional training data and longer training|
| ALBERT      | albert | albert-xlarge-v2 | 24 repeating layers, 128 embedding, 2048-hidden, 16-heads, 58M parameters; ALBERT xlarge model with no dropout, additional training data and longer training |
| ALBERT      | albert | albert-xxlarge-v2 | 12 repeating layer, 128 embedding, 4096-hidden, 64-heads, 223M parameters; ALBERT xxlarge model with no dropout, additional training data and longer training |
| CamemBERT     | camembert | camembert-base | 12-layer, 768-hidden, 12-heads, 110M parameters CamemBERT using the RoBERTa architecture |


---

## Acknowledgements

None of this would have been possible without the hard work by the HuggingFace team in developing the [Pytorch-Transformers](https://github.com/huggingface/pytorch-transformers) library.

_<div>Icon for the Social Media Preview made by <a href="https://www.flaticon.com/authors/freepik" title="Freepik">Freepik</a> from <a href="https://www.flaticon.com/" title="Flaticon">www.flaticon.com</a></div>_
