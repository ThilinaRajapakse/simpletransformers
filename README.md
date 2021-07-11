<!-- ALL-CONTRIBUTORS-BADGE:START - Do not remove or modify this section -->


<!-- ALL-CONTRIBUTORS-BADGE:END -->

# Simple Transformers

This library is based on the [Transformers](https://github.com/huggingface/transformers) library by HuggingFace. `Simple Transformers` lets you quickly train and evaluate Transformer models. Only 3 lines of code are needed to **initialize**, **train**, and **evaluate** a model.

**Supported Tasks:**

- Sequence Classification
- Token Classification (NER)
- Question Answering
- Language Model Fine-Tuning
- Language Model Training
- Language Generation
- T5 Model
- Seq2Seq Tasks
- Multi-Modal Classification
- Conversational AI.
- Text Representation Generation.

# Table of contents

<!--ts-->

- [Simple Transformers](#simple-transformers)
- [Table of contents](#table-of-contents)
  - [Setup](#setup)
    - [With Conda](#with-conda)
      - [Optional](#optional)
  - [Usage](#usage)
    - [A quick example](#a-quick-example)
    - [Experiment Tracking with Weights and Biases](#experiment-tracking-with-weights-and-biases)
  - [Current Pretrained Models](#current-pretrained-models)
  - [Contributors ✨](#contributors-)
  - [How to Contribute](#how-to-contribute)
    - [How to Update Docs](#how-to-update-docs)
  - [Acknowledgements](#acknowledgements)

<!--te-->

## Setup

### With Conda

1. Install `Anaconda` or `Miniconda` Package Manager from [here](https://www.anaconda.com/distribution/)
2. Create a new virtual environment and install packages.

```bash
$ conda create -n st python pandas tqdm
$ conda activate st
```

With using Cuda:

```bash
$ conda install pytorch>=1.6 cudatoolkit=11.0 -c pytorch
```

Without using Cuda

```bash
$ conda install pytorch cpuonly -c pytorch
```

3. Install `simpletransformers`.

```bash
$ pip install simpletransformers
```

#### Optional

1. Install `Weights` and `Biases` (wandb) for tracking and visualizing training in a web browser.

```bash
$ pip install wandb
```

## Usage

**All documentation is now live at [simpletransformers.ai](https://simpletransformers.ai/)**

`Simple Transformer` models are built with a particular Natural Language Processing (NLP) task in mind. Each such model comes equipped with features and functionality designed to best fit the task that they are intended to perform. The high-level process of using Simple Transformers models follows the same pattern.

1. Initialize a task-specific model
2. Train the model with `train_model()`
3. Evaluate the model with `eval_model()`
4. Make predictions on (unlabelled) data with `predict()`

However, there are necessary differences between the different models to ensure that they are well suited for their intended task. The key differences will typically be the differences in input/output data formats and any task specific features/configuration options. These can all be found in the documentation section for each task.

The currently implemented task-specific `Simple Transformer` models, along with their task, are given below.

| Task                                                      | Model                             |
| --------------------------------------------------------- | --------------------------------- |
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

- **Please refer to the relevant section in the [docs](https://simpletransformers.ai/) for more information on how to use these models.**
- Example scripts can be found in the [examples](https://github.com/ThilinaRajapakse/simpletransformers/tree/master/examples) directory.
- See the [Changelog](https://github.com/ThilinaRajapakse/simpletransformers/blob/master/CHANGELOG.md) for up-to-date changes to the project.

### A quick example

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

# Optional model configuration
model_args = ClassificationArgs(num_train_epochs=1)

# Create a ClassificationModel
model = ClassificationModel(
    "roberta", "roberta-base", args=model_args
)

# Train the model
model.train_model(train_df)

# Evaluate the model
result, model_outputs, wrong_predictions = model.eval_model(eval_df)

# Make predictions with the model
predictions, raw_outputs = model.predict(["Sam was a Wizard"])

```

### Experiment Tracking with Weights and Biases

- W&B Notebook -

---

## Current Pretrained Models

For a list of pretrained models, see [Hugging Face docs](https://huggingface.co/pytorch-transformers/pretrained_models.html).

The `model_types` available for each task can be found under their respective section. Any pretrained model of that type
found in the Hugging Face docs should work. To use any of them set the correct `model_type` and `model_name` in the `args`
dictionary.

---

## Contributors ✨

Thanks goes to these wonderful people ([emoji key](https://allcontributors.org/docs/en/emoji-key)):

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->

<!-- prettier-ignore-start -->

<!-- markdownlint-disable -->

<table>
  <tr>
    <td align="center"><a href="https://github.com/hawktang"><img src="https://avatars0.githubusercontent.com/u/2004071?v=4?s=100" width="100px;" alt=""/><br /><sub><b>hawktang</b></sub></a><br /><a href="https://github.com/ThilinaRajapakse/simpletransformers/commits?author=hawktang" title="Code">💻</a></td>
    <td align="center"><a href="http://datawizzards.io"><img src="https://avatars0.githubusercontent.com/u/22409996?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Mabu Manaileng</b></sub></a><br /><a href="https://github.com/ThilinaRajapakse/simpletransformers/commits?author=mabu-dev" title="Code">💻</a></td>
    <td align="center"><a href="https://www.facebook.com/aliosm97"><img src="https://avatars3.githubusercontent.com/u/7662492?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Ali Hamdi Ali Fadel</b></sub></a><br /><a href="https://github.com/ThilinaRajapakse/simpletransformers/commits?author=AliOsm" title="Code">💻</a></td>
    <td align="center"><a href="http://tovly.co"><img src="https://avatars0.githubusercontent.com/u/12242351?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Tovly Deutsch</b></sub></a><br /><a href="https://github.com/ThilinaRajapakse/simpletransformers/commits?author=TovlyDeutsch" title="Code">💻</a></td>
    <td align="center"><a href="https://github.com/hlo-world"><img src="https://avatars0.githubusercontent.com/u/9633055?v=4?s=100" width="100px;" alt=""/><br /><sub><b>hlo-world</b></sub></a><br /><a href="https://github.com/ThilinaRajapakse/simpletransformers/commits?author=hlo-world" title="Code">💻</a></td>
    <td align="center"><a href="https://github.com/huntertl"><img src="https://avatars1.githubusercontent.com/u/15113885?v=4?s=100" width="100px;" alt=""/><br /><sub><b>huntertl</b></sub></a><br /><a href="https://github.com/ThilinaRajapakse/simpletransformers/commits?author=huntertl" title="Code">💻</a></td>
    <td align="center"><a href="https://whattheshot.com"><img src="https://avatars2.githubusercontent.com/u/623763?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Yann Defretin</b></sub></a><br /><a href="https://github.com/ThilinaRajapakse/simpletransformers/commits?author=kinoute" title="Code">💻</a> <a href="https://github.com/ThilinaRajapakse/simpletransformers/commits?author=kinoute" title="Documentation">📖</a> <a href="#question-kinoute" title="Answering Questions">💬</a> <a href="#ideas-kinoute" title="Ideas, Planning, & Feedback">🤔</a></td>
  </tr>
  <tr>
    <td align="center"><a href="https://github.com/mananeau"><img src="https://avatars0.githubusercontent.com/u/29440170?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Manuel </b></sub></a><br /><a href="https://github.com/ThilinaRajapakse/simpletransformers/commits?author=mananeau" title="Documentation">📖</a> <a href="https://github.com/ThilinaRajapakse/simpletransformers/commits?author=mananeau" title="Code">💻</a></td>
    <td align="center"><a href="http://jacobsgill.es"><img src="https://avatars2.githubusercontent.com/u/9109832?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Gilles Jacobs</b></sub></a><br /><a href="https://github.com/ThilinaRajapakse/simpletransformers/commits?author=GillesJ" title="Documentation">📖</a></td>
    <td align="center"><a href="https://github.com/shasha79"><img src="https://avatars2.githubusercontent.com/u/5512649?v=4?s=100" width="100px;" alt=""/><br /><sub><b>shasha79</b></sub></a><br /><a href="https://github.com/ThilinaRajapakse/simpletransformers/commits?author=shasha79" title="Code">💻</a></td>
    <td align="center"><a href="http://www-lium.univ-lemans.fr/~garcia"><img src="https://avatars2.githubusercontent.com/u/14233427?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Mercedes Garcia</b></sub></a><br /><a href="https://github.com/ThilinaRajapakse/simpletransformers/commits?author=merc85garcia" title="Code">💻</a></td>
    <td align="center"><a href="https://github.com/hammad26"><img src="https://avatars1.githubusercontent.com/u/12643784?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Hammad Hassan Tarar</b></sub></a><br /><a href="https://github.com/ThilinaRajapakse/simpletransformers/commits?author=hammad26" title="Code">💻</a> <a href="https://github.com/ThilinaRajapakse/simpletransformers/commits?author=hammad26" title="Documentation">📖</a></td>
    <td align="center"><a href="https://github.com/todd-cook"><img src="https://avatars3.githubusercontent.com/u/665389?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Todd Cook</b></sub></a><br /><a href="https://github.com/ThilinaRajapakse/simpletransformers/commits?author=todd-cook" title="Code">💻</a></td>
    <td align="center"><a href="http://knuthellan.com/"><img src="https://avatars2.githubusercontent.com/u/51441?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Knut O. Hellan</b></sub></a><br /><a href="https://github.com/ThilinaRajapakse/simpletransformers/commits?author=khellan" title="Code">💻</a> <a href="https://github.com/ThilinaRajapakse/simpletransformers/commits?author=khellan" title="Documentation">📖</a></td>
  </tr>
  <tr>
    <td align="center"><a href="https://github.com/nagenshukla"><img src="https://avatars0.githubusercontent.com/u/39196228?v=4?s=100" width="100px;" alt=""/><br /><sub><b>nagenshukla</b></sub></a><br /><a href="https://github.com/ThilinaRajapakse/simpletransformers/commits?author=nagenshukla" title="Code">💻</a></td>
    <td align="center"><a href="https://www.linkedin.com/in/flaviussn/"><img src="https://avatars0.githubusercontent.com/u/20523032?v=4?s=100" width="100px;" alt=""/><br /><sub><b>flaviussn</b></sub></a><br /><a href="https://github.com/ThilinaRajapakse/simpletransformers/commits?author=flaviussn" title="Code">💻</a> <a href="https://github.com/ThilinaRajapakse/simpletransformers/commits?author=flaviussn" title="Documentation">📖</a></td>
    <td align="center"><a href="http://marctorrellas.github.com"><img src="https://avatars1.githubusercontent.com/u/22045779?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Marc Torrellas</b></sub></a><br /><a href="#maintenance-marctorrellas" title="Maintenance">🚧</a></td>
    <td align="center"><a href="https://github.com/adrienrenaud"><img src="https://avatars3.githubusercontent.com/u/6208157?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Adrien Renaud</b></sub></a><br /><a href="https://github.com/ThilinaRajapakse/simpletransformers/commits?author=adrienrenaud" title="Code">💻</a></td>
    <td align="center"><a href="https://github.com/jacky18008"><img src="https://avatars0.githubusercontent.com/u/9031441?v=4?s=100" width="100px;" alt=""/><br /><sub><b>jacky18008</b></sub></a><br /><a href="https://github.com/ThilinaRajapakse/simpletransformers/commits?author=jacky18008" title="Code">💻</a></td>
    <td align="center"><a href="https://github.com/seo-95"><img src="https://avatars0.githubusercontent.com/u/38254541?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Matteo Senese</b></sub></a><br /><a href="https://github.com/ThilinaRajapakse/simpletransformers/commits?author=seo-95" title="Code">💻</a></td>
    <td align="center"><a href="https://github.com/sarthakTUM"><img src="https://avatars2.githubusercontent.com/u/23062869?v=4?s=100" width="100px;" alt=""/><br /><sub><b>sarthakTUM</b></sub></a><br /><a href="https://github.com/ThilinaRajapakse/simpletransformers/commits?author=sarthakTUM" title="Documentation">📖</a> <a href="https://github.com/ThilinaRajapakse/simpletransformers/commits?author=sarthakTUM" title="Code">💻</a></td>
  </tr>
  <tr>
    <td align="center"><a href="https://github.com/djstrong"><img src="https://avatars1.githubusercontent.com/u/1849959?v=4?s=100" width="100px;" alt=""/><br /><sub><b>djstrong</b></sub></a><br /><a href="https://github.com/ThilinaRajapakse/simpletransformers/commits?author=djstrong" title="Code">💻</a></td>
    <td align="center"><a href="http://kozistr.tech"><img src="https://avatars2.githubusercontent.com/u/15344796?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Hyeongchan Kim</b></sub></a><br /><a href="https://github.com/ThilinaRajapakse/simpletransformers/commits?author=kozistr" title="Documentation">📖</a></td>
    <td align="center"><a href="https://github.com/Pradhy729"><img src="https://avatars3.githubusercontent.com/u/49659913?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Pradhy729</b></sub></a><br /><a href="https://github.com/ThilinaRajapakse/simpletransformers/commits?author=Pradhy729" title="Code">💻</a> <a href="#maintenance-Pradhy729" title="Maintenance">🚧</a></td>
    <td align="center"><a href="https://iknoorjobs.github.io/"><img src="https://avatars2.githubusercontent.com/u/22852967?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Iknoor Singh</b></sub></a><br /><a href="https://github.com/ThilinaRajapakse/simpletransformers/commits?author=iknoorjobs" title="Documentation">📖</a></td>
    <td align="center"><a href="https://github.com/galtay"><img src="https://avatars2.githubusercontent.com/u/663051?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Gabriel Altay</b></sub></a><br /><a href="https://github.com/ThilinaRajapakse/simpletransformers/commits?author=galtay" title="Code">💻</a></td>
    <td align="center"><a href="https://a-ware.io"><img src="https://avatars1.githubusercontent.com/u/47894090?v=4?s=100" width="100px;" alt=""/><br /><sub><b>flozi00</b></sub></a><br /><a href="https://github.com/ThilinaRajapakse/simpletransformers/commits?author=flozi00" title="Documentation">📖</a> <a href="https://github.com/ThilinaRajapakse/simpletransformers/commits?author=flozi00" title="Code">💻</a> <a href="#maintenance-flozi00" title="Maintenance">🚧</a></td>
    <td align="center"><a href="https://github.com/alexysdussier"><img src="https://avatars3.githubusercontent.com/u/60175018?v=4?s=100" width="100px;" alt=""/><br /><sub><b>alexysdussier</b></sub></a><br /><a href="https://github.com/ThilinaRajapakse/simpletransformers/commits?author=alexysdussier" title="Code">💻</a></td>
  </tr>
  <tr>
    <td align="center"><a href="https://github.com/jqueguiner"><img src="https://avatars1.githubusercontent.com/u/690878?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Jean-Louis Queguiner</b></sub></a><br /><a href="https://github.com/ThilinaRajapakse/simpletransformers/commits?author=jqueguiner" title="Documentation">📖</a></td>
    <td align="center"><a href="https://github.com/aced125"><img src="https://avatars2.githubusercontent.com/u/44452903?v=4?s=100" width="100px;" alt=""/><br /><sub><b>aced125</b></sub></a><br /><a href="https://github.com/ThilinaRajapakse/simpletransformers/commits?author=aced125" title="Code">💻</a></td>
    <td align="center"><a href="https://github.com/Laksh1997"><img src="https://avatars0.githubusercontent.com/u/59830552?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Laksh1997</b></sub></a><br /><a href="https://github.com/ThilinaRajapakse/simpletransformers/commits?author=Laksh1997" title="Code">💻</a></td>
    <td align="center"><a href="https://www.linkedin.com/in/changlinz/"><img src="https://avatars0.githubusercontent.com/u/29640620?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Changlin_NLP</b></sub></a><br /><a href="https://github.com/ThilinaRajapakse/simpletransformers/commits?author=alexucb" title="Code">💻</a></td>
    <td align="center"><a href="https://github.com/jpotoniec"><img src="https://avatars0.githubusercontent.com/u/11078342?v=4?s=100" width="100px;" alt=""/><br /><sub><b>jpotoniec</b></sub></a><br /><a href="https://github.com/ThilinaRajapakse/simpletransformers/commits?author=jpotoniec" title="Code">💻</a></td>
    <td align="center"><a href="https://github.com/fcggamou"><img src="https://avatars0.githubusercontent.com/u/20055856?v=4?s=100" width="100px;" alt=""/><br /><sub><b>fcggamou</b></sub></a><br /><a href="https://github.com/ThilinaRajapakse/simpletransformers/commits?author=fcggamou" title="Code">💻</a> <a href="https://github.com/ThilinaRajapakse/simpletransformers/commits?author=fcggamou" title="Documentation">📖</a></td>
    <td align="center"><a href="https://github.com/guy-mor"><img src="https://avatars2.githubusercontent.com/u/44950985?v=4?s=100" width="100px;" alt=""/><br /><sub><b>guy-mor</b></sub></a><br /><a href="https://github.com/ThilinaRajapakse/simpletransformers/issues?q=author%3Aguy-mor" title="Bug reports">🐛</a> <a href="https://github.com/ThilinaRajapakse/simpletransformers/commits?author=guy-mor" title="Code">💻</a></td>
  </tr>
  <tr>
    <td align="center"><a href="https://github.com/cahya-wirawan"><img src="https://avatars1.githubusercontent.com/u/7669893?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Cahya Wirawan</b></sub></a><br /><a href="https://github.com/ThilinaRajapakse/simpletransformers/commits?author=cahya-wirawan" title="Code">💻</a></td>
    <td align="center"><a href="https://github.com/BjarkePedersen"><img src="https://avatars1.githubusercontent.com/u/29751977?v=4?s=100" width="100px;" alt=""/><br /><sub><b>BjarkePedersen</b></sub></a><br /><a href="https://github.com/ThilinaRajapakse/simpletransformers/commits?author=BjarkePedersen" title="Code">💻</a></td>
    <td align="center"><a href="https://github.com/tekkkon"><img src="https://avatars2.githubusercontent.com/u/6827543?v=4?s=100" width="100px;" alt=""/><br /><sub><b>tekkkon</b></sub></a><br /><a href="https://github.com/ThilinaRajapakse/simpletransformers/commits?author=tekkkon" title="Code">💻</a></td>
    <td align="center"><a href="https://www.linkedin.com/in/garg-amit/"><img src="https://avatars1.githubusercontent.com/u/19791871?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Amit Garg</b></sub></a><br /><a href="https://github.com/ThilinaRajapakse/simpletransformers/commits?author=Amit80007" title="Code">💻</a></td>
    <td align="center"><a href="https://github.com/caprone"><img src="https://avatars1.githubusercontent.com/u/15055331?v=4?s=100" width="100px;" alt=""/><br /><sub><b>caprone</b></sub></a><br /><a href="https://github.com/ThilinaRajapakse/simpletransformers/issues?q=author%3Acaprone" title="Bug reports">🐛</a></td>
    <td align="center"><a href="https://www.linkedin.com/in/ather-fawaz-024596134/"><img src="https://avatars0.githubusercontent.com/u/42374034?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Ather Fawaz</b></sub></a><br /><a href="https://github.com/ThilinaRajapakse/simpletransformers/commits?author=atherfawaz" title="Code">💻</a></td>
    <td align="center"><a href="https://santi.uy"><img src="https://avatars3.githubusercontent.com/u/3905501?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Santiago Castro</b></sub></a><br /><a href="https://github.com/ThilinaRajapakse/simpletransformers/commits?author=bryant1410" title="Documentation">📖</a></td>
  </tr>
  <tr>
    <td align="center"><a href="https://github.com/taranais"><img src="https://avatars1.githubusercontent.com/u/859916?v=4?s=100" width="100px;" alt=""/><br /><sub><b>taranais</b></sub></a><br /><a href="https://github.com/ThilinaRajapakse/simpletransformers/commits?author=taranais" title="Code">💻</a></td>
    <td align="center"><a href="http://pablomarino.me"><img src="https://avatars1.githubusercontent.com/u/14850762?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Pablo N. Marino</b></sub></a><br /><a href="https://github.com/ThilinaRajapakse/simpletransformers/commits?author=pablonm3" title="Code">💻</a> <a href="https://github.com/ThilinaRajapakse/simpletransformers/commits?author=pablonm3" title="Documentation">📖</a></td>
    <td align="center"><a href="http://linkedin.com/in/strawberrypie/"><img src="https://avatars2.githubusercontent.com/u/29224443?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Anton Kiselev</b></sub></a><br /><a href="https://github.com/ThilinaRajapakse/simpletransformers/commits?author=strawberrypie" title="Code">💻</a> <a href="https://github.com/ThilinaRajapakse/simpletransformers/commits?author=strawberrypie" title="Documentation">📖</a></td>
    <td align="center"><a href="https://github.com/Sxela"><img src="https://avatars0.githubusercontent.com/u/11751592?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Alex</b></sub></a><br /><a href="https://github.com/ThilinaRajapakse/simpletransformers/commits?author=Sxela" title="Code">💻</a></td>
    <td align="center"><a href="https://github.com/karthik19967829"><img src="https://avatars1.githubusercontent.com/u/35610230?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Karthik Ganesan</b></sub></a><br /><a href="https://github.com/ThilinaRajapakse/simpletransformers/commits?author=karthik19967829" title="Code">💻</a></td>
    <td align="center"><a href="https://www.facebook.com/profile.php?id=100009572680557"><img src="https://avatars2.githubusercontent.com/u/18054828?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Zhylko Dima</b></sub></a><br /><a href="https://github.com/ThilinaRajapakse/simpletransformers/commits?author=Zhylkaaa" title="Code">💻</a></td>
    <td align="center"><a href="https://jonatanklosko.com"><img src="https://avatars1.githubusercontent.com/u/17034772?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Jonatan Kłosko</b></sub></a><br /><a href="https://github.com/ThilinaRajapakse/simpletransformers/commits?author=jonatanklosko" title="Code">💻</a></td>
  </tr>
  <tr>
    <td align="center"><a href="https://github.com/sarapapi"><img src="https://avatars0.githubusercontent.com/u/57095209?v=4?s=100" width="100px;" alt=""/><br /><sub><b>sarapapi</b></sub></a><br /><a href="https://github.com/ThilinaRajapakse/simpletransformers/commits?author=sarapapi" title="Code">💻</a> <a href="#question-sarapapi" title="Answering Questions">💬</a></td>
    <td align="center"><a href="https://ab-cse.web.app"><img src="https://avatars0.githubusercontent.com/u/25720695?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Abdul</b></sub></a><br /><a href="https://github.com/ThilinaRajapakse/simpletransformers/commits?author=macabdul9" title="Code">💻</a></td>
    <td align="center"><a href="https://github.com/jamesmilliman"><img src="https://avatars1.githubusercontent.com/u/8591478?v=4?s=100" width="100px;" alt=""/><br /><sub><b>James Milliman</b></sub></a><br /><a href="https://github.com/ThilinaRajapakse/simpletransformers/commits?author=jamesmilliman" title="Documentation">📖</a></td>
    <td align="center"><a href="https://parmarsuraj99.github.io/suraj-parmar/"><img src="https://avatars3.githubusercontent.com/u/9317265?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Suraj Parmar</b></sub></a><br /><a href="https://github.com/ThilinaRajapakse/simpletransformers/commits?author=parmarsuraj99" title="Documentation">📖</a></td>
    <td align="center"><a href="https://velog.io/@kwanhong66"><img src="https://avatars3.githubusercontent.com/u/5180452?v=4?s=100" width="100px;" alt=""/><br /><sub><b>KwanHong Lee</b></sub></a><br /><a href="#question-kwanhong66" title="Answering Questions">💬</a></td>
    <td align="center"><a href="http://julielab.de/Staff/Erik+F%C3%A4%C3%9Fler.html"><img src="https://avatars1.githubusercontent.com/u/4648560?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Erik Fäßler</b></sub></a><br /><a href="https://github.com/ThilinaRajapakse/simpletransformers/commits?author=khituras" title="Code">💻</a></td>
    <td align="center"><a href="https://github.com/ohstopityou"><img src="https://avatars3.githubusercontent.com/u/21691517?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Thomas Søvik</b></sub></a><br /><a href="#question-ohstopityou" title="Answering Questions">💬</a></td>
  </tr>
  <tr>
    <td align="center"><a href="https://github.com/gaganmanku96"><img src="https://avatars0.githubusercontent.com/u/20324385?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Gagandeep Singh</b></sub></a><br /><a href="https://github.com/ThilinaRajapakse/simpletransformers/commits?author=gaganmanku96" title="Code">💻</a> <a href="https://github.com/ThilinaRajapakse/simpletransformers/commits?author=gaganmanku96" title="Documentation">📖</a></td>
    <td align="center"><a href="http://www.esuli.it/"><img src="https://avatars3.githubusercontent.com/u/6543521?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Andrea Esuli</b></sub></a><br /><a href="https://github.com/ThilinaRajapakse/simpletransformers/commits?author=aesuli" title="Code">💻</a></td>
    <td align="center"><a href="https://github.com/DM2493"><img src="https://avatars1.githubusercontent.com/u/59502011?v=4?s=100" width="100px;" alt=""/><br /><sub><b>DM2493</b></sub></a><br /><a href="https://github.com/ThilinaRajapakse/simpletransformers/commits?author=DM2493" title="Code">💻</a></td>
    <td align="center"><a href="https://mapmeld.com/ml"><img src="https://avatars0.githubusercontent.com/u/643918?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Nick Doiron</b></sub></a><br /><a href="https://github.com/ThilinaRajapakse/simpletransformers/commits?author=mapmeld" title="Code">💻</a></td>
    <td align="center"><a href="https://github.com/abhinavg97"><img src="https://avatars3.githubusercontent.com/u/26171694?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Abhinav Gupta</b></sub></a><br /><a href="https://github.com/ThilinaRajapakse/simpletransformers/commits?author=abhinavg97" title="Code">💻</a></td>
    <td align="center"><a href="https://martinnormark.com"><img src="https://avatars3.githubusercontent.com/u/67565?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Martin H. Normark</b></sub></a><br /><a href="https://github.com/ThilinaRajapakse/simpletransformers/commits?author=martinnormark" title="Documentation">📖</a></td>
    <td align="center"><a href="https://github.com/mossadhelali"><img src="https://avatars3.githubusercontent.com/u/56701763?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Mossad Helali</b></sub></a><br /><a href="https://github.com/ThilinaRajapakse/simpletransformers/commits?author=mossadhelali" title="Code">💻</a></td>
  </tr>
  <tr>
    <td align="center"><a href="https://github.com/calebchiam"><img src="https://avatars0.githubusercontent.com/u/14286996?v=4?s=100" width="100px;" alt=""/><br /><sub><b>calebchiam</b></sub></a><br /><a href="https://github.com/ThilinaRajapakse/simpletransformers/commits?author=calebchiam" title="Code">💻</a></td>
    <td align="center"><a href="https://www.sartiano.info"><img src="https://avatars0.githubusercontent.com/u/1573433?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Daniele Sartiano</b></sub></a><br /><a href="https://github.com/ThilinaRajapakse/simpletransformers/commits?author=daniele-sartiano" title="Code">💻</a></td>
    <td align="center"><a href="https://github.com/tuner007"><img src="https://avatars1.githubusercontent.com/u/46425391?v=4?s=100" width="100px;" alt=""/><br /><sub><b>tuner007</b></sub></a><br /><a href="https://github.com/ThilinaRajapakse/simpletransformers/commits?author=tuner007" title="Documentation">📖</a></td>
    <td align="center"><a href="https://github.com/nilboy"><img src="https://avatars2.githubusercontent.com/u/17962699?v=4?s=100" width="100px;" alt=""/><br /><sub><b>xia jiang</b></sub></a><br /><a href="https://github.com/ThilinaRajapakse/simpletransformers/commits?author=nilboy" title="Code">💻</a></td>
    <td align="center"><a href="http://purl.org/net/hbuschme"><img src="https://avatars.githubusercontent.com/u/122398?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Hendrik Buschmeier</b></sub></a><br /><a href="https://github.com/ThilinaRajapakse/simpletransformers/commits?author=hbuschme" title="Documentation">📖</a></td>
    <td align="center"><a href="https://github.com/potpath"><img src="https://avatars.githubusercontent.com/u/8481150?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Mana Borwornpadungkitti</b></sub></a><br /><a href="https://github.com/ThilinaRajapakse/simpletransformers/commits?author=potpath" title="Documentation">📖</a></td>
    <td align="center"><a href="https://github.com/rayline"><img src="https://avatars.githubusercontent.com/u/11944753?v=4?s=100" width="100px;" alt=""/><br /><sub><b>rayline</b></sub></a><br /><a href="https://github.com/ThilinaRajapakse/simpletransformers/commits?author=rayline" title="Code">💻</a></td>
  </tr>
  <tr>
    <td align="center"><a href="https://github.com/mhdhdri"><img src="https://avatars.githubusercontent.com/u/13150376?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Mehdi Heidari</b></sub></a><br /><a href="https://github.com/ThilinaRajapakse/simpletransformers/commits?author=mhdhdri" title="Code">💻</a></td>
    <td align="center"><a href="https://github.com/whr778"><img src="https://avatars.githubusercontent.com/u/5939523?v=4?s=100" width="100px;" alt=""/><br /><sub><b>William Roe</b></sub></a><br /><a href="https://github.com/ThilinaRajapakse/simpletransformers/commits?author=whr778" title="Code">💻</a></td>
    <td align="center"><a href="https://github.com/alvaroabascar"><img src="https://avatars.githubusercontent.com/u/7307772?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Álvaro Abella Bascarán</b></sub></a><br /><a href="https://github.com/ThilinaRajapakse/simpletransformers/commits?author=alvaroabascar" title="Code">💻</a></td>
  </tr>
</table>

<!-- markdownlint-restore -->

<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->

This project follows the [all-contributors](https://github.com/all-contributors/all-contributors) specification. Contributions of any kind welcome!

*If you should be on this list but you aren't, or you are on the list but don't want to be, please don't hesitate to contact me!*

---

## How to Contribute

### How to Update Docs

The latest version of the docs is hosted on [Github Pages](https://simpletransformers.ai/), if you want to help document Simple Transformers
below are the steps to edit the docs.
Docs are built using [Jekyll](https://jekyllrb.com/) library, refer to their webpage for a detailed explanation of how it works.

1) **Install [Jekyll](https://jekyllrb.com/)**: Run the command `gem install bundler jekyll`
2) **Visualizing the docs on your local computer**:
   In your terminal cd into the docs directory of this repo, eg: `cd simpletransformers/docs`
   From the docs directory run this command to serve the Jekyll docs locally: `bundle exec jekyll serve`
   Browse to http://localhost:4000 or whatever url you see in the console to visualize the docs.
3) **Edit and visualize changes**:
   All the section pages of our docs can be found under `docs/_docs` directory, you can edit any file you want by following the markdown format and visualize the changes after refreshing the browser tab.

---

## Acknowledgements

None of this would have been possible without the hard work by the HuggingFace team in developing the [Transformers](https://github.com/huggingface/transformers) library.

_`<div>`Icon for the Social Media Preview made by `<a href="https://www.flaticon.com/authors/freepik" title="Freepik">`Freepik`</a>` from `<a href="https://www.flaticon.com/" title="Flaticon">`www.flaticon.com`</a></div>`_
