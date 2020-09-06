---
title: Conversational AI Model
permalink: /docs/convAI-data-formats/
excerpt: "Conversational AI Model"
last_modified_at: 2020/09/06 21:30:12
toc: true
---

## Data Formats

Data format follows the [Facebook Persona-Chat](http://arxiv.org/abs/1801.07243) format. A JSON formatted version by Hugging Face is found [here](https://s3.amazonaws.com/datasets.huggingface.co/personachat/personachat_self_original.json). The JSON file is directly compatible with this library (and it will be automatically downloaded and used if no dataset is specified).

Each entry in personachat is a **dict** with two keys `personality` and `utterances`, the dataset is a list of entries.

- `personality`:  **list of strings** containing the personality of the agent
- `utterances`: **list of dictionaries**, each of which has two keys which are **lists of strings**.
  - `candidates`: [next_utterance_candidate_1, ..., next_utterance_candidate_19]
        The last candidate is the ground truth response observed in the conversational data
  - `history`: [dialog_turn_0, ... dialog_turn N], where N is an odd number since the other user starts every conversation.

Preprocessing:

- Spaces before periods at end of sentences
- everything lowercase

Example train data:

```json
[
    {
        "personality": [
            "i like computers .",
            "i like reading books .",
            "i like talking to chatbots .",
            "i love listening to classical music ."
        ],
        "utterances": [
            {
                "candidates": [
                    "i try to wear all black every day . it makes me feel comfortable .",
                    "well nursing stresses you out so i wish luck with sister",
                    "yeah just want to pick up nba nfl getting old",
                    "i really like celine dion . what about you ?",
                    "no . i live near farms .",
                    "mother taught me to cook ! we are looking for an exterminator .",
                    "i enjoy romantic movie . what is your favorite season ? mine is summer .",
                    "editing photos takes a lot of work .",
                    "you must be very fast . hunting is one of my favorite hobbies .",
                    "hi there . i'm feeling great! how about you ?"
                ],
                "history": [
                    "hi , how are you ?"
                ]
            },
            {
                "candidates": [
                    "i have trouble getting along with family .",
                    "i live in texas , what kind of stuff do you do in ",
                    "toronto ?",
                    "that's so unique ! veganism and line dancing usually don't mix !",
                    "no , it isn't that big . do you travel a lot",
                    "that's because they are real ; what do you do for work ?",
                    "i am lazy all day lol . my mom wants me to get a job and move out",
                    "i was born on arbor day , so plant a tree in my name",
                    "okay , i should not tell you , its against the rules ",
                    "i like to talk to chatbots too ! do you know why ? ."
                ],
                "history": [
                    "hi , how are you ?",
                    "hi there . i'm feeling great! how about you ?",
                    "not bad ! i am trying out this chatbot ."
                ]
            },
            {
                "candidates": [
                    "ll something like that . do you play games ?",
                    "does anything give you relief ? i hate taking medicine for mine .",
                    "i decorate cakes at a local bakery ! and you ?",
                    "do you eat lots of meat",
                    "i am so weird that i like to collect people and cats",
                    "how are your typing skills ?",
                    "yeah . i am headed to the gym in a bit to weight lift .",
                    "yeah you have plenty of time",
                    "metal is my favorite , but i can accept that people listen to country . haha",
                    "that's why you desire to be controlled . let me control you person one .",
                    "two dogs they are the best , how about you ?",
                    "you do art ? what kind of art do you do ?",
                    "i love watching baseball outdoors on sunny days .",
                    "oh i see . do you ever think about moving ? i do , it is what i want .",
                    "because i am a chatbot too, silly !"
                ],
                "history": [
                    "hi , how are you ?",
                    "hi there . i'm feeling great! how about you ?",
                    "not bad ! i am trying out this chatbot .",
                    "i like to talk to chatbots too ! do you know why ? .",
                    "no clue, why don't you tell me ?"
                ]
            }
        ]
    }
]
```

### ConvAIModel

`class simpletransformers.conv_ai.ConvAIModel ( model_type, model_name, args=None, use_cuda=True, cuda_device=-1, **kwargs)`
This class is used to build Conversational AI.

`Class attributes`

- `tokenizer`: The tokenizer to be used.
- `model`: The model to be used.
            model_name: Default Transformer model name or path to Transformer model file (pytorch_model.bin).
- `device`: The device on which the model will be trained and evaluated.
- `results`: A python dict of past evaluation results for the TransformerModel object.
- `args`: A python dict of arguments used for training and evaluation.

`Parameters`

- `model_type`: (required) str - The type of model to use.
- `model_name`: (required) str - The exact model to use. Could be a pretrained model name or path to a directory containing a model. See [Current Pretrained Models](#current-pretrained-models) for all available models.
- `args`: (optional) python dict - A dictionary containing any settings that should be overwritten from the default values.
- `use_cuda`: (optional) bool - Default = True. Flag used to indicate whether CUDA should be used.
- `cuda_device`: (optional) int - Default = -1. Used to specify which GPU should be used.

`class methods`
**`train_model(self, train_file=None, output_dir=None, show_running_loss=True, args=None, eval_file=None)`**

Trains the model using 'train_file'

Args:

- train_df: ath to JSON file containing training data. The model will be trained on this file.
            output_dir: The directory where model files will be saved. If not given, self.args['output_dir'] will be used.

- output_dir (optional): The directory where model files will be saved. If not given, self.args['output_dir'] will be used.

- show_running_loss (Optional): Set to False to prevent training loss being printed.

- args (optional): Optional changes to the args dict of the model. Any changes made will persist for the model.

- eval_file (optional): Evaluation data against which evaluation will be performed when evaluate_during_training is enabled. If not given when evaluate_during_training is enabled, the evaluation data from PERSONA-CHAT will be used.

Returns:

- None

**`eval_model(self, eval_file, output_dir=None, verbose=True, silent=False)`**

Evaluates the model on eval_file. Saves results to output_dir.

Args:

- eval_file: Path to JSON file containing evaluation data. The model will be evaluated on this file.
If not given, eval dataset from PERSONA-CHAT will be used.

- output_dir: The directory where model files will be saved. If not given, self.args['output_dir'] will be used.

- verbose: If verbose, results will be printed to the console on completion of evaluation.

- silent: If silent, tqdm progress bars will be hidden.

Returns:

- result: Dictionary containing evaluation results. (correct, similar, incorrect)

- text: A dictionary containing the 3 dictionaries correct_text, similar_text (the predicted answer is a substring of the correct answer or vise versa), incorrect_text.

**`interact(self, personality=None)`**

Interact with a model in the terminal.

Args:

- personality (optional): A list of sentences that the model will use to build a personality.
If not given, a random personality from PERSONA-CHAT will be picked.

```python
model.interact(
    personality=[
        "i like computers .",
        "i like reading books .",
        "i love classical music .",
        "i am very social ."
    ]
)
```

Returns:

- None

**`train(self, train_dataloader, output_dir, show_running_loss=True, eval_dataloader=None, verbose=verbose)`**

Trains the model on train_dataset.
*Utility function to be used by the train_model() method. Not intended to be used directly.*

**`evaluate(self, eval_file, output_dir, verbose=True, silent=False)`**

Evaluates the model on eval_file.
*Utility function to be used by the eval_model() method. Not intended to be used directly*

**`load_and_cache_examples(self, dataset_path=None, evaluate=False, no_cache=False, verbose=True, silent=False)`**

Loads, tokenizes, and prepares data for training and/or evaluation.
*Utility function for train() and eval() methods. Not intended to be used directly*
