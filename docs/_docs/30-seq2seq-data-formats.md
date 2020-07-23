---
title: Seq2Seq Data Formats
permalink: /docs/seq2seq-data-formats/
excerpt: "Data formats for Seq2Seq."
last_modified_at: 2020/07/23 23:16:38
toc: true
---

As suggested by the name, both the inputs to and the outputs from a `Seq2SeqModel` is a sequence of text.

## Train Data Format

*Used with [`train_model()`](/docs/seq2seq-models/#training-a-seq2seq-model)*

The train data should be a Pandas DataFrame containing the 2 columns - `input_text`, `target_text`.

- `input_text`: The input text sequence.
- `target_text`: The target text sequence.


```python
train_data = [
    [
        "Perseus “Percy” Jackson is the main protagonist and the narrator of the Percy Jackson and the Olympians series.",
        "Percy is the protagonist of Percy Jackson and the Olympians",
    ],
    [
        "Annabeth Chase is one of the main protagonists in Percy Jackson and the Olympians.",
        "Annabeth is a protagonist in Percy Jackson and the Olympians.",
    ],
]

train_df = pd.DataFrame(
    train_data, columns=["input_text", "target_text"]
)

```


## Evaluation Data Format

*Used with [`eval_model()`](/docs/qa-model/#evaluating-a-seq2seq-model)*

The evaluation data format is identical to the train data format.

The evaluation data should be a Pandas DataFrame containing the 2 columns - `input_text`, `target_text`.

- `input_text`: The input text sequence.
- `target_text`: The target text sequence.


```python
eval_data = [
    [
        "Grover Underwood is a satyr and the Lord of the Wild. He is the satyr who found the demigods Thalia Grace, Nico and Bianca di Angelo, Percy Jackson, Annabeth Chase, and Luke Castellan.",
        "Grover is a satyr who found many important demigods.",
    ],
    [
        "Thalia Grace is the daughter of Zeus, sister of Jason Grace. After several years as a pine tree on Half-Blood Hill, she got a new job leading the Hunters of Artemis.",
        "Thalia is the daughter of Zeus and leader of the Hunters of Artemis.",
    ],
]

eval_df = pd.DataFrame(
    eval_data, columns=["input_text", "target_text"]
)

```


## Prediction Data Format
*Used with [`predict()`](/docs/qa-model/#making-predictions-with-a-seq2seq-model)*

The prediction data should be a list of strings.


```python
to_predict = [
    "Tyson is a Cyclops, a son of Poseidon, and Percy Jackson’s half brother. He is the current general of the Cyclopes army.",
    "Clarisse is the daughter of Ares and longtime head of the Ares cabin at Camp Half-Blood."
]
```
