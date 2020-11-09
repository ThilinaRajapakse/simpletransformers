---
title: NER Data Formats
permalink: /docs/ner-data-formats/
excerpt: "Data formats for named entity recognition."
last_modified_at: 2020/11/09 18:27:48
toc: true
---

The input data to a Simple Transformers NER task can be either a Pandas DataFrame or a path to a text file containing the data. The option to use a text file, in addition to the typical DataFrame, is provided as a convenience as many NER datasets are available as text files. When using text files as input, the data should be in the [CoNLL](https://www.clips.uantwerpen.be/conll2003/ner/) format as detailed below.


## Data Formats

### Pandas DataFrame

A DataFrame containing the 3 columns `sentence_id`, `words`, `labels`. Each value in `words` will have a corresponding `labels` value. The `sentence_id` determines which words belong to a given sentence. I.e. the words from the same sequence should be assigned the same unique `sentence_id`.

Consider the two sentences below;

- Harry Potter was a student at Hogwarts
- Albus Dumbledore founded the Order of the Phoenix

These two sentences can be prepared in a DataFrame as follows.

| sentence_id | words      | labels |
| ----------- | ---------- | ------ |
| 0           | Harry      | B-PER  |
| 0           | Potter     | I-PER  |
| 0           | was        | O      |
| 0           | a          | O      |
| 0           | student    | B-MISC |
| 0           | at         | O      |
| 0           | Hogwarts   | B-LOC  |
| 1           | Albus      | B-PER  |
| 1           | Dumbledore | I-PER  |
| 1           | founded    | O      |
| 1           | the        | O      |
| 1           | Order      | B-ORG  |
| 1           | of         | I-ORG  |
| 1           | the        | I-ORG  |
| 1           | Phoenix    | I-ORG  |
{: .text-center}

### Text file in CoNLL format

The CoNLL format is a text file with one word per line with sentences separated by an empty line. The first *word* in a line should be the `word` and the last *word* should be the `label`.

Consider the two sentences below;

- Harry Potter was a student at Hogwarts
- Albus Dumbledore founded the Order of the Phoenix

These two sentences can be prepared in a CoNLL formatted text file as follows.

```text
Harry B-PER
Potter I-PER
was O
a O
student B-MISC
at B-PER
Hogwarts I-PER

Albus B-PER
Dumbledore I-PER
founded O
the O
Order B-ORG
of I-ORG
the I-ORG
Phoenix I-ORG
```

**Note:** You can use custom labels as explained in the [Custom Labels](/docs/ner-specifics/#custom-labels) section.
{: .notice--info}

### Data format for LayoutLM models

[LayoutLM](https://arxiv.org/pdf/1912.13318.pdf) model (LayoutLM: Pre-training of Text and Layout for
Document Image Understanding) is pre-trained to consider both the text and layout information for document image understanding and information extraction tasks.

Although the paper discusses using combinations of text, layout, and image features, Simple Transformers currently only supports text + layout as inputs.

The data format for LayoutLM is similar to the default format described above but it also includes the bounding box information (`x0`, `y0`, `x1`, `y1`) in addition to the text. Here, `x0` and `y0` are the coordinates of the top-left vertex of the bounding box and `x1` and `y1` are the coordinates of the bottom-right vertex of the bounding box.

**Note:** The bounding box coordinates must be normalized to between 0-1000 where (0,0) is the top-left corner of the image.
{: .notice--info}

Dataframe format:

| sentence_id | words      | labels | xo  | y0  | x1  | y1  |
| ----------- | ---------- | ------ | --- | --- | --- | --- |
| 0           | Harry      | B-PER  | 12  | 34  | 29  | 50  |
| 0           | Potter     | I-PER  | 30  | 34  | 40  | 50  |
| 0           | was        | O      | 56  | 23  | 78  | 45  |
| 0           | a          | O      | 11  | 24  | 16  | 56  |
| 0           | student    | B-MISC | 43  | 33  | 67  | 54  |
| 0           | at         | O      | 24  | 43  | 40  | 42  |
| 0           | Hogwarts   | B-LOC  | 43  | 24  | 40  | 50  |
| 1           | Albus      | B-PER  | 11  | 34  | 43  | 654 |
| 1           | Dumbledore | I-PER  | 30  | 24  | 42  | 43  |
| 1           | founded    | O      | 24  | 34  | 752 | 784 |
| 1           | the        | O      | 30  | 11  | 752 | 224 |
| 1           | Order      | B-ORG  | 11  | 34  | 543 | 543 |
| 1           | of         | I-ORG  | 30  | 11  | 654 | 714 |
| 1           | the        | I-ORG  | 30  | 11  | 224 | 999 |
| 1           | Phoenix    | I-ORG  | 30  | 34  | 654 | 752 |
{: .text-center}

CoNLL format:

```text
Harry B-PER 12 55 14 68 59
Potter I-PER 15 56 19 59
was O 20 30 40 50
a O 42 16 52 69
student B-MISC 14 56 44 95
at B-PER 58 110 64 180
Hogwarts I-PER 56 56 85 89

Albus B-PER 510 520 530 525
Dumbledore I-PER 518 548 539 569
founded O 645 648 656 657
the O 644 688 677 699
Order B-ORG 540 585 576 598
of I-ORG 956 902 968 904
the I-ORG 945 963 948 966
Phoenix I-ORG 914 912 924 926
```


**Note:** The bounding box coordinates in the two examples above are random values and may not be realistic.
{: .notice--info}

## Train Data Format

*Used with [`train_model()`](/docs/ner-model/#training-a-nermodel)*

Train data can be in the form of a Pandas DataFrame or in a CoNLL style formatted text file.

### Pandas

```python
train_data = [
    [0, "Harry", "B-PER"],
    [0, "Potter", "I-PER"],
    [0, "was", "O"],
    [0, "a", "O"],
    [0, "student", "B-MISC"],
    [0, "at", "O"],
    [0, "Hogwarts", "B-LOC"],
    [1, "Albus", "B-PER"],
    [1, "Dumbledore", "I-PER"],
    [1, "founded", "O"],
    [1, "the", "O"],
    [1, "Order", "B-ORG"],
    [1, "of", "I-ORG"],
    [1, "the", "I-ORG"],
    [1, "Phoenix", "I-ORG"],
]
train_data = pd.DataFrame(
    train_data, columns=["sentence_id", "words", "labels"]
)

```
### Text file

```python
train_data = "data/train.txt"
```

## Evaluation Data Format

*Used with [`eval_model()`](/docs/ner-model/#evaluating-a-nermodel)*

Evaluation data can be in the form of a Pandas DataFrame or in a CoNLL style formatted text file.

### Pandas

```python
eval_data = [
    [0, "Sirius", "B-PER"],
    [0, "Black", "I-PER"],
    [0, "was", "O"],
    [0, "a", "O"],
    [0, "prisoner", "B-MISC"],
    [0, "at", "O"],
    [0, "Azkaban", "B-LOC"],
    [1, "Lord", "B-PER"],
    [1, "Voldemort", "I-PER"],
    [1, "founded", "O"],
    [1, "the", "O"],
    [1, "Death", "B-ORG"],
    [1, "Eaters", "I-ORG"],
]
eval_data = pd.DataFrame(
    eval_data, columns=["sentence_id", "words", "labels"]
)

```

### Text file

```python
eval_data = "data/eval.txt"
```


## Prediction Data Format

*Used with [`predict()`](/docs/ner-model/#making-predictins-with-a-nermodel)*

The prediction data should be one of the following formats.

### Automatically split sentences on spaces

By default, the input should be a list of strings, where each string is a sequence on which the model will perform NER. Here, each string will be split into words by splitting on spaces.

```python
to_pedict = [
    "Ron is Harry's best friend",
    "Hermione was the best in her class",
]
```

### Manually split sentences

While splitting on spaces makes sense in most situations, you may wish to decide how to split the sentences yourself. This may be particularly useful when working with languages other than English, where splitting on spaces might not be ideal.

In this case, the inputs should be a list of lists, with the inner list containing the split words of a sequence, while the outer list contains the list of sequences.

```python
to_predict = [
    ["Ron", "is", "Harry's", "best", "friend"],
    ["Hrmione", "was", "the ", "best", "in", "her", "class"],
]
```

**Note:** You must pass `split_on_space=False` to the `predict()` method when manually splitting sentences. See [`predict()`](/docs/ner-model/#making-predictins-with-a-nermodel) method.
{: .notice--warning}
