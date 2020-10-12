---
title: Seq2Seq Minimal Start
permalink: /docs/seq2seq-minimal-start/
excerpt: "Minimal start for Seq2Seq."
last_modified_at: 2020/10/12 14:09:22
---

## Generic Encoder-Decoder minimal start

```python
import logging

import pandas as pd
from simpletransformers.seq2seq import Seq2SeqModel, Seq2SeqArgs


logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

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

train_df = pd.DataFrame(train_data, columns=["input_text", "target_text"])

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

eval_df = pd.DataFrame(eval_data, columns=["input_text", "target_text"])

# Configure the model
model_args = Seq2SeqArgs()
model_args.num_train_epochs = 200
model_args.evaluate_generated_text = True
model_args.evaluate_during_training = True
model_args.evaluate_during_training_verbose = True

model = Seq2SeqModel(
    "roberta",
    "roberta-base",
    "bert-base-cased",
    args=model_args,
)

# Train the model
model.train_model(train_df, eval_data=eval_df)

# Evaluate the model
result = model.eval_model(eval_df)

# Use the model for prediction
print(
    model.predict(
        [
            "Tyson is a Cyclops, a son of Poseidon, and Percy Jackson’s half brother. He is the current general of the Cyclopes army."
        ]
    )
)

# Loading a saved model
model_reloaded = Seq2SeqModel(
    "roberta",
    encoder_decoder_name="outputs",
    args=model_args,
)

# Use the model for prediction
print(
    model_reloaded.predict(
        [
            "Tyson is a Cyclops, a son of Poseidon, and Percy Jackson’s half brother. He is the current general of the Cyclopes army."
        ]
    )
)

```

## BART Model


```python
import logging

import pandas as pd
from simpletransformers.seq2seq import Seq2SeqModel, Seq2SeqArgs


logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

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

train_df = pd.DataFrame(train_data, columns=["input_text", "target_text"])

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

eval_df = pd.DataFrame(eval_data, columns=["input_text", "target_text"])

# Configure the model
model_args = Seq2SeqArgs()
model_args.num_train_epochs = 200
model_args.evaluate_generated_text = True
model_args.evaluate_during_training = True
model_args.evaluate_during_training_verbose = True

model = Seq2SeqModel(
    encoder_decoder_type="bart",
    encoder_decoder_name="facebook/bart-large",
    args=model_args,
)

# Train the model
model.train_model(train_df, eval_data=eval_df)

# Evaluate the model
result = model.eval_model(eval_df)

# Use the model for prediction
print(
    model.predict(
        [
            "Tyson is a Cyclops, a son of Poseidon, and Percy Jackson’s half brother. He is the current general of the Cyclopes army."
        ]
    )
)

# Loading a saved model
model_reloaded = Seq2SeqModel(
    encoder_decoder_type="bart",
    encoder_decoder_name="outputs",
    args=model_args,
)

# Use the model for prediction
print(
    model_reloaded.predict(
        [
            "Tyson is a Cyclops, a son of Poseidon, and Percy Jackson’s half brother. He is the current general of the Cyclopes army."
        ]
    )
)

```


## MarianMT Model

**Note:** MarianMT models are not intended to be trained/fine-tuned. There are thousands of ready-to-use translation models available. See [here](/docs/seq2seq-model/#marianmt-models).
{: .notice--info}


```python
import logging

import pandas as pd
from simpletransformers.seq2seq import Seq2SeqModel, Seq2SeqArgs


logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

model_args = Seq2SeqArgs()
model_args.num_beams = 10

model = Seq2SeqModel(
    encoder_decoder_type="marian",
    encoder_decoder_name="Helsinki-NLP/opus-mt-en-de",
    args=model_args,
)

src = [
    "Perseus “Percy” Jackson is the main protagonist and the narrator of the Percy Jackson and the Olympians series.",
    "Annabeth Chase is one of the main protagonists in Percy Jackson and the Olympians.",
    "Thalia Grace is the daughter of Zeus, sister of Jason Grace. After several years as a pine tree on Half-Blood Hill, she got a new job leading the Hunters of Artemis.",
]

predictions = model.predict(src)

for en, de in zip(src, predictions):
    print("-------------")
    print(en)
    print(de)
    print()

```

## Guides

- [BART for Paraphrasing with Simple Transformers](https://towardsdatascience.com/bart-for-paraphrasing-with-simple-transformers-7c9ea3dfdd8c?source=friends_link&sk=07420669325ac550f86b86bad362633c)
