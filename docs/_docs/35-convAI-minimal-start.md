---
title: Conversational AI Examples
permalink: /docs/convAI-minimal-start/
excerpt: "Conversational AI Examples"
last_modified_at: 2020/10/15 23:16:38
toc: true
---

### Minimal Example

You can download the pretrained (OpenAI GPT based) Conversation AI model open-sourced by Hugging Face [here](https://s3.amazonaws.com/models.huggingface.co/transfer-learning-chatbot/gpt_personachat_cache.tar.gz).

For the minimal example given below, you can download the model and extract it to `gpt_personachat_cache`. Note that you can use any of the other GPT or GPT-2 models but they will require more training.

You will also need to create the JSON file given in the Data Format section and save it as `data/minimal_train.json`.

```python
from simpletransformers.conv_ai import ConvAIModel


train_args = {
    "num_train_epochs": 50,
    "save_model_every_epoch": False,
}

# Create a ConvAIModel
model = ConvAIModel("gpt", "gpt_personachat_cache", use_cuda=True, args=train_args)

# Train the model
model.train_model("data/minimal_train.json")

# Evaluate the model
model.eval_model()

# Interact with the trained model.
model.interact()

```

The `interact()` method can be given a list of Strings which will be used to build a personality. If a list of Strings is not given, a random personality will be chosen from PERSONA-CHAT instead.

### Real Dataset Example

- [Persona-Chat Conversational AI](https://medium.com/@chaturangarajapakshe/how-to-train-your-chatbot-with-simple-transformers-da25160859f4?sk=edd04e406e9a3523fcfc46102529e775)
