---
title: Conversational AI Specifics
permalink: /docs/convAI-specifics/
excerpt: "Conversational AI Specifics"
last_modified_at: 2020/09/06 21:31:35
toc: true
---

Chatbot creation based on the Hugging Face [State-of-the-Art Conversational AI](https://github.com/huggingface/transfer-learning-conv-ai).


## Usage Steps

Using a `ConvAIModel` in Simple Transformers follows the [standard pattern](/docs/usage/#task-specific-models) except for the interaction functionality.

1. Initialize a `ConvAIModel`
2. Train the model with `train_model()`
3. Evaluate the model with `eval_model()`
4. Interact with the model `interact()`


## Supported model types

- GPT
- GPT2


## Interacting with a `ConvAIModel`

### `interact()`

The `interact()` method can be used to talk with the model (interactively). Optionally, you can provide a list of strings to the method which will be used to build a *persona* for the chatbot. If it is not given, a random personality from the PERSONA-CHAT dataset will be used.

### `interact_single()`

The `interact_single()` method can be used to communicate with the model through single messages, i.e. by providing the current message and the history of the conversation. Optionally, you can provide a list of strings to the method which will be used to build a *persona* for the chatbot. If it is not given, a random personality from the PERSONA-CHAT dataset will be used.
