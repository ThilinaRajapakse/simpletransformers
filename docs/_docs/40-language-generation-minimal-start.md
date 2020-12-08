---
title: Language Generation Minimal Start
permalink: /docs/language-generation-minimal-start/
excerpt: "Minimal start for Language Generation tasks."
last_modified_at: 2020/12/08 12:19:33
---

```python
import logging

from simpletransformers.language_generation import LanguageGenerationModel, LanguageGenerationArgs


logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

model = LanguageGenerationModel("gpt2", "gpt2")
model.generate("Let's give a minimal start to the model like")

```
