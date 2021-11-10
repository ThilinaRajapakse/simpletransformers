---
title: Retrieval Minimal Start
permalink: /docs/retrieval-minimal-start/
excerpt: "Minimal start for Retrieval."
last_modified_at: 2021/11/10 16:38:46
---

```python
import logging

import pandas as pd
from simpletransformers.retrieval import RetrievalModel, RetrievalArgs

logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)


train_data = [
    {
        "query_text": "Who is the protaganist of Dune?",
        "title": "Dune (novel)",
        "gold_passage": 'Dune is set in the distant future amidst a feudal interstellar society in which various noble houses control planetary fiefs. It tells the story of young Paul Atreides, whose family accepts the stewardship of the planet Arrakis. While the planet is an inhospitable and sparsely populated desert wasteland, it is the only source of melange, or "spice", a drug that extends life and enhances mental abilities. Melange is also necessary for space navigation, which requires a kind of multidimensional awareness and foresight that only the drug provides. As melange can only be produced on Arrakis, control of the planet is a coveted and dangerous undertaking. The story explores the multilayered interactions of politics, religion, ecology, technology, and human emotion, as the factions of the empire confront each other in a struggle for the control of Arrakis and its spice.',
    },
    {
        "query_text": "Who is the author of Dune?"
        "title": "Dune (novel)",
        "gold_passage": "Dune is a 1965 science fiction novel by American author Frank Herbert, originally published as two separate serials in Analog magazine. It tied with Roger Zelazny's This Immortal for the Hugo Award in 1966 and it won the inaugural Nebula Award for Best Novel. It is the first installment of the Dune saga; in 2003, it was described as the world's best-selling science fiction novel.",
    }
]

eval_data = [
    {
        "query_text": "How many Dune sequels did Herbet write?",
        "title": "Dune (novel)",
        "gold_passage": "Herbert wrote five sequels: Dune Messiah, Children of Dune, God Emperor of Dune, Heretics of Dune, and Chapterhouse: Dune. Following Herbert's death in 1986, his son Brian Herbert and author Kevin J. Anderson continued the series in over a dozen additional novels since 1999.",
    },
    {
        "query_text": "What is Arrakis?"
        "title": "Dune (novel)",
        "gold_passage": "Duke Leto Atreides of House Atreides, ruler of the ocean planet Caladan, is assigned by the Padishah Emperor Shaddam IV to serve as fief ruler of the planet Arrakis. Although Arrakis is a harsh and inhospitable desert planet, it is of enormous importance because it is the only planetary source of melange, or the \"spice\", a unique and incredibly valuable substance that extends human youth, vitality and lifespan — the official reason for its high demand in the Empire. It is also through the consumption of spice that the Guild navigators are able to navigate around the stars to find paths to planetary or spatial targets. Shaddam sees House Atreides as a potential future rival and threat, and conspires with House Harkonnen, currently in charge of spice harvesting on Arrakis and longstanding enemies of House Atreides, to destroy Leto and his family after their arrival. Leto is aware his assignment is a trap of some kind, but he must obey the Emperor’s orders.",
    }
]

eval_df = pd.DataFrame(
    eval_data
)


# Configure the model
model_args = RetrievalArgs()
model_args.num_train_epochs = 40
model_args.no_save = True

model_type = "dpr"
context_encoder_name = "facebook/dpr-ctx_encoder-single-nq-base"
question_encoder_name = "facebook/dpr-question_encoder-single-nq-base"

model = RetrievalModel(
    model_type=model_type,
    context_encoder_name=context_encoder_name,
    query_encoder_name=question_encoder_name,
)


# Train the model
model.train_model(train_df, eval_data=eval_df)

# Evaluate the model
result = model.eval_model(eval_df)

# Make predictions with the model
to_predict = [
    'Who was the author of "Dune"?',
]

predicted_passages, doc_ids, doc_vectors, doc_dicts = model.predict(to_predict)

```

