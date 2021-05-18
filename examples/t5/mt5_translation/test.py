import logging
import sacrebleu
import pandas as pd
from simpletransformers.t5 import T5Model, T5Args


logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

model_args = T5Args()
model_args.max_length = 512
model_args.length_penalty = 1
model_args.num_beams = 10

model = T5Model("mt5", "outputs_base", args=model_args)

eval_df = pd.read_csv("data/eval.tsv", sep="\t").astype(str)

sinhala_truth = [
    eval_df.loc[eval_df["prefix"] == "translate english to sinhala"][
        "target_text"
    ].tolist()
]
to_sinhala = eval_df.loc[eval_df["prefix"] == "translate english to sinhala"][
    "input_text"
].tolist()

english_truth = [
    eval_df.loc[eval_df["prefix"] == "translate sinhala to english"][
        "target_text"
    ].tolist()
]
to_english = eval_df.loc[eval_df["prefix"] == "translate sinhala to english"][
    "input_text"
].tolist()

# Predict
sinhala_preds = model.predict(to_sinhala)

eng_sin_bleu = sacrebleu.corpus_bleu(sinhala_preds, sinhala_truth)
print("--------------------------")
print("English to Sinhalese: ", eng_sin_bleu.score)

english_preds = model.predict(to_english)

sin_eng_bleu = sacrebleu.corpus_bleu(english_preds, english_truth)
print("Sinhalese to English: ", sin_eng_bleu.score)
