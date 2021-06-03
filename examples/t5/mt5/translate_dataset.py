import logging

from tqdm.auto import tqdm
import pandas as pd

from translation_models import load_german, load_dutch, load_swedish, load_romance


logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)


model_map = {
    "english-dutch": load_dutch(),
    "english-german": load_german(),
    "english-swedish": load_swedish(),
    "english-romance": load_romance(),
}


def do_translate(input_text, target_language=None):
    if target_language == "german":
        return model_map["english-german"].predict(input_text)
    elif target_language == "dutch":
        return model_map["english-dutch"].predict(input_text)
    elif target_language == "swedish":
        return model_map["english-swedish"].predict(input_text)
    elif target_language == "spanish":
        return model_map["english-romance"].predict(
            [">>es<< " + text for text in input_text]
        )
    elif target_language == "french":
        return model_map["english-romance"].predict(
            [">>fr<< " + text for text in input_text]
        )


def translate_dataset(input_file, target_language):
    df = pd.read_csv(input_file, sep="\t").astype(str)
    df = df[df["prefix"] == "binary classification"]
    input_text = df["input_text"].tolist()

    translated_text = do_translate(input_text, target_language=target_language)

    df["input_text"] = translated_text

    return df


languages = ["dutch", "german", "french", "swedish", "spanish"]

for lang in tqdm(languages):
    translated_dataset = translate_dataset("data/eval.tsv", lang)
    translated_dataset.to_csv(f"data/{lang}_eval.tsv", "\t")
