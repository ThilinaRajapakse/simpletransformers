import warnings

import pandas as pd


def load_data(
    file_path, input_text_column, target_text_column, label_column, keep_label=1
):
    df = pd.read_csv(file_path, sep="\t", error_bad_lines=False)
    df = df.loc[df[label_column] == keep_label]
    df = df.rename(
        columns={input_text_column: "input_text", target_text_column: "target_text"}
    )
    df = df[["input_text", "target_text"]]
    df["prefix"] = "paraphrase"

    return df


def clean_unnecessary_spaces(out_string):
    if not isinstance(out_string, str):
        warnings.warn(f">>> {out_string} <<< is not a string.")
        out_string = str(out_string)
    out_string = (
        out_string.replace(" .", ".")
        .replace(" ?", "?")
        .replace(" !", "!")
        .replace(" ,", ",")
        .replace(" ' ", "'")
        .replace(" n't", "n't")
        .replace(" 'm", "'m")
        .replace(" 's", "'s")
        .replace(" 've", "'ve")
        .replace(" 're", "'re")
    )
    return out_string
