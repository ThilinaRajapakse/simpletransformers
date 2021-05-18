import pandas as pd


def load_rte_data_file(filepath):
    df = pd.read_json(filepath, lines=True)
    df = df.rename(
        columns={"premise": "text_a", "hypothesis": "text_b", "label": "labels"}
    )
    df = df[["text_a", "text_b", "labels"]]
    return df


def load_rte_test(filepath):
    df = pd.read_json(filepath, lines=True)
    df = df.rename(columns={"premise": "text_a", "hypothesis": "text_b"})
    return df
