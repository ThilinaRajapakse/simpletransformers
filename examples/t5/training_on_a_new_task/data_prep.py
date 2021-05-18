import gzip
import os

import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm


def parse(path):
    g = gzip.open(path, "rb")
    for l in g:
        yield eval(l)


def getDF(path):
    i = 0
    df = {}
    for d in parse(path):
        df[i] = d
        i += 1

    return pd.DataFrame.from_dict(df, orient="index")


categories = [
    category[3:]
    for category in os.listdir("data")
    if category.endswith(".gz") and category.startswith("qa")
]

for category in tqdm(categories):
    if not os.path.isfile(f"data/{category.split('.')[0]}.tsv"):
        try:
            df1 = getDF(f"data/qa_{category}")
            df2 = getDF(f"data/meta_{category}")

            df = pd.merge(df1, df2, on="asin", how="left")
            df = df[["question", "answer", "description"]]
            df = df.dropna()
            df = df.drop_duplicates(subset="answer")
            print(df.head())

            df.to_csv(f"data/{category.split('.')[0]}.tsv", "\t")
        except:
            pass

df = pd.concat(
    (
        pd.read_csv(f"data/{f}", sep="\t")
        for f in os.listdir("data")
        if f.endswith(".tsv")
    )
)
df = df[["question", "description"]]
df["description"] = df["description"].apply(lambda x: x[2:-2])
df.columns = ["target_text", "input_text"]
df["prefix"] = "ask_question"

df.to_csv(f"data/data_all.tsv", "\t")

train_df, eval_df = train_test_split(df, test_size=0.05)

train_df.to_csv("data/train_df.tsv", "\t")
eval_df.to_csv("data/eval_df.tsv", "\t")
