import pandas as pd

df = pd.read_csv("data/cs.AI.tsv", sep="\t")
abstracts = df["abstract"].tolist()

with open("data/train.txt", "w") as f:
    for abstract in abstracts[:-10]:
        f.writelines(abstract + "\n")

with open("data/test.txt", "w") as f:
    for abstract in abstracts[-10:]:
        f.writelines(abstract + "\n")
