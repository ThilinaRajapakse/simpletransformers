import os
from datasets import load_dataset


os.makedirs("data/msmarco", exist_ok=True)

print("=== Downloading MSMARCO ===")
print("Downloading MSMARCO training triples...")
dataset = load_dataset("thilina/negative-sampling")["train"]

print("Dataset loaded. Sample:")
print(dataset[0])

qrels = load_dataset("BeIR/msmarco-qrels")["validation"]

print("Saving dataset to disk...")
# Save the dataset to disk
dataset.to_csv("data/msmarco/msmarco-train.tsv", sep="\t", index=False)
qrels.to_csv("data/msmarco/devs.tsv", sep="\t", index=False)

print("Done.")
print("=== MSMARCO download complete ===")
