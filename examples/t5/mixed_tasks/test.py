import json
from datetime import datetime
from pprint import pprint
from statistics import mean

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import accuracy_score, f1_score
from transformers.data.metrics.squad_metrics import compute_exact, compute_f1

from simpletransformers.t5 import T5Model


def f1(truths, preds):
    return mean([compute_f1(truth, pred) for truth, pred in zip(truths, preds)])


def exact(truths, preds):
    return mean([compute_exact(truth, pred) for truth, pred in zip(truths, preds)])


def pearson_corr(preds, labels):
    return pearsonr(preds, labels)[0]


def spearman_corr(preds, labels):
    return spearmanr(preds, labels)[0]


model_args = {
    "overwrite_output_dir": True,
    "max_seq_length": 196,
    "eval_batch_size": 32,
    "use_multiprocessing": False,
    "num_beams": None,
    "do_sample": True,
    "max_length": 50,
    "top_k": 50,
    "top_p": 0.95,
    "num_return_sequences": 3,
}

# Load the trained model
model = T5Model("outputs", args=model_args)

# Load the evaluation data
df = pd.read_csv("data/train.tsv", sep="\t").astype(str)

# Prepare the data for testing
to_predict = [
    prefix + ": " + str(input_text)
    for prefix, input_text in zip(df["prefix"].tolist(), df["input_text"].tolist())
]
truth = df["target_text"].tolist()
tasks = df["prefix"].tolist()

# Get the model predictions
preds = model.predict(to_predict)

# Saving the predictions if needed
with open(f"predictions/predictions_{datetime.now()}.txt", "w") as f:
    for i, text in enumerate(df["input_text"].tolist()):
        f.write(str(text) + "\n\n")

        f.write("Truth:\n")
        f.write(truth[i] + "\n\n")

        f.write("Prediction:\n")
        for pred in preds[i]:
            f.write(str(pred) + "\n")
        f.write(
            "________________________________________________________________________________\n"
        )

# Taking only the first prediction
preds = [pred[0] for pred in preds]
df["predicted"] = preds

# Evaluating the tasks separately
output_dict = {
    "binary classification": {
        "truth": [],
        "preds": [],
    },
    "multilabel classification": {
        "truth": [],
        "preds": [],
    },
    "similarity": {
        "truth": [],
        "preds": [],
    },
}

results_dict = {}

for task, truth_value, pred in zip(tasks, truth, preds):
    output_dict[task]["truth"].append(truth_value)
    output_dict[task]["preds"].append(pred)

print("-----------------------------------")
print("Results: ")
for task, outputs in output_dict.items():
    if task == "multilabel classification":
        try:
            task_truth = output_dict[task]["truth"]
            task_preds = output_dict[task]["preds"]
            results_dict[task] = {
                "F1 Score": f1(task_truth, task_preds),
                "Exact matches": exact(task_truth, task_preds),
            }
            print(f"Scores for {task}:")
            print(f"F1 score: {f1(task_truth, task_preds)}")
            print(f"Exact matches: {exact(task_truth, task_preds)}")
            print()
        except:
            pass
    elif task == "binary classification":
        try:
            task_truth = [int(t) for t in output_dict[task]["truth"]]
            task_preds = [int(p) for p in output_dict[task]["preds"]]
            results_dict[task] = {
                "F1 Score": f1_score(task_truth, task_preds),
                "Accuracy Score": accuracy_score(task_truth, task_preds),
            }
            print(f"Scores for {task}:")
            print(f"F1 score: {results_dict[task]['F1 Score']}")
            print(f"Accuracy Score: {results_dict[task]['Accuracy Score']}")
            print()
        except:
            pass
    if task == "similarity":
        task_truth = [float(t) for t in output_dict[task]["truth"]]
        task_preds = [float(p) for p in output_dict[task]["preds"]]
        results_dict[task] = {
            "Pearson Correlation": pearson_corr(task_truth, task_preds),
            "Spearman Correlation": spearman_corr(task_truth, task_preds),
        }
        print(f"Scores for {task}:")
        print(f"Pearson Correlation: {results_dict[task]['Pearson Correlation']}")
        print(f"Spearman Correlation: {results_dict[task]['Spearman Correlation']}")
        print()

with open(f"results/result_{datetime.now()}.json", "w") as f:
    json.dump(results_dict, f)
