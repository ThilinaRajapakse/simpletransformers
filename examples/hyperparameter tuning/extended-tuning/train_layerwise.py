import logging
from statistics import mean

import pandas as pd
import prettyprinter
import wandb
from prettyprinter import pprint
from sklearn.metrics import accuracy_score

from simpletransformers.classification import ClassificationArgs, ClassificationModel
from utils import load_rte_data_file

prettyprinter.install_extras(
    include=[
        "dataclasses",
    ],
    warn_on_error=True,
)


logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

# Preparing train data
train_df = load_rte_data_file("data/train.jsonl")
eval_df = pd.read_json("data/eval_df.jsonl", lines=True, orient="records")
test_df = pd.read_json("data/test_df.jsonl", lines=True, orient="records")

sweep_result = pd.read_csv("sweep_results/deep-sweep.csv")

best_params = sweep_result.to_dict()

model_args = ClassificationArgs()
model_args.eval_batch_size = 32
model_args.evaluate_during_training = True
model_args.evaluate_during_training_silent = False
model_args.evaluate_during_training_steps = 1000
model_args.learning_rate = 4e-5
model_args.manual_seed = 4
model_args.max_seq_length = 256
model_args.multiprocessing_chunksize = 5000
model_args.no_cache = True
# model_args.no_save = True
model_args.num_train_epochs = 10
model_args.overwrite_output_dir = True
model_args.reprocess_input_data = True
model_args.train_batch_size = 16
model_args.gradient_accumulation_steps = 2
model_args.train_custom_parameters_only = False
model_args.save_eval_checkpoints = False
model_args.save_model_every_epoch = False
model_args.labels_list = ["not_entailment", "entailment"]
model_args.output_dir = "tuned_output"
model_args.best_model_dir = "tuned_output/best_model"
model_args.wandb_project = "RTE - Hyperparameter Optimization"
model_args.wandb_kwargs = {"name": "best-params"}

layer_params = []
param_groups = []
cleaned_args = {}

for key, value in best_params.items():
    if key.startswith("layer_"):
        layer_keys = key.split("_")[-1]
        start_layer = int(layer_keys.split("-")[0])
        end_layer = int(layer_keys.split("-")[-1])
        for layer_key in range(start_layer, end_layer):
            layer_params.append(
                {
                    "layer": layer_key,
                    "lr": value[0],
                }
            )
    elif key.startswith("params_"):
        params_key = key.split("_")[-1]
        param_groups.append(
            {
                "params": [params_key],
                "lr": value[0],
                "weight_decay": model_args.weight_decay
                if "bias" not in params_key
                else 0.0,
            }
        )
    elif key == "num_train_epochs":
        cleaned_args[key] = value[0]

cleaned_args["custom_layer_parameters"] = layer_params
cleaned_args["custom_parameter_groups"] = param_groups
model_args.update_from_dict(cleaned_args)

pprint(model_args)

# Create a TransformerModel
model = ClassificationModel("roberta", "roberta-large", use_cuda=True, args=model_args)

# Train the model
model.train_model(
    train_df,
    eval_df=eval_df,
    accuracy=lambda truth, predictions: accuracy_score(
        truth, [round(p) for p in predictions]
    ),
)

model.eval_model(test_df, verbose=True)
