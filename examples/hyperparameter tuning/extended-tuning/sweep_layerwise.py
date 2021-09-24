import logging
from statistics import mean

import pandas as pd
import wandb
from sklearn.metrics import accuracy_score

from simpletransformers.classification import ClassificationArgs, ClassificationModel
from utils import load_rte_data_file

layer_parameters = {
    f"layer_{i}-{i + 6}": {"min": 0.0, "max": 5e-5} for i in range(0, 24, 6)
}

sweep_config = {
    "name": "layerwise-sweep-batch-16",
    "method": "bayes",
    "metric": {"name": "accuracy", "goal": "maximize"},
    "parameters": {
        "num_train_epochs": {"min": 1, "max": 40},
        "params_classifier.dense.weight": {"min": 0, "max": 1e-3},
        "params_classifier.dense.bias": {"min": 0, "max": 1e-3},
        "params_classifier.out_proj.weight": {"min": 0, "max": 1e-3},
        "params_classifier.out_proj.bias": {"min": 0, "max": 1e-3},
        **layer_parameters,
    },
    "early_terminate": {
        "type": "hyperband",
        "min_iter": 6,
    },
}

sweep_id = wandb.sweep(sweep_config, project="RTE - Hyperparameter Optimization")

logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

# Preparing train data
train_df = load_rte_data_file("data/train.jsonl")
eval_df = pd.read_json("data/eval_df", lines=True, orient="records")
test_df = pd.read_json("data/test_df", lines=True, orient="records")

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
model_args.no_save = True
model_args.num_train_epochs = 10
model_args.overwrite_output_dir = True
model_args.reprocess_input_data = True
model_args.train_batch_size = 16
model_args.gradient_accumulation_steps = 2
model_args.train_custom_parameters_only = False
model_args.labels_list = ["not_entailment", "entailment"]
model_args.wandb_project = "RTE - Hyperparameter Optimization"


def train():
    # Initialize a new wandb run
    wandb.init()

    # Get sweep hyperparameters
    args = {
        key: value["value"]
        for key, value in wandb.config.as_dict().items()
        if key != "_wandb"
    }

    # Extracting the hyperparameter values
    cleaned_args = {}
    layer_params = []
    param_groups = []
    for key, value in args.items():
        if key.startswith("layer_"):
            # These are layer parameters
            layer_keys = key.split("_")[-1]

            # Get the start and end layers
            start_layer = int(layer_keys.split("-")[0])
            end_layer = int(layer_keys.split("-")[-1])

            # Add each layer and its value to the list of layer parameters
            for layer_key in range(start_layer, end_layer):
                layer_params.append(
                    {
                        "layer": layer_key,
                        "lr": value,
                    }
                )
        elif key.startswith("params_"):
            # These are parameter groups (classifier)
            params_key = key.split("_")[-1]
            param_groups.append(
                {
                    "params": [params_key],
                    "lr": value,
                    "weight_decay": model_args.weight_decay
                    if "bias" not in params_key
                    else 0.0,
                }
            )
        else:
            # Other hyperparameters (single value)
            cleaned_args[key] = value
    cleaned_args["custom_layer_parameters"] = layer_params
    cleaned_args["custom_parameter_groups"] = param_groups
    model_args.update_from_dict(cleaned_args)

    # Create a TransformerModel
    model = ClassificationModel(
        "roberta", "roberta-large", use_cuda=True, args=model_args
    )

    # Train the model
    model.train_model(
        train_df,
        eval_df=eval_df,
        accuracy=lambda truth, predictions: accuracy_score(
            truth, [round(p) for p in predictions]
        ),
    )

    # model.eval_model(eval_df, f1=f1_score)

    # Sync wandb
    wandb.join()


wandb.agent(sweep_id, train)
