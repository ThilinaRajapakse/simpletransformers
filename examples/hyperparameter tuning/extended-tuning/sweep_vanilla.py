import logging
from statistics import mean, mode

import pandas as pd
import wandb
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

from simpletransformers.classification import ClassificationArgs, ClassificationModel
from utils import load_rte_data_file

sweep_config = {
    "name": "vanilla-sweep-batch-16",
    "method": "bayes",
    "metric": {"name": "accuracy", "goal": "maximize"},
    "parameters": {
        "num_train_epochs": {"min": 1, "max": 40},
        "learning_rate": {"min": 0, "max": 4e-4},
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
model_args.eval_batch_size = 8
model_args.evaluate_during_training = True
model_args.evaluate_during_training_silent = False
model_args.evaluate_during_training_steps = 1000
model_args.learning_rate = 4e-4
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

    # Create a TransformerModel
    model = ClassificationModel(
        "roberta",
        "roberta-large",
        use_cuda=True,
        args=model_args,
        sweep_config=wandb.config,
    )

    # Train the model
    model.train_model(
        train_df,
        eval_df=eval_df,
        accuracy=lambda truth, predictions: accuracy_score(
            truth, [round(p) for p in predictions]
        ),
    )

    # Sync wandb
    wandb.join()


wandb.agent(sweep_id, train)
