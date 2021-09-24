import logging
from statistics import mean

import pandas as pd
import prettyprinter
import wandb
from prettyprinter import pprint
from sklearn.metrics import accuracy_score, f1_score

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
eval_df = pd.read_json("data/eval_df", lines=True, orient="records")
test_df = pd.read_json("data/test_df", lines=True, orient="records")

model_args = ClassificationArgs()
model_args.eval_batch_size = 32
model_args.evaluate_during_training = True
model_args.evaluate_during_training_silent = False
model_args.evaluate_during_training_steps = -1
model_args.learning_rate = 0.00003173
model_args.manual_seed = 4
model_args.max_seq_length = 256
model_args.multiprocessing_chunksize = 5000
model_args.no_cache = True
model_args.no_save = True
model_args.num_train_epochs = 40
model_args.overwrite_output_dir = True
model_args.reprocess_input_data = True
model_args.train_batch_size = 16
model_args.gradient_accumulation_steps = 2
model_args.train_custom_parameters_only = False
model_args.labels_list = ["not_entailment", "entailment"]
model_args.output_dir = "vanilla_output"
model_args.best_model_dir = "vanilla_output/best_model"
model_args.wandb_project = "RTE - Hyperparameter Optimization"
model_args.wandb_kwargs = {"name": "vanilla"}

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
