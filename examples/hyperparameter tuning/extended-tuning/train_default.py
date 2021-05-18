import logging
from statistics import mean

import pandas as pd
import wandb
from sklearn.metrics import accuracy_score

from simpletransformers.classification import ClassificationArgs, ClassificationModel
from utils import load_rte_data_file

logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

# Preparing train data
# train_df = load_rte_data_file("data/train.jsonl")
train_df = pd.read_json("data/augmented_train.jsonl", lines=True, orient="records")
eval_df = pd.read_json("data/eval_df", lines=True, orient="records")
test_df = pd.read_json("data/test_df", lines=True, orient="records")

model_args = ClassificationArgs()
model_args.eval_batch_size = 32
model_args.evaluate_during_training = True
model_args.evaluate_during_training_silent = False
model_args.evaluate_during_training_steps = -1
model_args.save_eval_checkpoints = False
model_args.save_model_every_epoch = False
model_args.learning_rate = 1e-5
model_args.manual_seed = 4
model_args.max_seq_length = 256
model_args.multiprocessing_chunksize = 5000
model_args.no_cache = True
model_args.num_train_epochs = 3
model_args.overwrite_output_dir = True
model_args.reprocess_input_data = True
model_args.train_batch_size = 16
model_args.gradient_accumulation_steps = 2
model_args.labels_list = ["not_entailment", "entailment"]
model_args.output_dir = "default_output"
model_args.best_model_dir = "default_output/best_model"
model_args.wandb_project = "RTE - Hyperparameter Optimization"
model_args.wandb_kwargs = {"name": "augmented-default"}

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
