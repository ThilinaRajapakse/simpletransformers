import logging

import pandas as pd
from simpletransformers.retrieval import RetrievalModel, RetrievalArgs

# Configuring logging
logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

# Specifying the path to the training data
train_data_path = "data/msmarco/msmarco-train.tsv"

# Loading the training data
if train_data_path.endswith(".tsv"):
    train_data = pd.read_csv(train_data_path, sep="\t")
else:
    train_data = train_data_path

# Configuring the model arguments
model_args = RetrievalArgs()
model_args.reprocess_input_data = True
model_args.overwrite_output_dir = True
model_args.use_cached_eval_features = False
model_args.include_title = False if "msmarco" in train_data_path else True
model_args.max_seq_length = 256
model_args.num_train_epochs = 40
model_args.train_batch_size = 16
model_args.use_hf_datasets = True
model_args.learning_rate = 1e-6
model_args.warmup_steps = 5000
model_args.save_steps = 300000
model_args.evaluate_during_training = True
model_args.evaluate_during_training_steps = False
model_args.save_model_every_epoch = False
model_args.wandb_project = "Retrieval training example"
model_args.hard_negatives = False
model_args.n_gpu = 1
model_args.data_format = "beir"
model_args.output_dir = f"trained_models/pretrained/DPR-base-msmarco"
model_args.wandb_kwargs = {"name": f"DPR-base-msmarco"}

# Defining the model type and names
model_type = "custom"
model_name = None
context_name = "bert-base-multilingual-cased"
question_name = "bert-base-multilingual-cased"

# Main execution
if __name__ == "__main__":
    # Creating the model
    model = RetrievalModel(
        model_type,
        model_name,
        context_name,
        question_name,
        args=model_args,
    )

    # Training the model
    model.train_model(
        train_data,
        eval_set="dev",
    )
