from simpletransformers.language_modeling import LanguageModelingModel
import logging
import argparse

logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

train_args = {
    "reprocess_input_data": False,
    "overwrite_output_dir": True,
    "num_train_epochs": 3,
    "save_eval_checkpoints": True,
    "save_model_every_epoch": False,
    "learning_rate": 5e-4,
    "warmup_steps": 10000,
    "train_batch_size": 32,
    "eval_batch_size": 128,
    "gradient_accumulation_steps": 1,
    "block_size": 128,
    "max_seq_length": 128,
    "dataset_type": "line_by_line",
    "wandb_project": "GPT2 - Indonesian",
    "wandb_kwargs": {"name": "GPT2-SMALL"},
    "logging_steps": 100,
    "evaluate_during_training": True,
    "evaluate_during_training_steps": 50000,
    "evaluate_during_training_verbose": True,
    "use_cached_eval_features": True,
    "sliding_window": True,
    "vocab_size": 52000,
    "generator_config": {
        "embedding_size": 128,
        "hidden_size": 256,
        "num_hidden_layers": 3,
    },
    "discriminator_config": {
        "embedding_size": 128,
        "hidden_size": 256,
    },
    "fp16": False,
    "mlm": False,
    "local_rank": -1,
}

parser = argparse.ArgumentParser()
parser.add_argument("--local_rank", type=int, default=-1,
                    help="Local rank. Necessary for using the torch.distributed.launch utility.")
args = parser.parse_args()

train_args["local_rank"] = args.local_rank

train_file = "/mnt/mldata/data/LM/ulmfit/wiki/id-2/valid.txt"
test_file = "/mnt/mldata/data/LM/ulmfit/wiki/id-2/test.txt"

model = LanguageModelingModel(
    "gpt2",
    None,
    args=train_args,
    train_files=train_file,
)


model.train_model(
    train_file, eval_file=test_file,
)

model.eval_model(test_file)