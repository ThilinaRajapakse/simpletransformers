from dataclasses import dataclass, field
from multiprocessing import cpu_count
import sys


def get_default_process_count():
    process_count = cpu_count() - 2 if cpu_count() > 2 else 1
    if sys.platform == "win32":
        process_count = min(process_count, 61)

    return process_count


@dataclass
class ModelArgs:
    adam_epsilon: float = 1e-8
    best_model_dir: str = "outputs/best_model"
    cache_dir: str = "cache_dir/"
    config: dict = field(default_factory=dict)
    do_lower_case: bool = False
    early_stopping_consider_epochs: bool = False
    early_stopping_delta: float = 0
    early_stopping_metric: str = "eval_loss"
    early_stopping_metric_minimize: bool = True
    early_stopping_patience: int = 3
    encoding: str = None
    eval_batch_size: int = 8
    evaluate_during_training: bool = False
    evaluate_during_training_silent: bool = True
    evaluate_during_training_steps: int = 2000
    evaluate_during_training_verbose: bool = False
    fp16: bool = True
    fp16_opt_level: str = "O1"
    gradient_accumulation_steps: int = 1
    learning_rate: float = 4e-5
    local_rank: int = -1
    logging_steps: int = 50
    manual_seed: int = None
    max_grad_norm: float = 1.0
    max_seq_length: int = 128
    multiprocessing_chunksize: int = 500
    n_gpu: int = 1
    no_cache: bool = False
    no_save: bool = False
    num_train_epochs: int = 1
    output_dir: str = "outputs/"
    overwrite_output_dir: bool = False
    process_count: int = field(default_factory=get_default_process_count)
    reprocess_input_data: bool = True
    save_best_model: bool = True
    save_eval_checkpoints: bool = True
    save_model_every_epoch: bool = True
    save_steps: int = 2000
    save_optimizer_and_scheduler: bool = True
    silent: bool = False
    tensorboard_dir: str = None
    train_batch_size: int = 8
    use_cached_eval_features: bool = False
    use_early_stopping: bool = False
    use_multiprocessing: bool = True
    wandb_kwargs: dict = field(default_factory=dict)
    wandb_project: str = None
    warmup_ratio: float = 0.06
    warmup_steps: int = 0
    weight_decay: int = 0


@dataclass
class ClassificationArgs(ModelArgs):
    lazy_delimiter: str = "\t"
    lazy_header_row: bool = True
    lazy_labels_column: int = 1
    lazy_text_a_column: bool = None
    lazy_text_b_column: bool = None
    lazy_text_column: int = 0
    regression: bool = False
    sliding_window: bool = False
    stride: float = 0.8
    tie_value: int = 1
