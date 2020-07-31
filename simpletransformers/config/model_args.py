from dataclasses import dataclass, field, fields, asdict
from multiprocessing import cpu_count
import json
import sys
import os

from torch.utils.data import Dataset


def get_default_process_count():
    process_count = cpu_count() - 2 if cpu_count() > 2 else 1
    if sys.platform == "win32":
        process_count = min(process_count, 61)

    return process_count


def get_special_tokens():
    return ["<s>", "<pad>", "</s>", "<unk>", "<mask>"]


@dataclass
class ModelArgs:
    adam_epsilon: float = 1e-8
    best_model_dir: str = "outputs/best_model"
    cache_dir: str = "cache_dir/"
    custom_layer_parameters: list = field(default_factory=list)
    custom_parameter_groups: list = field(default_factory=list)
    train_custom_parameters_only: bool = False
    config: dict = field(default_factory=dict)
    dataloader_num_workers: int = field(default_factory=get_default_process_count)
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

    def update_from_dict(self, new_values):
        if isinstance(new_values, dict):
            for key, value in new_values.items():
                setattr(self, key, value)
        else:
            raise (TypeError(f"{new_values} is not a Python dict."))

    def save(self, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "model_args.json"), "w") as f:
            json.dump(asdict(self), f)

    def load(self, input_dir):
        if input_dir:
            model_args_file = os.path.join(input_dir, "model_args.json")
            if os.path.isfile(model_args_file):
                with open(model_args_file, "r") as f:
                    model_args = json.load(f)

                self.update_from_dict(model_args)


@dataclass
class ClassificationArgs(ModelArgs):
    """
    Model args for a ClassificationModel
    """

    labels_list: list = field(default_factory=list)
    labels_map: dict = field(default_factory=dict)
    lazy_delimiter: str = "\t"
    lazy_labels_column: int = 1
    lazy_loading: bool = False
    lazy_loading_start_line: int = 1
    lazy_text_a_column: bool = None
    lazy_text_b_column: bool = None
    lazy_text_column: int = 0
    regression: bool = False
    sliding_window: bool = False
    stride: float = 0.8
    tie_value: int = 1


@dataclass
class MultiLabelClassificationArgs(ModelArgs):
    """
    Model args for a MultiLabelClassificationModel
    """

    sliding_window: bool = False
    stride: float = 0.8
    threshold: float = 0.5
    tie_value: int = 1
    labels_list: list = field(default_factory=list)
    labels_map: dict = field(default_factory=dict)
    lazy_loading: bool = False


@dataclass
class NERArgs(ModelArgs):
    """
    Model args for a NERModel
    """

    classification_report: bool = False
    labels_list: list = field(default_factory=list)
    lazy_loading: bool = False
    lazy_loading_start_line: int = 0


@dataclass
class QuestionAnsweringArgs(ModelArgs):
    """
    Model args for a QuestionAnsweringModel
    """

    doc_stride: int = 384
    early_stopping_metric: str = "correct"
    early_stopping_metric_minimize: bool = False
    lazy_loading: bool = False
    max_answer_length: int = 100
    max_query_length: int = 64
    n_best_size: int = 20
    null_score_diff_threshold: float = 0.0


@dataclass
class T5Args(ModelArgs):
    """
    Model args for a T5Model
    """

    dataset_class: Dataset = None
    do_sample: bool = False
    early_stopping: bool = True
    evaluate_generated_text: bool = False
    length_penalty: float = 2.0
    max_length: int = 20
    max_steps: int = -1
    num_beams: int = 1
    num_return_sequences: int = 1
    preprocess_inputs: bool = True
    repetition_penalty: float = 1.0
    top_k: float = None
    top_p: float = None
    use_multiprocessed_decoding: bool = True


@dataclass
class LanguageModelingArgs(ModelArgs):
    """
    Model args for a LanguageModelingModel
    """

    block_size: int = -1
    config_name: str = None
    dataset_class: Dataset = None
    dataset_type: str = "None"
    discriminator_config: dict = field(default_factory=dict)
    discriminator_loss_weight: float = 50.0
    generator_config: dict = field(default_factory=dict)
    max_steps: int = -1
    min_frequency: int = 2
    mlm: bool = True
    mlm_probability: float = 0.15
    sliding_window: bool = False
    special_tokens: list = field(default_factory=get_special_tokens)
    stride: float = 0.8
    tie_generator_and_discriminator_embeddings: bool = True
    tokenizer_name: str = None
    vocab_size: int = None
    clean_text: bool = True
    handle_chinese_chars: bool = True
    strip_accents: bool = True
    local_rank: int = -1


@dataclass
class Seq2SeqArgs(ModelArgs):
    """
    Model args for a Seq2SeqModel
    """

    base_marian_model_name: str = None
    dataset_class: Dataset = None
    do_sample: bool = False
    early_stopping: bool = True
    evaluate_generated_text: bool = False
    length_penalty: float = 2.0
    max_length: int = 20
    max_steps: int = -1
    num_beams: int = 1
    num_return_sequences: int = 1
    repetition_penalty: float = 1.0
    top_k: float = None
    top_p: float = None
    use_multiprocessed_decoding: bool = False


@dataclass
class LanguageGenerationArgs(ModelArgs):
    """
    Model args for a LanguageGenerationModel
    """

    do_sample: bool = True
    early_stopping: bool = True
    evaluate_generated_text: bool = False
    length_penalty: float = 2.0
    max_length: int = 20
    max_steps: int = -1
    num_beams: int = 1
    num_return_sequences: int = 1
    repetition_penalty: float = 1.0
    top_k: float = 50
    top_p: float = 0.95
    prompt: str = ""
    stop_token: str = None
    temperature: float = 1.0
    padding_text: str = ""
    xlm_language: str = ""
    config_name: str = None
    tokenizer_name: str = None


@dataclass
class ConvAIArgs(ModelArgs):
    """
    Model args for a ConvAIModel
    """

    do_sample: bool = True
    lm_coef: float = 2.0
    max_history: int = 2
    max_length: int = 20
    mc_coef: float = 1.0
    min_length: int = 1
    num_candidates: int = 2
    personality_permutations: int = 1
    temperature: float = 0.7
    top_k: float = 0
    top_p: float = 0.9


@dataclass
class MultiModalClassificationArgs(ModelArgs):
    """
    Model args for a MultiModalClassificationModel
    """

    regression: bool = False
    num_image_embeds: int = 1
    text_label: str = "text"
    labels_label: str = "labels"
    images_label: str = "images"
    image_type_extension: str = ""
    data_type_extension: str = ""
