import logging
import pandas as pd

from simpletransformers.language_modeling import (
    LanguageModelingModel,
    LanguageModelingArgs,
    GenerationArgs,
)
from simpletransformers.retrieval import (
    RetrievalModel,
    RetrievalArgs,
)


logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)
trainsformers_modules_logger = logging.getLogger("transformers_modules")
trainsformers_modules_logger.setLevel(logging.ERROR)


rag = True

if rag:
    from rag_setup import model as rag_model


train_file = "../data/squad-train.jsonl"


model_args = LanguageModelingArgs()
model_args.reprocess_input_data = True
model_args.overwrite_output_dir = True
model_args.num_train_epochs = 1
model_args.save_eval_checkpoints = False
model_args.save_model_every_epoch = False
model_args.train_batch_size = 2
model_args.eval_batch_size = 4
model_args.gradient_accumulation_steps = 1
model_args.manual_seed = 4
model_args.fp16 = True
model_args.dataset_type = "simple"
model_args.logging_steps = 100
model_args.evaluate_during_training = False
model_args.mlm = False
model_args.use_multiprocessing = False
model_args.use_hf_datasets = True
model_args.peft = True
model_args.qlora = False
model_args.nf4 = True
model_args.loftq_bits = 4
model_args.lora_config = {"r": 8}
model_args.data_format = "jsonl"
model_args.trust_remote_code = True
model_args.save_steps = 1000
model_args.optimizer = "Adam8bit"
model_args.chunk_text = False
model_args.max_seq_length = 500

if not rag:
    model_args.wandb_project = "llama-adapter-tuning-squad"
    model_args.wandb_kwargs = {"name": "squad-llama-2-7b-vanilla"}
    model_args.output_dir = "squad-llama-2-7b"

if rag:
    model_args.rag = rag
    model_args.wandb_project = "llama-adapter-tuning-squad"
    model_args.wandb_kwargs = {"name": "squad-llama-2-7b-rag"}
    model_args.output_dir = "squad-llama-2-7b-rag"


model = LanguageModelingModel(
    "causal",
    # "outputs",
    # "stabilityai/stablelm-zephyr-3b",
    "meta-llama/Llama-2-7b-hf",
    args=model_args,
    retrieval_model=rag_model if rag else None,
)

model.train_model(
    train_file,
)

# generation_args = GenerationArgs()
# generation_args.max_length = None
# generation_args.max_new_tokens = 100

# test_df = pd.read_json("data/test.jsonl", lines=True)

# to_predict = test_df["input_text"].tolist()

# responses, _ = model.predict(
#     to_predict,
#     generation_args=generation_args,
# )

# print(responses[:5])

# test_df["generated_text"] = responses

# test_df.to_json("data/test_output-finetuned.jsonl", orient="records", lines=True)
