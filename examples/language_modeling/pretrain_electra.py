import logging
import os

import torch

from prepare_data import prepare_data
from simpletransformers.language_modeling import (
    LanguageModelingModel,
    LanguageModelingArgs,
)

# TODO consider using methods from DeBERTa to improve it
# TODO consider using unigram lm tokenizer: SentencePieceUnigramTokenizer

if __name__ == '__main__':
    prepare_data()

    os.environ['TOKENIZERS_PARALLELISM'] = "true"

    logging.basicConfig(level=logging.INFO)
    transformers_logger = logging.getLogger("transformers")
    transformers_logger.setLevel(logging.WARNING)

    cuda_available = torch.cuda.is_available()
    num_gpus = torch.cuda.device_count()

    total_batch_size = 256  # ELECTRA: 256 for base, 2048 for large
    max_batch_size_for_gpu = 32  # 16GB V100

    # IMPORTANT if we set the embedding_size to 128 instead of 768 we get problems if we run the tie_weights() function,
    # the weights of the generator_lm_head (in_features) are changing,
    # leading to dimension errors in matrix multiplication
    # Thus all calls to tie_weights have been disabled. Is this a problem?
    model_args = LanguageModelingArgs(
        generator_config={
            "max_position_embeddings": 4096,
            "embedding_size": 768,
            "hidden_size": 768,
            "num_hidden_layers": 3,
        },
        discriminator_config={
            "max_position_embeddings": 4096,
            "embedding_size": 768,
            "hidden_size": 768,
            "num_hidden_layers": 12,
        },
        reprocess_input_data=False,
        overwrite_output_dir=True,
        evaluate_during_training=True,
        n_gpu=num_gpus,  # run with python -m torch.distributed.launch pretrain_electra.py
        num_train_epochs=1,
        eval_batch_size=max_batch_size_for_gpu,
        train_batch_size=max_batch_size_for_gpu,
        gradient_accumulation_steps=int(total_batch_size / max_batch_size_for_gpu),
        learning_rate=2e-4,  # ELECTRA paper searched in 1e-4, 2e-4, 3e-4, 5e-4
        warmup_steps=10_000,  # as specified in ELECTRA paper
        dataset_type="simple",
        vocab_size=30000,
        use_longformer_electra=True,
    )

    train_file, test_file = "data/train.txt", "data/test.txt"

    model = LanguageModelingModel(
        "electra",
        None,
        args=model_args,
        train_files=train_file,
        use_cuda=cuda_available
    )

    # Train the model
    model.train_model(train_file, eval_file=test_file)

    # Evaluate the model
    result = model.eval_model(test_file)
