import logging
import os

from simpletransformers.language_modeling import (
    LanguageModelingModel,
    LanguageModelingArgs,
)

if __name__ == '__main__':
    os.environ['TOKENIZERS_PARALLELISM'] = "true"

    logging.basicConfig(level=logging.INFO)
    transformers_logger = logging.getLogger("transformers")
    transformers_logger.setLevel(logging.WARNING)

    # IMPORTANT if we set the embedding_size to 128 instead of 768 we get problems if we run the tie_weights() function,
    # the weights of the generator_lm_head (in_features) are changing leading to dimension errors in matrix multiplication
    # Thus all calls to tie_weights have been disabled. Is this a problem?
    model_args = LanguageModelingArgs()
    model_args.generator_config = {
        "max_position_embeddings": 4096,
        "embedding_size": 768,
        "hidden_size": 768,
        "num_hidden_layers": 3,
    }
    model_args.discriminator_config = {
        "max_position_embeddings": 4096,
        "embedding_size": 768,
        "hidden_size": 768,
        "num_hidden_layers": 12,
    }

    model_args.reprocess_input_data = True
    model_args.overwrite_output_dir = True
    model_args.num_train_epochs = 1
    model_args.dataset_type = "simple"
    model_args.vocab_size = 30000
    model_args.use_longformer_electra = True
    model_args.tie_generator_and_discriminator_embeddings = True

    train_file = "data/train.txt"
    test_file = "data/test.txt"

    model = LanguageModelingModel(
        "electra",
        None,
        args=model_args,
        train_files=train_file,
        use_cuda=False
    )

    # Train the model
    model.train_model(train_file, eval_file=test_file)

    # Evaluate the model
    result = model.eval_model(test_file)
