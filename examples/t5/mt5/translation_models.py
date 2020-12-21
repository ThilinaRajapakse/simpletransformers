import logging

from simpletransformers.seq2seq import Seq2SeqModel, Seq2SeqArgs


logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

model_args = Seq2SeqArgs()
model_args.reprocess_input_data = True
model_args.overwrite_output_dir = True
model_args.eval_batch_size = 64
model_args.use_multiprocessing = False
model_args.max_seq_length = 196
model_args.max_length = 512
model_args.num_beams = None
model_args.do_sample = True
model_args.top_k = 50
model_args.top_p = 0.95

use_cuda = True


def load_german():
    english_to_german_model = Seq2SeqModel(
        encoder_decoder_type="marian",
        encoder_decoder_name="Helsinki-NLP/opus-mt-en-de",
        use_cuda=use_cuda,
        args=model_args,
    )
    return english_to_german_model


def load_dutch():
    english_to_dutch_model = Seq2SeqModel(
        encoder_decoder_type="marian",
        encoder_decoder_name="Helsinki-NLP/opus-mt-en-nl",
        use_cuda=use_cuda,
        args=model_args,
    )
    return english_to_dutch_model


def load_swedish():
    english_to_swedish_model = Seq2SeqModel(
        encoder_decoder_type="marian",
        encoder_decoder_name="Helsinki-NLP/opus-mt-en-sw",
        use_cuda=use_cuda,
        args=model_args,
    )
    return english_to_swedish_model


def load_romance():
    english_to_romance_model = Seq2SeqModel(
        encoder_decoder_type="marian",
        encoder_decoder_name="Helsinki-NLP/opus-mt-en-roa",
        use_cuda=use_cuda,
        args=model_args,
    )
    return english_to_romance_model
