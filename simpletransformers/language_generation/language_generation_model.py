import argparse
import logging
import random
import os
import json

import numpy as np

import torch
from simpletransformers.config.global_args import global_args
from simpletransformers.config.model_args import LanguageGenerationArgs
from simpletransformers.language_generation.language_generation_utils import PREPROCESSING_FUNCTIONS
from transformers import (
    CTRLConfig,
    CTRLLMHeadModel,
    CTRLTokenizer,
    GPT2Config,
    GPT2LMHeadModel,
    GPT2Tokenizer,
    OpenAIGPTConfig,
    OpenAIGPTLMHeadModel,
    OpenAIGPTTokenizer,
    TransfoXLConfig,
    TransfoXLLMHeadModel,
    TransfoXLTokenizer,
    XLMConfig,
    XLMTokenizer,
    XLMWithLMHeadModel,
    XLNetConfig,
    XLNetLMHeadModel,
    XLNetTokenizer,
)

logger = logging.getLogger(__name__)

MAX_LENGTH = int(10000)  # Hardcoded max length to avoid infinite loop


class LanguageGenerationModel:
    def __init__(
        self, model_type, model_name, args=None, use_cuda=True, cuda_device=-1, **kwargs,
    ):

        """
        Initializes a LanguageGenerationModel model.

        Args:
            model_type: The type of model (gpt2, ctrl, openai-gpt, xlnet, transfo-xl, xlm)
            model_name: Default Transformer model name or path to a directory containing Transformer model file (pytorch_nodel.bin).
            args (optional): Default args will be used if this parameter is not provided. If provided, it should be a dict containing the args that should be changed in the default args.
            use_cuda (optional): Use GPU if available. Setting to False will force model to use CPU only.
            cuda_device (optional): Specific GPU that should be used. Will use the first available GPU by default.
            **kwargs (optional): For providing proxies, force_download, resume_download, cache_dir and other options specific to the 'from_pretrained' implementation where this will be supplied.
        """  # noqa: ignore flake8"

        MODEL_CLASSES = {
            "gpt2": (GPT2Config, GPT2LMHeadModel, GPT2Tokenizer),
            "ctrl": (CTRLConfig, CTRLLMHeadModel, CTRLTokenizer),
            "openai-gpt": (OpenAIGPTConfig, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer),
            "xlnet": (XLNetConfig, XLNetLMHeadModel, XLNetTokenizer),
            "transfo-xl": (TransfoXLConfig, TransfoXLLMHeadModel, TransfoXLTokenizer),
            "xlm": (XLMConfig, XLMWithLMHeadModel, XLMTokenizer),
        }

        self.args = self._load_model_args(model_name)

        if isinstance(args, dict):
            self.args.update_from_dict(args)
        elif isinstance(args, LanguageGenerationArgs):
            self.args = args

        if "sweep_config" in kwargs:
            sweep_config = kwargs.pop("sweep_config")
            sweep_values = {key: value["value"] for key, value in sweep_config.as_dict().items() if key != "_wandb"}
            self.args.update_from_dict(sweep_values)

        if self.args.manual_seed:
            random.seed(self.args.manual_seed)
            np.random.seed(self.args.manual_seed)
            torch.manual_seed(self.args.manual_seed)
            if self.args.n_gpu > 0:
                torch.cuda.manual_seed_all(self.args.manual_seed)

        if use_cuda:
            if torch.cuda.is_available():
                if cuda_device == -1:
                    self.device = torch.device("cuda")
                else:
                    self.device = torch.device(f"cuda:{cuda_device}")
            else:
                raise ValueError(
                    "'use_cuda' set to True when cuda is unavailable."
                    "Make sure CUDA is available or set `use_cuda=False`."
                )
        else:
            self.device = "cpu"

        self.args.model_name = model_name
        self.args.model_type = model_type

        config_class, model_class, tokenizer_class = MODEL_CLASSES[model_type]

        if self.args.tokenizer_name:
            self.tokenizer = tokenizer_class.from_pretrained(self.args.tokenizer_name, cache_dir=self.args.cache_dir)
        else:
            self.tokenizer = tokenizer_class.from_pretrained(model_name, cache_dir=self.args.cache_dir, **kwargs)
            self.args.tokenizer_name = model_name

        if self.args.config_name:
            self.config = config_class.from_pretrained(self.args.config_name, cache_dir=self.args.cache_dir)
        else:
            self.config = config_class.from_pretrained(model_name, cache_dir=self.args.cache_dir, **kwargs)

        self.model = model_class.from_pretrained(
            model_name, config=self.config, cache_dir=self.args.cache_dir, **kwargs,
        )

        self.model.to(self.device)

    def generate(self, prompt=None, args=None, verbose=True):

        """
        Generate text using a LanguageGenerationModel

        Args:
            prompt (optional): A prompt text for the model. If given, will override args.prompt
            args (optional): Optional changes to the args dict of the model. Any changes made will persist for the model.
            verbose (optional): If verbose, generated text will be logged to the console.
        Returns:
            generated_sequences: Sequences of text generated by the model.
        """  # noqa: ignore flake8"

        model = self.model
        tokenizer = self.tokenizer
        device = self.device

        if args:
            self.args.update_from_dict(args)

        if prompt:
            self.args.prompt = prompt
        elif not self.args.prompt:
            self.args.prompt = input("Model prompt >>> ")

        prompt_text = self.args.prompt
        args = self.args

        # Different models need different input formatting and/or extra arguments
        requires_preprocessing = args.model_type in PREPROCESSING_FUNCTIONS.keys()
        if requires_preprocessing:
            prepare_input = PREPROCESSING_FUNCTIONS.get(args.model_type)
            preprocessed_prompt_text = prepare_input(args, model, tokenizer, prompt_text)
            encoded_prompt = tokenizer.encode(
                preprocessed_prompt_text,
                add_special_tokens=False,
                return_tensors="pt",
                add_space_before_punct_symbol=True,
            )
        else:
            encoded_prompt = tokenizer.encode(prompt_text, add_special_tokens=False, return_tensors="pt")
        encoded_prompt = encoded_prompt.to(device)

        output_sequences = model.generate(
            input_ids=encoded_prompt,
            max_length=args.max_length + len(encoded_prompt[0]),
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            do_sample=args.do_sample,
            num_return_sequences=args.num_return_sequences,
        )

        # Remove the batch dimension when returning multiple sequences
        if len(output_sequences.shape) > 2:
            output_sequences.squeeze_()

        generated_sequences = []

        for generated_sequence_idx, generated_sequence in enumerate(output_sequences):
            if verbose:
                logger.info("=== GENERATED SEQUENCE {} ===".format(generated_sequence_idx + 1))
            generated_sequence = generated_sequence.tolist()

            # Decode text
            text = tokenizer.decode(generated_sequence, clean_up_tokenization_spaces=True)

            # Remove all text after the stop token
            text = text[: text.find(args.stop_token) if args.stop_token else None]

            # Add the prompt at the beginning of the sequence. Remove the excess text that was used for pre-processing
            total_sequence = (
                prompt_text + text[len(tokenizer.decode(encoded_prompt[0], clean_up_tokenization_spaces=True)) :]
            )

            generated_sequences.append(total_sequence)
            if verbose:
                logger.info(total_sequence)

        return generated_sequences

    def _save_model_args(self, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        self.args.save(output_dir)

    def _load_model_args(self, input_dir):
        args = LanguageGenerationArgs()
        args.load(input_dir)
        return args
