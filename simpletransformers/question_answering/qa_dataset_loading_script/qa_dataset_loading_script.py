from __future__ import absolute_import, division, print_function

import json
from dataclasses import dataclass

import datasets


"""
Adapted from the Huggingface code at https://github.com/huggingface/datasets/blob/master/datasets/squad_v2/squad_v2.py
"""


class QAConfig(datasets.BuilderConfig):
    """BuilderConfig for SQUAD."""

    def __init__(self, is_training, **kwargs):
        """BuilderConfig for SQUADV2.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(QAConfig, self).__init__(**kwargs)
        self.is_training = is_training


class QA(datasets.GeneratorBasedBuilder):
    BUILDER_CONFIG_CLASS = QAConfig

    def _info(self):
        return datasets.DatasetInfo(
            features=datasets.Features(
                {
                    "qas_id": datasets.Value("string"),
                    "question_text": datasets.Value("string"),
                    "context_text": datasets.Value("string"),
                    "answer_text": datasets.Value("string"),
                    "start_position_character": datasets.Value("int32"),
                    "is_impossible": datasets.Value("bool"),
                    "answers": datasets.features.Sequence(
                        {
                            "text": datasets.Value("string"),
                            "answer_start": datasets.Value("int32"),
                        }
                    ),
                }
            ),
            supervised_keys=None,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"filepath": self.config.data_files},
            ),
        ]

    def _generate_examples(self, filepath):
        """Yields examples."""
        # TODO(squad_v2): Yields (key, example) tuples from the dataset
        with open(filepath, encoding="utf-8") as f:
            examples_to_process = json.load(f)
            for paragraph in examples_to_process:
                context_text = paragraph["context"].strip()
                for qa in paragraph["qas"]:
                    qas_id = qa["id"]
                    question_text = qa["question"]
                    start_position_character = -1
                    answer_text = ""
                    answers = []

                    if "is_impossible" in qa:
                        is_impossible = qa["is_impossible"]
                    else:
                        is_impossible = False

                    if not is_impossible:
                        if self.config.is_training:
                            answer = qa["answers"][0]
                            answer_text = answer["text"]
                            start_position_character = answer["answer_start"]
                        else:
                            answers = qa["answers"]

                    yield qas_id, {
                        "qas_id": qas_id,
                        "question_text": question_text,
                        "context_text": context_text,
                        "answer_text": answer_text,
                        "start_position_character": start_position_character,
                        "is_impossible": is_impossible,
                        "answers": answers,
                    }
