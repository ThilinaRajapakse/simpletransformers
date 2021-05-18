# coding=utf-8
# Copyright 2020 HuggingFace Datasets Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""Introduction to the CoNLL-2003 Shared Task: Language-Independent Named Entity Recognition"""

import logging

import datasets


"""
Adapted from the Huggingface code at https://github.com/huggingface/datasets/blob/master/datasets/conll2003/conll2003.py
"""


class NERConfig(datasets.BuilderConfig):
    """BuilderConfig for NER"""

    def __init__(self, **kwargs):
        """BuilderConfig for NER.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(NERConfig, self).__init__(**kwargs)


class NER(datasets.GeneratorBasedBuilder):
    """NER dataset."""

    BUILDER_CONFIG_CLASS = NERConfig

    def _info(self):
        return datasets.DatasetInfo(
            features=datasets.Features(
                {
                    "sentence_id": datasets.Value("string"),
                    "words": datasets.Sequence(datasets.Value("string")),
                    "labels": datasets.Sequence(datasets.Value("string")),
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
        logging.info("⏳ Generating examples from = %s", filepath)
        with open(filepath, encoding="utf-8") as f:
            guid = 0
            words = []
            labels = []
            for line in f:
                if line.startswith("-DOCSTART-") or line == "" or line == "\n":
                    if words:
                        yield guid, {
                            "sentence_id": str(guid),
                            "words": words,
                            "labels": labels,
                        }
                        guid += 1
                        words = []
                        labels = []
                else:
                    # conll2003 words are space separated
                    splits = line.split(" ")
                    words.append(splits[0])
                    labels.append(splits[-1].rstrip())
            # last example
            yield guid, {
                "sentence_id": str(guid),
                "words": words,
                "labels": labels,
            }
