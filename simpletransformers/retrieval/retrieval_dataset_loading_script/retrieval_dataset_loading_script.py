from __future__ import absolute_import, division, print_function

import json

import datasets


"""
Adapted from the Huggingface code at https://github.com/huggingface/datasets/blob/master/datasets/squad_v2/squad_v2.py
"""


class RetrievalConfig(datasets.BuilderConfig):
    """BuilderConfig for DPR style JSON."""

    def __init__(self, hard_negatives, include_title, **kwargs):
        """BuilderConfig for DPR style JSON.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(RetrievalConfig, self).__init__(**kwargs)
        self.hard_negatives = hard_negatives
        self.include_title = include_title


class Retrieval(datasets.GeneratorBasedBuilder):
    BUILDER_CONFIG_CLASS = RetrievalConfig

    def _info(self):
        if self.config.hard_negatives:
            return datasets.DatasetInfo(
                features=datasets.Features(
                    {
                        "query_text": datasets.Value("string"),
                        "gold_passage": datasets.Value("string"),
                        "hard_negative": datasets.Value("string"),
                    }
                ),
                supervised_keys=None,
            )
        else:
            return datasets.DatasetInfo(
                features=datasets.Features(
                    {
                        "query_text": datasets.Value("string"),
                        "gold_passage": datasets.Value("string"),
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
        with open(filepath, encoding="utf-8") as f:
            examples_to_process = json.load(f)
            for i, example in enumerate(examples_to_process):
                query_text = example["question"]
                if self.config.include_title:
                    passage = (
                        example["positive_ctxs"][0]["title"]
                        + " "
                        + example["positive_ctxs"][0]["text"]
                    )
                else:
                    passage = example["positive_ctxs"][0]["text"]
                gold_passage = passage

                if self.config.hard_negatives:
                    if example["hard_negative_ctxs"]:
                        if self.config.include_title:
                            hard_passage = (
                                example["hard_negative_ctxs"][0]["title"]
                                + " "
                                + example["hard_negative_ctxs"][0]["text"]
                            )
                        else:
                            hard_passage = example["hard_negative_ctxs"][0]["text"]
                        hard_negatives = hard_passage
                    elif example["negative_ctxs"]:
                        if self.config.include_title:
                            hard_passage = (
                                example["negative_ctxs"][0]["title"]
                                + " "
                                + example["negative_ctxs"][0]["text"]
                            )
                        else:
                            hard_passage = example["negative_ctxs"][0]["text"]
                        hard_negatives = hard_passage
                    else:
                        continue

                    yield i, {
                        "query_text": query_text,
                        "gold_passage": gold_passage,
                        "hard_negative": hard_negatives,
                    }
                else:
                    yield i, {
                        "query_text": query_text,
                        "gold_passage": gold_passage,
                    }
