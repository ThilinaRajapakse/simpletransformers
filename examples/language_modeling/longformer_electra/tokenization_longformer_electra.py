# coding=utf-8
# Copyright 2020 The Allen Institute for AI team and The HuggingFace Inc. team.
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
from transformers import LongformerTokenizer

from transformers.utils import logging

logger = logging.get_logger(__name__)

VOCAB_FILES_NAMES = {"vocab_file": "vocab.json", "merges_file": "merges.txt"}


class LongformerElectraTokenizer(LongformerTokenizer):
    r"""
    Construct a Longformer tokenizer.

    [`LongformerTokenizer`] is identical to [`RobertaTokenizer`]. Refer to the superclass for usage examples and
    documentation concerning parameters.
    """
    vocab_files_names = VOCAB_FILES_NAMES
