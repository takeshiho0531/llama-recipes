# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from .grammar_dataset import get_dataset as get_grammar_dataset
from .alpaca_dataset import InstructionDataset as get_alpaca_dataset
from .samsum_dataset import get_preprocessed_samsum as get_samsum_dataset
from .wiki40b_ja_dataset import get_preprocessed_wiki40b_ja as get_wiki40b_ja_dataset
from .wiki_ja_small_dataset import get_preprocessed_wiki_ja_small as get_wiki_ja_small_dataset