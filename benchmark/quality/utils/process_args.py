# coding=utf-8
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field
from typing import Optional

import transformers


@dataclass
class ModelArguments:
    model_name: str = field(default=None, metadata={"help": "Output model local path, do not set manually"})


@dataclass
class PruneArguments:
    cache: str = field(default=None)
    prune_key_prefill: bool = field(default=False)
    prune_key_decode: bool = field(default=False)
    prune_value_prefill: bool = field(default=False)
    prune_value_decode: bool = field(default=False)
    start_layer: int = field(default=0)
    end_layer: int = field(default=32)
    sink: Optional[int] = field(default=None)
    local_window: Optional[int] = field(default=None)
    prune_key_prefill_ratio: Optional[float] = field(default=None)
    prune_value_prefill_ratio: Optional[float] = field(default=None)
    block_seq_size: Optional[int] = field(default=None)


@dataclass
class DataArguments:
    output_postfix: Optional[str] = field(
        default="./outputs",
        metadata={"help": "The output path."},
    )
    e: Optional[bool] = field(
        default=False,
        metadata={"help": "Evaluate on LongBench-E."},
    )
    fast_eval: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to use fast evaluation."},
    )


def process_args():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, PruneArguments))
    model_args, data_args, prune_args = parser.parse_args_into_dataclasses()
    return model_args, data_args, prune_args
