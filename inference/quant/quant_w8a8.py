#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Qwen3-4B-Instruct-2507 量化（llmcompressor 0.14 API）。

用法: python3 quant_w8a8.py [INT8|W4A16]
校准集直接以 datasets.Dataset 对象传入（绕开 registry/路径语义，第 7 跑定稿形态）。
"""
import sys
import json as _json
from datasets import Dataset
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier

SCHEME = sys.argv[1] if len(sys.argv) > 1 else "INT8"
SUFFIX = {"INT8": "W8A8-INT8", "W4A16": "W4A16"}.get(SCHEME, SCHEME)
SRC = "/models/Qwen3-4B-Instruct-2507"
OUT = "/models/Qwen3-4B-Instruct-2507-" + SUFFIX

recipe = QuantizationModifier(targets="Linear", scheme=SCHEME, ignore=["lm_head"])

oneshot(
    model=SRC,
    tokenizer=SRC,
    recipe=recipe,
    output_dir=OUT,
    dataset=Dataset.from_list(
        _json.load(open("/models/calib_data/calib.json"))),
    max_seq_length=2048,
    num_calibration_samples=256,
)
print("QUANT_DONE", OUT)
