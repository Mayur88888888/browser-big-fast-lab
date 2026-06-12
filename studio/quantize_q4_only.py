#!/usr/bin/env python3
"""Quantize an fp32 decoder ONNX to 4-bit MatMulNBits — WITHOUT any float16 conversion.

Avoids the convert_float_to_float16 graph corruption (dangling InsertedPrecisionFreeCast at
layernorm) that produced garbage in the q4f16 build. Result: 4-bit weights + fp32 residuals
(embeddings, norms, scales) and fp32 KV cache → load it in the harness with ?kv=fp32.
Bigger than q4f16 (~6.6GB vs 5.36GB) but correct by construction.

Usage: python quantize_q4_only.py <fp32.onnx> <out.onnx>
"""
import sys, os
from onnxruntime.quantization.matmul_nbits_quantizer import (
    MatMulNBitsQuantizer, RTNWeightOnlyQuantConfig,
)

SRC, OUT = sys.argv[1], sys.argv[2]
os.makedirs(os.path.dirname(OUT) or ".", exist_ok=True)
print(f"[q4] quantizing {SRC} -> {OUT} (block=32, asymmetric, no float16)", flush=True)
q = MatMulNBitsQuantizer(SRC, block_size=32, is_symmetric=False, bits=4,
                         algo_config=RTNWeightOnlyQuantConfig())
q.process()
q.model.save_model_to_file(OUT, use_external_data_format=True)
data = OUT + "_data" if os.path.exists(OUT + "_data") else OUT + ".data"
sz = os.path.getsize(data) / 1e9 if os.path.exists(data) else 0
print(f"[q4] saved: graph {os.path.getsize(OUT)/1e6:.1f}MB + data {sz:.2f}GB", flush=True)
print("DONE", flush=True)
