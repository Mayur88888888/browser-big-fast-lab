#!/usr/bin/env python3
"""Build a true q4f16 from fp32 ONNX using onnxconverter_common (not the buggy
onnxruntime.transformers.float16, which inserted a dangling InsertedPrecisionFreeCast at a
layernorm → garbage). fp16 path keeps the embedding at 1.24GB (<2GB, so no >2GB chunk → no
int32 overflow on load) and KV fp16 (harness default).

Pipeline: fp32 --onnxconverter_common.float16--> fp16 --MatMulNBits(block32,asym)--> q4f16
Usage: python quantize_q4f16_v2.py <fp32.onnx> <out_dir>
"""
import sys, os, glob, shutil, onnx
from onnxconverter_common import float16
from onnxruntime.quantization.matmul_nbits_quantizer import (
    MatMulNBitsQuantizer, RTNWeightOnlyQuantConfig,
)

FP32, OUT = sys.argv[1], sys.argv[2]
onnx_dir = os.path.join(OUT, "onnx"); os.makedirs(onnx_dir, exist_ok=True)
fp16_model = os.path.join(onnx_dir, "model_fp16.onnx")
q4_model = os.path.join(onnx_dir, "model_q4f16.onnx")

print("[fp16] in-memory convert (onnxconverter_common, op_block_list=[] -> convert EVERYTHING)...", flush=True)
m = onnx.load(FP32)  # 32GB; convert in place
# op_block_list=[] converts layernorm/etc too — avoids the mixed-precision Cast type mismatch
# that keeping layernorm fp32 produced. Qwen3 runs fully-fp16 anyway (q4f16 = fp16 activations).
m16 = float16.convert_float_to_float16(m, keep_io_types=False, disable_shape_infer=True, op_block_list=[])
del m
onnx.save(m16, fp16_model, save_as_external_data=True, all_tensors_to_one_file=True,
          location="model_fp16.onnx_data", size_threshold=1024, convert_attribute=True)
del m16
print(f"[fp16] saved {fp16_model}", flush=True)

print("[q4] quantizing fp16 -> q4f16 (block=32, asymmetric)...", flush=True)
q = MatMulNBitsQuantizer(fp16_model, block_size=32, is_symmetric=False, bits=4,
                         algo_config=RTNWeightOnlyQuantConfig())
q.process()
q.model.save_model_to_file(q4_model, use_external_data_format=True)
print(f"[q4] saved {q4_model}", flush=True)

src_dir = os.path.dirname(os.path.dirname(FP32)) if os.path.basename(os.path.dirname(FP32)) == "onnx" else os.path.dirname(FP32)
for b in ("config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json",
          "vocab.json", "merges.txt", "special_tokens_map.json", "added_tokens.json"):
    for cand in (os.path.join(os.path.dirname(FP32), b), os.path.join(src_dir, b)):
        if os.path.exists(cand): shutil.copy(cand, OUT); break
data = q4_model + "_data" if os.path.exists(q4_model + "_data") else q4_model + ".data"
sz = os.path.getsize(data)/1e9 if os.path.exists(data) else 0
print(f"[size] q4f16 data {sz:.2f}GB", flush=True)
print("DONE", flush=True)
