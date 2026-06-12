#!/usr/bin/env python3
"""Stage 2 of the Qwen3-8B q4f16 export (runs on the 64GB Studio).

Takes optimum's fp32 ONNX export (decoder-with-past, single graph) and produces an
onnx-community-style q4f16 build that the raw-ORT harness (web/run-one-ort.html) loads:

    <out>/config.json + tokenizer files
    <out>/onnx/model_fp16.onnx   (+ model_fp16.onnx_data)
    <out>/onnx/model_q4f16.onnx  (+ model_q4f16.onnx_data)

Pipeline (correctness-first ordering, matches how onnx-community builds q4f16):
    fp32  --convert_float_to_float16-->  fp16  --MatMulNBits(block32,asym)-->  q4f16

Memory: float16 conversion replaces tensors in place (peak ~= the 33GB fp32 model);
quantization reloads the 16GB fp16 from disk. Both fit the 64GB box with headroom.

Usage:  .venv/bin/python quantize_q4f16.py <fp32_export_dir> <out_dir>
"""
import sys, os, glob, shutil, onnx
from onnxruntime.transformers.float16 import convert_float_to_float16
from onnxruntime.quantization.matmul_nbits_quantizer import (
    MatMulNBitsQuantizer, RTNWeightOnlyQuantConfig,
)

BLOCK_SIZE = 32          # matches onnx-community Qwen3 q4f16 (webgpu MatMulNBits)
IS_SYMMETRIC = False     # asymmetric (zero-points) is WebGPU-OK for *dense* MatMulNBits


def io_sig(path):
    m = onnx.load(path, load_external_data=False)
    return ([i.name for i in m.graph.input], [o.name for o in m.graph.output])


def main(fp32_dir, out_dir):
    onnx_dir = os.path.join(out_dir, "onnx")
    os.makedirs(onnx_dir, exist_ok=True)
    fp32_model = os.path.join(fp32_dir, "model.onnx")
    fp16_model = os.path.join(onnx_dir, "model_fp16.onnx")
    q4_model = os.path.join(onnx_dir, "model_q4f16.onnx")

    ins, outs = io_sig(fp32_model)
    print(f"[io] inputs ({len(ins)}): {ins}", flush=True)
    print(f"[io] outputs ({len(outs)}): {outs}", flush=True)
    # sanity: the harness feeds these exact names
    need_in = {"input_ids", "attention_mask", "position_ids"}
    assert need_in <= set(ins), f"missing harness inputs: {need_in - set(ins)}"
    assert any(n.startswith("past_key_values.") for n in ins), "no past_key_values.* inputs"
    assert "logits" in outs and any(n.startswith("present.") for n in outs), "no logits/present.* outputs"
    print("[io] OK — decoder-with-past graph matches the run-one-ort.html contract", flush=True)

    print("[fp16] loading fp32 (with external data) + converting in place...", flush=True)
    m = onnx.load(fp32_model)  # pulls external data into memory (~33GB)
    m16 = convert_float_to_float16(m, keep_io_types=False, disable_shape_infer=True)
    del m
    onnx.save(
        m16, fp16_model, save_as_external_data=True, all_tensors_to_one_file=True,
        location="model_fp16.onnx_data", size_threshold=1024, convert_attribute=True,
    )
    del m16
    print(f"[fp16] saved {fp16_model}", flush=True)

    print(f"[q4] quantizing fp16 -> q4f16 (block={BLOCK_SIZE}, symmetric={IS_SYMMETRIC})...", flush=True)
    quant = MatMulNBitsQuantizer(
        fp16_model, block_size=BLOCK_SIZE, is_symmetric=IS_SYMMETRIC, bits=4,
        algo_config=RTNWeightOnlyQuantConfig(),
    )
    quant.process()
    quant.model.save_model_to_file(q4_model, use_external_data_format=True)
    print(f"[q4] saved {q4_model}", flush=True)

    # tokenizer + config alongside the onnx/ dir (resolve() in the harness reads repo root)
    keep = ("config.json", "generation_config.json", "tokenizer.json",
            "tokenizer_config.json", "vocab.json", "merges.txt",
            "special_tokens_map.json", "added_tokens.json", "chat_template.jinja")
    for f in glob.glob(os.path.join(fp32_dir, "*")):
        b = os.path.basename(f)
        if b in keep:
            shutil.copy(f, out_dir)
    print("[copy] config + tokenizer ->", out_dir, flush=True)

    for label, p in (("fp16", fp16_model), ("q4f16", q4_model)):
        data = p + "_data"
        sz = os.path.getsize(data) / 1e9 if os.path.exists(data) else 0
        # ORT may name external data <model>.data instead of <model>_data
        if not sz and os.path.exists(p + ".data"):
            sz = os.path.getsize(p + ".data") / 1e9
        print(f"[size] {label}: graph {os.path.getsize(p)/1e6:.1f}MB + data {sz:.2f}GB", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
