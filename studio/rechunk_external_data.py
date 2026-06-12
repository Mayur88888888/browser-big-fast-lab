#!/usr/bin/env python3
"""Re-split a single-file ONNX external-data blob into ≤MAXCHUNK numbered chunks.

Why: onnxruntime-web@1.26.0 (WebGPU EP, wasm32) sizes its wasm heap to the external-data
it stages, and wasm32 caps at 4 GB. A single 5.36 GB `.onnx.data` ⇒ the load dies with
`WebAssembly.Memory initial 85763 > 65536`. Splitting into ≤1.8 GB chunks lets ORT load
them incrementally (per-chunk wasm) and clear the wall — the difference between our failing
single-chunk 8B and indira's loading 3-chunk 5.3 GB on the same runtime.

Memory-light: reads ONLY the .onnx proto (~MB), then copies raw byte ranges from the source
data file into chunk files and rewrites each tensor's external-data offset. Never loads the
multi-GB weights into RAM (the laptop has ~700 MB free).

Usage: python rechunk_external_data.py <src.onnx> <src.onnx.data> <out_dir> [max_gb]
Writes <out_dir>/<basename>.onnx + <basename>.onnx_data[, _data_1, _data_2, ...]
"""
import onnx, os, sys
from onnx.external_data_helper import _get_all_tensors

SRC_ONNX, SRC_DATA, OUT_DIR = sys.argv[1], sys.argv[2], sys.argv[3]
MAXCHUNK = int(float(sys.argv[4]) * 1e9) if len(sys.argv) > 4 else int(1.8e9)
ALIGN = 64
BASE = os.path.basename(SRC_ONNX)                      # model_q4f16.onnx
os.makedirs(OUT_DIR, exist_ok=True)

m = onnx.load(SRC_ONNX, load_external_data=False)      # proto only — tiny

def ext(t):
    d = {kv.key: kv.value for kv in t.external_data}
    return d.get("location"), int(d.get("offset", "0") or 0), int(d.get("length", "0") or 0)

def chunk_name(i):
    return f"{BASE}_data" if i == 0 else f"{BASE}_data_{i}"

tensors = [t for t in _get_all_tensors(m)
           if t.HasField("data_location") and t.data_location == onnx.TensorProto.EXTERNAL]
print(f"external tensors: {len(tensors)}  max chunk: {MAXCHUNK/1e9:.2f}GB", flush=True)

src = open(SRC_DATA, "rb")
idx, off = 0, 0
out = open(os.path.join(OUT_DIR, chunk_name(0)), "wb")
for t in tensors:
    _, soff, length = ext(t)
    if off + length > MAXCHUNK and off > 0:           # start a new chunk at a tensor boundary
        out.close(); idx += 1; off = 0
        out = open(os.path.join(OUT_DIR, chunk_name(idx)), "wb")
    new_off = off
    src.seek(soff); remaining = length
    while remaining > 0:                              # stream-copy the tensor's bytes
        buf = src.read(min(remaining, 8 << 20)); out.write(buf); remaining -= len(buf)
    off += length
    pad = (-off) % ALIGN
    if pad: out.write(b"\0" * pad); off += pad        # keep tensors aligned
    del t.external_data[:]                            # rewrite the reference
    for k, v in (("location", chunk_name(idx)), ("offset", str(new_off)), ("length", str(length))):
        e = t.external_data.add(); e.key, e.value = k, v
out.close(); src.close()

onnx.save(m, os.path.join(OUT_DIR, BASE))             # tensors stay external → writes graph only
print(f"chunks: {idx + 1}", flush=True)
for i in range(idx + 1):
    p = os.path.join(OUT_DIR, chunk_name(i))
    print(f"  {chunk_name(i)}: {os.path.getsize(p)/1e9:.2f} GB", flush=True)
print("DONE", flush=True)
