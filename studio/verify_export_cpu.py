#!/usr/bin/env python3
"""Coherence-check a q4f16 decoder-with-past ONNX via onnxruntime CPU EP.

Isolates "is the EXPORT correct?" from any browser/WebGPU/wasm/chunking issue: if CPU
gives coherent text the weights are fine (and a garbage browser run is WebGPU-specific);
if CPU is also garbage, the fp32->fp16->q4 pipeline is broken.

Usage: python verify_export_cpu.py <model.onnx> <tokenizer_dir_or_id> [prompt]
"""
import sys, numpy as np, onnxruntime as ort
from transformers import AutoConfig, AutoTokenizer

MODEL, TOKID = sys.argv[1], sys.argv[2]
PROMPT = sys.argv[3] if len(sys.argv) > 3 else "What is the capital of France?"

cfg = AutoConfig.from_pretrained(TOKID)
L = cfg.num_hidden_layers
Hkv = getattr(cfg, "num_key_value_heads", cfg.num_attention_heads)
hd = getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)
print(f"arch: L={L} Hkv={Hkv} head_dim={hd} vocab={cfg.vocab_size}", flush=True)

tok = AutoTokenizer.from_pretrained(TOKID)
text = f"<|im_start|>user\n{PROMPT} /no_think<|im_end|>\n<|im_start|>assistant\n"
ids = tok(text, add_special_tokens=False).input_ids

print("loading session (CPU)...", flush=True)
sess = ort.InferenceSession(MODEL, providers=["CPUExecutionProvider"])
onames = [o.name for o in sess.get_outputs()]
li = onames.index("logits")

# auto-detect KV dtype from the model's input metadata (q4f16 -> fp16, q4-no-float16 -> fp32)
kv_type = {i.name: i.type for i in sess.get_inputs()}.get("past_key_values.0.key", "tensor(float16)")
kv_np = np.float16 if "float16" in kv_type else np.float32
print(f"KV dtype: {kv_type} -> {kv_np.__name__}", flush=True)
past = {f"past_key_values.{i}.{kv}": np.zeros((1, Hkv, 0, hd), kv_np)
        for i in range(L) for kv in ("key", "value")}
cur, posbase, out = ids, 0, []
eos = set(np.atleast_1d(cfg.eos_token_id).tolist()) if cfg.eos_token_id is not None else {151645}
for step in range(24):
    seq = len(cur); total = posbase + seq
    feeds = {"input_ids": np.array([cur], np.int64),
             "attention_mask": np.ones((1, total), np.int64),
             "position_ids": np.array([list(range(posbase, total))], np.int64), **past}
    o = sess.run(None, feeds)
    nxt = int(np.asarray(o[li], np.float32)[0, -1].argmax())
    if nxt in eos:
        print(f"[eos at step {step}]", flush=True); break
    out.append(nxt)
    past = {f"past_key_values.{i}.{kv}": o[onames.index(f"present.{i}.{kv}")]
            for i in range(L) for kv in ("key", "value")}
    posbase, cur = total, [nxt]

txt = tok.decode(out, skip_special_tokens=True)
print("OUTPUT:", repr(txt), flush=True)
print("VERDICT:", "COHERENT — export OK (browser garbage = WebGPU-specific)"
      if len(txt.split()) >= 3 and not set(txt.strip()) <= set("! ") else
      "GARBAGE — export/quantize pipeline is broken", flush=True)
