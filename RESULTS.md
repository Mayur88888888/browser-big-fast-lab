# Results — browser-big-fast lab

The one comparison table, plus the raw run log it distills from. Process notes live in `plan/`; this file is the deliverable.

## Environment

| | |
|---|---|
| GPU / backend | Apple **Metal-3** WebGPU (adapter vendor `apple`), `shader-f16` supported |
| WebGPU limits | `maxBufferSize` & `maxStorageBufferBindingSize` ≈ **4 GB** (4294967292 B) — the hard single-buffer ceiling |
| Renderer memory | `jsHeapSizeLimit` = 4 GB; holds **4.33 GB cumulative** ArrayBuffers + 1.5 GB single fine; single **≥2 GB** alloc hard-`RangeError`s (V8 2³¹ cap) |
| Runtime | transformers.js `@huggingface/transformers@4` (bundled ORT-web), CDN ESM |
| Harness | `web/run-one.html` (one model / one page-load), static server `:8131`. Verified in the Claude preview chromium (real Metal-3, not SwiftShader). |
| Measurement caveats | absolute tok/s drifts down over a long GPU session → measure per fresh page-load; cold-load includes HF download + shader compile |

## ⚠ The loading wall (central obstacle) — transformers.js path caps at <2 GB

Qwen3-1.7B q4f16 = a **1.33 GB single embedded `.onnx`** (no external-data chunks; every 1.7B variant is ≥1.33 GB). Loading it via the transformers.js pipeline → **`std::bad_alloc` at session creation**, even as the first model in a fresh renderer. Not a missing file, not accumulated memory, not the GPU (~4 GB buffers free), not the renderer (holds 4 GB+). It's a **C++ `std::bad_alloc` inside the ORT-web wasm heap**: the off-the-shelf pipeline loads embedded weights *through* wasm, and the load+working-set crosses a ~2 GB wasm allocation limit between 0.6B (~0.4 GB ✅) and 1.7B (1.33 GB ❌).

**Implication:** the transformers.js-pipeline harness is fine for the ≤1B rung but **cannot reach 1.7B+ — nor any of T1/T2/T3's 5–6 GB models.** The proven route past this is the **raw ORT-web external-data → GPU load path** (weights stream into GPU buffers, bypassing the wasm heap), which is how indira ran 5.3 GB and kohra ran 3 GB in-browser. Building/porting that loader is the next keystone. (A newer ORT-web build with a larger wasm `MAXIMUM_MEMORY` — the handoff's T0 lever — may also lift the embedded-path cap; the external-data route is the safer bet.)

### ✅ Resolved — raw `onnxruntime-web@1.26.0` clears it (no re-export needed for Qwen)

It was the **older bundled ORT version**, not the embedded weights. Raw ORT@1.26.0 (`web/ort-load-probe.html`) creates a session for the same 1.33 GB embedded q4f16 fine (59.7 s, mostly download). `web/run-one-ort.html` then drives the onnx-community **decoder-with-past** graph directly (prefill → greedy KV-cache decode; I/O `input_ids, attention_mask, position_ids, past_key_values.{0..L-1}.{key,value}` → `logits, present.*`) and Qwen3-1.7B q4f16 **generates coherently** (15.2 tok/s, TTFT 0.71 s). So the two-backend harness is: **transformers.js for ≤1B convenience, raw ORT@1.26.0 for everything bigger.** External-data re-export (Studio) is deferred to genuinely-bigger / non-onnx-community models. Open ceiling test: 8B q4 (~4.5 GB embedded) vs ORT@1.26.0's wasm cap at session-create.

## T0 — q4 on WebGPU (keystone): **WORKS** ✅

**Verdict:** off-the-shelf onnx-community q4 (`q4f16`, dense `MatMulNBits`) computes **correctly** on this Apple Metal-3 GPU via transformers.js@4. kohra's own earlier q4 export miscomputed here (wrong argmax) — that was a **kohra-export-recipe bug, not a platform wall**. The handoff's pessimistic premise ("q4 broken ⇒ fp16 ceiling") does **not** bind on the onnx-community path. Hand-rolled exports must match that recipe (symmetric where ORT-web requires it). Full logit-diff-vs-CPU harness deferred (only needed to debug a specific bad export).

## Comparison table (end-of-lab target)

Filling incrementally. `—` = not yet measured. TTFT marked ⚠ until the prompt-put probe bug is fixed.

| Mode | Model | dtype | TTFT | tok/s | cold load | mem peak | quality | ToolCall-15 | infill | in-browser today? |
|---|---|---|---|---|---|---|---|---|---|---|
| AR baseline (ladder) | Qwen3-0.6B | q4f16 | 0.20 | **50.6** | ~30 s | JS-heap ~70 MB¹ | coherent | — | n/a | ✅ (transformers.js) |
| AR baseline (ladder) | Qwen3-1.7B | q4f16 | 0.71 | **15.2** (17.4 decode) | ~60 s | —¹ | coherent | — | n/a | ✅ (raw ORT@1.26.0 loader) |
| AR baseline (ladder) | Qwen3-4B | q4f16 | 1.0 | **9.2** (10.5 decode) | ~122 s | —¹ | coherent | — | n/a | ✅ (raw ORT, 2.77 GB / 2 chunks) |
| AR baseline (ladder) | Qwen3-8B | q4f16 | — | — | — | — | — | — | n/a | ⛔ needs conversion (onnx-community 8B is ORT-GenAI int4 / 6GB-single-file, not flat q4f16) |
| AR baseline | Gemma4 E4B(-QAT) | — | — | — | — | — | — | — | n/a | — |
| AR+MTP | E4B-QAT+drafter | — | — | — | — | — | — | — | n/a | — |
| Sparse MoE AR | LFM2.5-8B-A1B | — | — | — | — | — | — | — | n/a | — |
| Diffusion MoE | LLaDA-MoE (base+TD) | — | — | — | — | — | — | — | ✓ | — |

¹ JS-heap only (`performance.memory`), **not** GPU memory — real GPU high-water still unmeasured (known gap; weights ≈ model file size + KV cache is the working proxy until measured).

## Raw run log

- **2026-06-12** · Qwen3-0.6B-ONNX · q4f16 · prompt "The capital of France is" (ChatML, /no_think) · max 48 → `"France is Paris."` · coherent ✅ · 4 tok / 0.46 s (too short for reliable tok/s) · cold load 30.4 s · the decisive T0 correctness data point.
- **2026-06-12** · Qwen3-0.6B-ONNX · q4f16 · prompt "Write a short paragraph explaining why the sky is blue." · max 96 → coherent 70-tok paragraph · **48.2 tok/s** · gen 1.45 s · cold load 28 s · TTFT ⚠0 (probe bug, since fixed).
- **2026-06-12** · Qwen3-0.6B-ONNX · q4f16 · same prompt · max 96 (after TTFT fix) → coherent · **50.6 tok/s** · **TTFT 0.20 s** · cold load ~40 s. TTFT probe validated (skip the prompt `put`).
- **2026-06-12** · Qwen3-1.7B-ONNX · q4f16 (1.33 GB embedded) · **FAIL — `Can't create a session … std::bad_alloc`** at session creation, both as 4th model in a session AND as the first model in a fresh renderer. → the wasm load wall (see section above). dtype-switch won't help (all 1.7B variants ≥1.33 GB).
- **2026-06-12** · memory probe (fresh renderer): cumulative ArrayBuffers 0.5+1+1.33+1.5 GB = 4.33 GB OK; single 2 GB → `RangeError`; `jsHeapSizeLimit` 4 GB; `deviceMemory` 8. → renderer/GPU not the bottleneck; wasm heap is.
- **2026-06-12** · `ort-load-probe.html` · Qwen3-1.7B q4f16 · raw `onnxruntime-web@1.26.0` → **session CREATED ✅** in 59.7 s. Same embedded file that bad_alloc'd under transformers.js. I/O = decoder-with-past, 28 layers. → the wall was the ORT version, not the model.
- **2026-06-12** · `run-one-ort.html` (raw ORT@1.26.0 AR KV-cache loop) · Qwen3-1.7B q4f16 · max 80, /no_think → **coherent** ("The sky appears blue because of the way sunlight interacts…") · **15.2 tok/s** (17.4 decode) · **TTFT 0.71 s** · cold load ~60 s · arch auto-read 28L/8kv/headdim128. Empty `<think></think>` confirms /no_think works; stripped in post.
- **2026-06-12** · Qwen3-4B q4f16 · **first attempt FAILED** — `Failed to load external data file "model_q4f16.onnx_data_1" … not found in preloaded files`. 4B q4f16 is **multi-chunk** external data (`.onnx_data` 2.10 GB + `.onnx_data_1` 677 MB); the loader only registered chunk 0. → fixed `run-one-ort.html` to enumerate + register all chunks (`_data`, `_data_1`, …).
- **2026-06-12** · Qwen3-4B q4f16 (2.77 GB / 2 chunks) · after multi-chunk fix → **coherent** ("…shorter wavelengths of light, like blue, are scattered more efficiently…") · **9.2 tok/s** (10.5 decode) · **TTFT 1.0 s** · cold load 122 s · 36L/8kv/headdim128. Loader generalizes to multi-chunk big models (the shape T1/T2/T3 need).
- **2026-06-12** · Qwen3-8B · **BLOCKED on packaging, not memory** — `AutoTokenizer.from_pretrained` threw `tokenizer_class of undefined` (tokenizer is in a subfolder). `onnx-community/Qwen3-8B-ONNX` is the **ORT-GenAI** layout: `onnxruntime/webgpu/webgpu-int4-kld-block-32/{model.onnx, model.onnx.data(6.0 GB), tokenizer*}` — int4-kld (not q4f16), a **single 6 GB** data file (trips the 2 GB ArrayBuffer cap), GenAI-format graph. Not a drop-in. The real q4f16 ceiling test needs a **conversion** (Studio: export Qwen3-8B → q4f16 MatMulNBits ONNX, multi-chunk external data, matching the smaller rungs). The memory ceiling question (does ~4.5 GB load) remains OPEN.

### Dense AR ladder so far (q4f16, Metal-3 WebGPU)

| rung | tok/s | decode tok/s | TTFT | cold load | backend |
|---|---|---|---|---|---|
| 0.6B | 50.6 | — | 0.20 s | ~30 s | transformers.js |
| 1.7B | 15.2 | 17.4 | 0.71 s | ~60 s | raw ORT@1.26.0 |
| 4B | 9.2 | 10.5 | 1.0 s | ~122 s | raw ORT@1.26.0 (2.77 GB) |
| 8B | — | — | — | — | needs q4f16 conversion |

Clean memory-bandwidth scaling: bigger = slower (50→15→9 tok/s), TTFT and cold-load climb with size. This is the AR-baseline curve the MoE/MTP/diffusion modes get compared against.

## T6 — Custom-WGSL kernel backend (the speed frontier) — NEW, not yet measured

A second backend arrived from outside this lab: **hand-written WGSL inference engines**
(webml-community / Xenova lineage), validated in LocalMind at **Gemma 4 E2B ~250 tok/s**
and **LFM2.5 230M ~1,000 tok/s** on this same Metal-3 — i.e. **5–30× the ORT-web ladder
above**. This obsoletes the handoff's "do not build custom kernels" line (decided when q4
looked broken; the new move is to **fork a proven Apache-2.0 engine**, not build one). The
fork lives in [`custom-kernels/`](custom-kernels/) (branch `qwen3-spike`, see its `PHASE0.md`).

**The decisive tradeoff (memory):**
- The forkable engine (`tylerstraub/gemma4-webgpu`) is **F16-everywhere** — weights
  dequantized to F16 on GPU. A 4B model ≈ **8 GB GPU** (> the lab's 8B-q4 at 5.36 GB!).
  → **the speed frontier at ≤~3B**, *not* a "big" play in the sidecar slice.
- The LFM2 engine keeps weights **in-shader q4/q8** (memory-efficient, like ORT's
  `MatMulNBits` but hand-written + faster) — but it's **minified-only** (no forkable source).
- So "**big AND fast** custom kernels" = port in-shader-q4 into the forkable framework
  (LFM2 bundle + ORT `MatMulNBits` as references). The headline roadmap item if speed lands.

**Decider experiment (run first):** custom-kernel **Qwen3-1.7B / 4B tok/s vs the ORT
baselines (15.2 / 9.2)** on this machine. 1.7B fits F16 cleanly; 4B may exceed the slice
(expected caveat row). One number decides whether T6 is worth pursuing.

| Mode | Model | dtype | tok/s | mem peak | in-browser today? |
|---|---|---|---|---|---|
| Custom-WGSL (T6) | Qwen3-1.7B | F16 (custom) | — *(vs ORT 15.2)* | ~3.4 GB F16 | spike — `custom-kernels/`, not yet wired |
| Custom-WGSL (T6) | Qwen3-4B | F16 (custom) | — *(vs ORT 9.2)* | ~8 GB F16 ⚠ | spike — may exceed slice |
| Custom-WGSL (T6) | Gemma 4 E2B | GGUF→F16 | ~250 (LocalMind) | ~4 GB F16 | ✅ the fork's native model |
