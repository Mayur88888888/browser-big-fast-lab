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
| AR baseline (ladder) | Qwen3-4B | q4f16 | — | — | — | — | — | — | n/a | — |
| AR baseline (ladder) | Qwen3-8B | q4f16 | — | — | — | — | — | — | n/a | — |
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
