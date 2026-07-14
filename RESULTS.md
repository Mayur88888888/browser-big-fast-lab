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

## T6 — Custom-WGSL kernel backend (the speed frontier) — MEASURED 2026-06-27

A second backend: **hand-written WGSL inference engine**, raw WebGPU, **no ORT, no wasm**.
Forked `tylerstraub/gemma4-webgpu` (Apache-2.0, webml-community/Xenova lineage) into
[`custom-kernels/`](custom-kernels/) (branch `qwen3-spike`) and **generalised it to Qwen3**
(see [the port commit](custom-kernels/) — an `arch: gemma4|qwen3` discriminator + 8 deltas).
Obsoletes the handoff's "do not build custom kernels" line; **rewriting/extending kernels is
on the agenda.**

### Environment (differs from the ORT rows above)

| | |
|---|---|
| Backend | Pure **TypeScript + raw WebGPU (WGSL)**. No onnxruntime, **no wasm heap** — GGUF weights stream over HTTP Range → CPU-dequant to F16 → straight into GPU storage buffers. |
| Weight layout | **F16-everywhere on GPU** (any source quant dequantized once at load). One matmul path; memory-heavy (a 4B model ≈ 8 GB GPU). |
| Tuning | `apple-m-series` profile auto-selected (engine also ships `nvidia-blackwell`, `generic`). |
| Harness | `custom-kernels/` workbench (`/workbench.html`, `window.lab`), Vite dev `127.0.0.1:5175`. Driven via chrome-devtools MCP `evaluate_script`. |
| Metric | `lab.bench(prompt,{maxTokens,runs})` → **pure-decode tok/s** = `(tokens-1)/(coreDecodeMs/1000)`, excludes TTFT and caller-body time (methodology-matched to the engine's `runBench`). |
| Configs | **short** = prompt "Hello, how are you?", maxTokens 64. **long** = default raven/crow prompt, maxTokens 200. Distinct workloads — `attnScore` is O(seqLen)/token, so long ≠ short. Warm one run before measuring. |
| Same machine | Identical Metal-3 laptop as the ORT rows → **apples-to-apples on hardware**, different runtime. |

### Result — coherent, ~2.5× the ORT-web ladder (NOT the 5–30× the handoff claimed)

| Model | dtype | tok/s (long / short) | TTFT | vs ORT | cold load | mem (F16 GPU) | coherent? |
|---|---|---|---|---|---|---|---|
| Qwen3-1.7B | GGUF q4_k_m → F16 | **43.3 / 35.6** | ~0.25–0.46 s | **2.4–2.9×** (ORT 15.2) | 65 s (1.1 GB) | ~3.4 GB | ✅ ("…is **Paris**.") |
| Qwen3-4B | GGUF q4_k_m → F16 | **~23 / 7.7¹** | ~0.9 s | **2.5×** (ORT 9.2) | 125 s (2.5 GB) | ~8 GB **(fits 24 GB laptop, no OOM)** | ✅ ("…corvids, part of the family…") |
| Gemma 4 E2B (native) | GGUF q4_k_m → F16 | **26.3 / 25.4** | ~0.3 s | — (baseline proxy) | 173 s (3.1 GB) | ~4 GB | ✅ |

¹ 4B short cfg = 7.7 is a prefill-amortization artifact; GPU-timestamp profile (43.6 ms/token = 22.9 tok/s) confirms the long number is the true steady-state.

**Per-kernel forward profile** (GPU timestamps): 1.7B forward = 22.9 ms/tok, 4B = 43.6 ms/tok. Both dominated by **core matmuls Qwen3 shares with ORT** — FFN gate/up 40–44%, FFN down 14–17%, QKV 14–15%, lmHead 9–15%. So the ~2.5× is the **engine** (F16-everywhere, fused norm+rope, one matmul path), not Gemma-overhead removal.

**Why the handoff said "5–30×":** that was a **Blackwell-NVIDIA** measurement mis-attributed to this Apple laptop (the engine's `nvidia-blackwell` tuning profile is the tell). Benched here, Gemma E2B = 25.4 tok/s — 10× below the claimed ~250. **~2.5× is the true Apple-Metal-3 frontier.**

**Correctness: `crossLabDiff` GREEN** (2026-06-27). Qwen3-1.7B q4_k_m engine vs HF `Qwen/Qwen3-1.7B` bf16 reference, same tokens: embed 0.99990, per-layer all ≥0.985 (00 .99912 / 06 .99230 / 13 .98555 / 20 .99200 / 26 .99702), final 0.99635, **logits 0.99636, argmax=Paris on both**. Smooth monotonic F16 drift, no kernel-bug cliff → numerically equivalent within q4+F16 tolerance. Reference: `custom-kernels/reference/pytorch/qwen3_smoke.py` + committed `public/ref/qwen3_smoke.npz`. The decisive port bug found en route — Gemma hardcodes attention `scaling=1.0`; Qwen3 needs `1/√head_dim`, else softmax over-sharpens → degenerate repetition. **Caveat (tokenizer layer, not forward pass):** the engine's `tokenizer.encode` drops GPT-2 `Ġ` space markers and the chat template is still Gemma-specific — generation survives but real-prompt quality is degraded; tracked for the generalised-engine work.

**The memory tradeoff:** F16-everywhere = fast but heavy. 4B = ~8 GB F16 (fits the 24 GB laptop). 8B F16 ≈ 16 GB → impractical → the **"big AND fits" lever is in-shader q4/q8** (keep weights quantized in GPU like ORT `MatMulNBits` / the LFM2 minified engine). **RESOLVED 2026-06-28: in-shader Q4_K runs Qwen3-8B in a browser tab** (4.34 GB layer weights vs 13.9 GB F16) — see the Q4_K + 8B section below. Originally the trade was speed (scalar GEMV) for reach — **now closed**: the MR4 Q4_K kernel (2026-06-28) makes 4-bit the *fastest* path (1.58× F16 same-session on 1.7B), so the 4–8 GB-class sweet spot is in hand at no speed cost. See the Q4_K speed table below.

### In-shader Q8_0 — the memory lever (MEASURED 2026-06-27)

`weightQuant: 'q8'` keeps the 7 big per-layer matmul weights **quantized in GPU**
(symmetric int8 + one f16 scale per 32-block, re-derived from source at load) and
dequantizes them inside the GEMV loop, instead of F16-everywhere. lmHead/embeddings/
norms stay F16. Shader `matmul_q8.wgsl`; committed `903d8f6`.

| Qwen3-1.7B (q4_k_m source) | F16 | **Q8 in-shader** |
|---|---|---|
| crossLabDiff logits cosine | 0.99636 | **0.99666** (argmax=Paris ✓) |
| tps (short / long) | 35.6 / 43.3 | 34.7 / 42.0 (= F16) |
| layer-weight GPU mem | 2.82 GB | **1.50 GB** (1.88×) |
| total weight GPU mem | 3.44 GB | **2.12 GB** (1.62×, diluted by tied F16 embed) |

**Near-lossless** (Q8 has headroom over the Q4_K source) and **no speed cost** (GEMV is
bandwidth-bound → smaller reads offset the unpack ALU). Proves the in-shader-dequant
architecture end-to-end (load requant → shader → bind groups → dispatch → crossLabDiff).
**Unblocks 8B on a 24 GB laptop:** ~16 GB F16 → ~9 GB Q8. **Q4_K in-shader (store raw
super-blocks, 4× vs F16) is the next increment** → the comfortable 4–8B sweet spot.

### In-shader Q4_K (4-bit) + the 8B-in-browser demo (MEASURED 2026-06-28)

`weightQuant: 'q4k'` keeps the 7 big per-layer matmul weights at **~5 bits/value in GPU**
(4-bit block-affine + an f16 scale/min per 32-elem sub-block — GGUF Q4_K's *structure*
but f16 sub-scales instead of 6-bit-packed, so a Q4_K source round-trips near-losslessly
and only the few Q6_K tensors take a real 6→4 down-quant). Dequant happens in the GEMV
loop. Shader `matmul_q4k.wgsl` (shares the matmulQ8 bind layout). lmHead/embeddings/norms
stay F16. Commit `9cd1f2f` (custom-kernels).

| Qwen3-1.7B (q4_k_m source) | F16 | Q8 in-shader | **Q4_K in-shader** |
|---|---|---|---|
| crossLabDiff logits cosine | 0.99636 | 0.99666 | **0.906** (argmax=ĠParis ✓) |
| crossLabDiff per-layer cosine | ≥0.985 | ≥0.985 | **0.92–0.997** (smooth, no cliff) |
| coherent greedy decode | ✓ | ✓ | ✓ ("…Paris. The capital of Italy is Rome…") |
| tps — scalar kernel (initial) | 35.6 | 34.7 | 10.2 (1-row GEMV) |
| tps — **MR4 kernel (2026-06-28)** | — | — | **40.6** (1.58× F16 same-session) |
| layer-weight GPU mem | 2.82 GB | 1.50 GB | **~0.88 GB** (3.2×) |

The lower logit cosine (0.906 vs F16's 0.996) is **expected 4-bit degradation**, not a bug:
the curve is smooth (no kernel-bug cliff), argmax is preserved (=HF reference), and greedy
generation is fully coherent and factually correct.

**Speed — the MR4 kernel closed (then reversed) the gap.** The initial Q4_K was a scalar
1-row GEMV (no fast path) → slow. `matmul_q4k_mr.wgsl` produces 4 output rows/workgroup
(input read once, dequant 4 rows in-loop, shared tree-reduce), applied to **all** q4k
matmuls — where F16 only MR's the FFN. Bit-identical output (Node check + crossLabDiff
reproduced the scalar numbers exactly). Same-session, same-contention Qwen3-1.7B:

| | short (64) | long (200) |
|---|---|---|
| F16 | 25.7 | 24.4 |
| **Q4_K-MR** | **40.6** | **40.4** |

**4-bit is now the *fastest* path AND 3.2× smaller** — MR covers all matmuls (vs F16's
FFN-only) and 4-bit moves ¼ the weight bytes.

**Measurement caveat (contention):** absolute tps depends on machine load — single-token
decode is latency-bound and this is a shared Metal GPU (F16 was 35.6 a less-busy session,
25.7 here). So cite the **same-session ratio**, not the absolute: Q4_K-MR is **1.58× F16**
measured back-to-back. (The earlier "scalar q4k 10.2 vs F16 35.6" was also same-session;
the ~4× scalar→MR absolute jump spans different-contention periods, so the F16-relative
numbers are the trustworthy ones.)

**The 8B-in-browser demo — the headline F16 couldn't reach:**

| Qwen3-8B Q4_K_M (in-shader q4k) | value |
|---|---|
| loads + runs in a browser tab | **✓** (Apple Metal-3, 24 GB laptop) |
| first token after "The capital of France is" | **ĠParis** ✓ |
| coherent generation | ✓ "…Rome. …Germany is Berlin. …Spain is Madrid. …Belgium is Brussels." |
| tps (short bench) | **5.2** (3.2 cold greedy) |
| load time | 432 s (4.7 GB download-bound) |
| layer-weight GPU mem | **4.34 GB** (vs **13.9 GB** F16) — 3.2× |
| total weight GPU mem | ~6.8 GB (4.34 q4k + 1.24 embed + 1.24 lm_head, both F16) |

8B F16 layer weights alone (~13.9 GB) plus a 2.49 GB F32 embed-dequant intermediate make
the F16 path impractical; in-shader Q4_K + a **chunked embed/lm_head decode** (16M-elem
chunks → no >2 GB F32 intermediate) load and run it. This is the 4–8 GB-class sweet spot.

### Custom-WGSL raw run log (what we did to get there)

- **2026-06-27** · Gemma 4 E2B (engine-native) · GGUF q4_k_m→F16 · `lab.bench('Hello…',64,runs3)` + default/200 · **25.4 short / 26.3 long tok/s** (runs 25.37–25.43, rock-stable) · load 173 s (3.1 GB, PLE stream dominates). *Purpose: same-machine proxy decider before porting — revealed the handoff's ~250 was Blackwell, not Apple.* Profile: forward 37.3 ms/tok, FFN gate/up 47%, lmHead 13.8%, PLE ~6–7%.
- **2026-06-27** · Qwen3-1.7B · GGUF q4_k_m→F16 (Unsloth) · first run after port → **degenerate** ("is the model of the model…") at 44.9 tok/s → diagnosed over-sharp softmax → **fixed attention scaling to 1/√head_dim** → re-ran **coherent** ("The capital of France is Paris."). Bench: **43.3 long / 35.6 short tok/s**, TTFT 0.25–0.46 s, tied embeddings, load 65 s.
- **2026-06-27** · Qwen3-4B · GGUF q4_k_m→F16 (Unsloth) · added untied-LM-head plumbing (this GGUF actually ties) · loaded **~8 GB F16 with no OOM** on the 24 GB laptop · **coherent** ("…Both are corvids, part of the family…") · **~23 tok/s long** (GPU-profile 43.6 ms/tok agrees) · TTFT ~0.9 s · load 125 s (2.5 GB).
- **2026-06-27** · Qwen3-1.7B · **`weightQuant:'q8'`** (in-shader Q8_0) · loaded clean, **coherent** ("…is Paris. Now, let's create a simple program…") · crossLabDiff logits **0.99666**, argmax=Paris (= F16 quality) · tps 34.7/42.0 (= F16) · weight mem 3.44→2.12 GB. *Purpose: prove the in-shader memory lever. Result: correct + same speed + ~1.9× smaller layer weights.* The decisive design choice — store int8+f16-scale per 32-block, dequant in the GEMV loop; mr4 fast-path disabled for q8 (scalar GEMV).
- **2026-06-28** · Qwen3-1.7B · **`weightQuant:'q4k'`** (in-shader 4-bit) · crossLabDiff smooth (embed 0.9999, per-layer 0.92–0.997, logits **0.906, argmax=ĠParis**) · raw-token greedy decode **coherent** ("ĠParis. The capital of Italy is Rome. …Japan is Tokyo. …Korea is Seoul. …China is Beijing.") · tps **10.2 short** · layer-weight mem ~0.88 GB (3.2× vs F16) · load 71 s. *Purpose: the deeper 4-bit memory lever for the 4–8B class. Result: correct (4-bit-expected quality, no cliff) + coherent + ~3.2× smaller.* **Bug fixed mid-run:** `meta` is a WGSL reserved keyword → the shader silently failed to compile → layer-0 cosine 0.004 garbage; renaming the binding to `qmeta` produced the green sweep. Decode-quality method: argmax token (=ĠParis) is the decisive signal when 4-bit logit cosine drops; crossLabDiff *smoothness* (vs a cliff) separates quant error from a kernel bug.
- **2026-06-28** · **Qwen3-8B Q4_K_M** · **`weightQuant:'q4k'`** · **loaded + ran in a browser tab** (the payoff F16 couldn't reach) · 36 layers, untied lm_head · first token after "The capital of France is" = **ĠParis**, **coherent** ("…Rome. …Germany is Berlin. …Spain is Madrid. …Belgium is Brussels.") · tps **5.2 short / 3.2 cold** · layer-weight mem **4.34 GB** (vs 13.9 GB F16) · total ~6.8 GB on the 24 GB laptop · load 432 s (4.7 GB, download-bound). *Purpose: the headline — big-AND-fits. Result: 8B runs in-browser at 4-bit.* **Enabler:** `dequantToF16Chunked` — the 8B token_embd (622M elems) needs a 2.49 GB F32 intermediate in one shot (blows V8's alloc ceiling); decoding in 16M-elem chunks into the single 1.24 GB F16 output dodges it. Embeddings/lm_head stay F16.
- **2026-06-28** · Qwen3-1.7B · **`matmul_q4k_mr.wgsl`** (the speed half) · multi-row Q4_K matmul (R=4 rows/wg, mirrors `matmulQuantMR4`) applied to **all** q4k matmuls via `dispatchMatmulRows` (ceil(M/4)). Correctness **bit-identical** to the scalar kernel — Node check 17920 elems maxDiff=0; in-browser crossLabDiff reproduced the scalar sweep exactly (logits 0.90593, L13 0.92092, argmax=ĠParis, coherent). Bench (same session/contention): **Q4_K-MR 40.6 short / 40.4 long vs F16 25.7 / 24.4 → 1.58×/1.65× F16**. *Purpose: recover the speed the scalar GEMV left on the table. Result: 4-bit is now the **fastest** path AND 3.2× smaller.* Why faster than F16: MR covers every q4k matmul (F16 MR's FFN only) and 4-bit moves ¼ the weight bytes. Method note: predicted 1.5–2.2× (overhead-bound classification correct); the larger realized gain = MR-everywhere + 4-bit bandwidth. Used the **same-session F16 anchor** (not the cross-session scalar 10.2) as the contention-clean comparison.
- **2026-06-28** · **MR for F16 + Q8** (`matmul_q8_mr.wgsl`; F16 attention routed through clamped `matmulQuantMR4`; unified dispatch behind `matmulRowsPerWg`). Every mode now does 4 rows/workgroup for **all** layer matmuls (F16 previously MR'd only the FFN; Q8 was fully scalar). Correctness **bit-exact-preserved** (Qwen3-1.7B crossLabDiff logits): **Q8 0.99667** (=prior 0.99666), **F16 0.99636** (=prior baseline), both coherent. *Purpose: widen the lead over transformers.js@4 across all modes. Result: every quant benefits; q4k remains the fastest+smallest.* **Bug caught by the Gemma regression check:** the tokenizer was mis-detecting arch — keyed byte-level-BPE on "merges present", but Gemma 4 GGUFs carry merges yet are SentencePiece (vocab uses ▁), so Gemma got byte-level-tokenized → garbage. Fixed to key on `tokenizer.ggml.model==='gpt2'`; Gemma E2B back to spm + coherent ("…Paris."). The matmul change itself never touched Gemma's numerics (Qwen3 f16 bit-exact proved the shared kernel).
- **2026-06-28** · **Qwen3-8B Q4_K_M re-measured with the MR kernel** · coherent (" Paris. The capital of Italy is Rome. …Germany is Berlin. …Spain is…") · **6.76 tok/s short / 5.66 cold** vs the prior scalar **5.2** → ~1.3× (smaller than 1.7B's gain — 8B is more bandwidth-bound, and the machine was heavily contended this run). Load 403 s. Confirms MR helps the 8B class too; 8B remains the ours-only headline (no flat-q4f16 ONNX exists for transformers.js).

## T7 — DeepSeek-OCR family in-browser: Unlimited-OCR spike (RECON 2026-07-14)

**Target:** [baidu/Unlimited-OCR](https://huggingface.co/baidu/Unlimited-OCR) (released 2026-06-22, **MIT license**) — DeepSeek-OCR's DeepEncoder + DeepSeek-V2-MoE decoder with all decoder attention replaced by R-SWA (constant KV cache, one-shot 32K long-doc parsing). Eventual consumer: LocalMind's OCR tile. Deliverable of this section: export feasibility, op-gap, quant plan, **go/no-go**.

### Verdict: **GO — the spike brief's hard 80% doesn't exist in this model**

Three premise-busting findings, each verified against primary sources (`config.json`, `modeling_deepseekv2.py`, `modeling_unlimitedocr.py`, `deepencoder.py` — vendored into `deepseek-ocr-spike/upstream/` — plus the GGUF headers parsed byte-level from HF):

1. **There is no MLA.** `config.json`: `"use_mla": false`, `q_lora_rank: null`, `kv_lora_rank: null`, `qk_rope_head_dim: 0`. When `use_mla=false` the modeling code routes to `SlidingWindowLlamaAttention(LlamaAttention)` — **bog-standard Llama MHA**: separate q/k/v/o `[1280×1280]` projections, no biases, RoPE (θ=10000 default, full head_dim), softmax/√d. 10 heads × 128 head_dim, `num_kv_heads = 10` (not even GQA). The engine already runs this attention (it's the Qwen3 path minus per-head QK-norm).
2. **R-SWA is KV-cache management, not a kernel.** The reference implementation is a **ring buffer**: reference tokens (vision + prompt prefill) are kept forever; generated tokens overwrite slots in a `sliding_window_size = 128` ring (`kcache[:,:,slot:slot+1,:] = k` at `slot = prefill_len + ring_pos`). Decode attends over the full (constant-size) cache with **no mask at all** (q_len=1, every slot valid — softmax is permutation-invariant, and keys keep their original RoPE rotations so position gaps are benign). Corollary that collapses prototype scope: **for the first 128 generated tokens, R-SWA ≡ full causal attention** (the ring hasn't wrapped) — a phase-1 prototype needs zero attention changes; the ring is a phase-2 feature for the long-doc headline.
3. **GGUFs already exist and GGUF is the engine's native input.** Community conversions (via llama.cpp PR #17400, not yet in upstream main) ship a full K-quant spread + fp16 mmproj: [sahilchachra/Unlimited-OCR-GGUF](https://huggingface.co/sahilchachra/Unlimited-OCR-GGUF) — **Q4_K_M 1.95 GB + mmproj-F16 812 MB = 2.76 GB in-tab total** (IQ4_XS 1.64 GB as the tighter option, needs an I-quant decoder). Both files downloaded to `deepseek-ocr-spike/models/` and header-verified. No ONNX export exists anywhere (checked onnx-community + HF search) — the ONNX path means building the export ourselves; the GGUF path means **no export step at all**.

### Architecture ground truth (from GGUF headers + config.json — trust these, not the blog posts)

| Component | Fact |
|---|---|
| Decoder | `deepseek2-ocr` GGUF arch · **12 layers** · hidden 1280 · vocab 129,280 · untied `output.weight` (Q6_K) · GGUF size label `64x550M` |
| Attention | MHA 10q/10kv × head_dim 128 · RoPE θ=10000 · RMSNorm(ε=1e-6) · **no QK-norm, no biases** · scaling 1/√128 |
| MoE (layers 1–11) | router `ffn_gate_inp [1280,64]` F32 · softmax scoring, **greedy top-6 of 64**, norm_topk_prob · experts packed 3-D: `ffn_{gate,up}_exps [1280,896,64]`, `ffn_down_exps [896,1280,64]` · **2 shared experts pre-fused** into one FFN `[1280,1792]` (added residually, ungated) |
| Layer 0 | plain dense FFN, intermediate 6848 (`first_k_dense_replace: 1`) |
| FFN activation | SiLU (SwiGLU) — same as Qwen3 path |
| Vision (mmproj, 476 tensors, F16/F32) | **SAM ViT-B**: patch-embed conv 16×16 s16 (1024px → 64×64), 12 blocks, 768-dim, fused QKV+bias, **window-14 attention** (global at layers 2/5/8/11), **decomposed rel-pos** (`pos_h/pos_w [64,27]` per block), LayerNorm+bias, GELU; neck + 2 stride-2 convs (256→512→1024 ch) → **16×16×1024** = 16× compression. **CLIP-L**: 24 blocks, 1024-dim, 16 heads — consumes **SAM's output as its patch_embeds** (not raw pixels at full res), 257 tokens incl. CLS, interpolated pos-embeds. Fusion: `cat(CLIP[:,1:], SAM.flatten)` → 2048 → **one linear** `mm.model.fc [2048→1280]` → **256 vision tokens per 1024² tile** + `image_newline`/`view_seperator` embeddings |
| Sequence splice | `<image>` placeholder = token id **128815**, embeddings masked-scatter'd in; grid layout with newline token per row; global view + local crop views |
| Tokenizer | BPE (`gpt2`-family, `deepseek-v3` pre), 129,280 vocab · chat template = **trivial passthrough** (`{% for m in messages %}{{m['content']}}{% endfor %}`) — prompts are raw DeepSeek-OCR style (`<image>\n<|grounding|>Convert the document to markdown.`) |
| KV cache | **constant**: (prefill_len + 128) slots × 12 layers × 1280 × 2 × f16 ≈ **61 KB/slot** → single page ≈ 90 MB, *independent of output length* |

### Op-gap table — custom-WGSL engine path (Path B, recommended)

| Op / capability | Engine today | Gap | Effort |
|---|---|---|---|
| MHA + RoPE + RMSNorm + SiLU FFN | ✅ Qwen3 path (attn 1/√d, rope, rms_norm, silu_mul) | none — config flags only (`qk_norm: false`) | ~0 |
| Untied lm_head, large vocab | ✅ (Qwen3-8B: untied, chunked embed decode) | none | 0 |
| **MoE block** | ❌ dense only | router GEMV (tiny, F32) + **softmax-top-6 kernel** (64 logits, 1 workgroup) + **expert-indexed GEMV** (matmul_q4k_mr + expert-offset read from a GPU buffer — no CPU readback) + scaled-accumulate; shared expert = existing FFN at I=1792; layer-0 dense = existing | **3–5 days** incl. crossLabDiff green |
| **Ring-buffer KV (R-SWA)** | sliding-window plumbing exists (Gemma), but mask-based, not ring | slot-index write in kv_cache_store + fixed attn length P+128; **deferrable — ≤128-token outputs are exactly causal** | 2–3 days (phase 2) |
| `inputs_embeds` injection (vision tokens) | ❌ token-id lookup only | prefill path that copies precomputed 1280-dim rows from a GPU buffer instead of embedding_lookup | ~1 day |
| Q5_0 source dequant (`ffn_down_exps`) | ❌ (F32/F16/BF16/Q8_0/Q4_K/Q5_K/Q6_K) | ~30-line decoder in gguf.ts | hours |
| 3-D expert tensor upload | per-2D-tensor buffers | one buffer per exps tensor + expert stride (gate_exps Q4_K ≈ 41 MB/layer — trivially under the 4 GB buffer cap) | in MoE item |
| Batched prefill | ❌ token-at-a-time (M=1 engine) | OCR prefill ≈ 300–1500 tokens → ~5–15 s/page at decode speed; acceptable for prototype, batch later | later opt |
| **Vision stack in WGSL** | ❌ (no conv2d, LayerNorm+bias, window-attn, rel-pos) | full SAM+CLIP+projector port | **2–3 wks (defer)** |
| Vision via ORT-web hybrid | lab already drives raw ORT-web | export DeepEncoder-only to ONNX fp16 (plain PyTorch vision graph, no cache/MoE/custom-attn → standard `torch.onnx.export` territory; bicubic pos-embed interp precomputed at export) + JS glue: ORT output → GPU buffer → engine | **3–5 days** |

**Path A (ONNX/transformers.js@4) assessment:** upstream has the ops — v4.0.0 ships **QMoE** (`com.microsoft.QMoE`), MLA (unneeded here), **DeepSeek-V3 arch** (PR #1586, architecture-only), and OCR-VLM precedent (GLM-OCR, LightOnOCR). No `deepseek-ocr`/`unlimited-ocr` arch in transformers.js and no optimum export recipe — the work is a hand-written export (decoder w/ past_kv + QMoE insertion + vision graph) plus a transformers.js model class; R-SWA lives outside the graph as JS-side KV eviction (the decode loop already feeds `past_key_values` explicitly, so eviction is tensor slicing between steps). Estimate **2–4 weeks**, most of it export tooling — and it duplicates what the GGUF path gets for free. **Decision: Path B first; Path A only as an upstream contribution later** (it's the LocalMind drop-in shape, worth doing once the kernels prove the model).

### Quantization plan

- **Consume the community imatrix quants as-is** (they're calibrated; `Unlimited-OCR.imatrix` published). Q4_K_M mix: attn/gate/up Q4_K, `ffn_down_exps` Q5_0, some attn_v/output Q6_K, router F32. The engine requants source→F16→in-shader q4k for layer matmuls; the Q5_0→q4k double-quant is the only added loss — measure via crossLabDiff, fall back to in-shader q8 for down_exps if the cliff shows there.
- Budget check: layer weights in-shader q4k ≈ 1.2 GB + embed/lm_head F16 662 MB + vision 812 MB ≈ **2.7 GB GPU weights + ~90 MB constant KV** — inside the 4 GB single-buffer ceiling with room, and a 24 GB laptop doesn't blink. Embeddings/lm_head → q8 later if the sidecar slice demands it.
- Accuracy gate (lab hard rule): CPU-reference harness — `deepseek_ocr_smoke.py` mirroring `reference/pytorch/qwen3_smoke.py` (HF `trust_remote_code` bf16, text-only prompt, layer-by-layer npz) → crossLabDiff green before any browser claim.
- Speed envelope: ~570 M active params/token (label `64x550M` ✓ my per-tensor arithmetic) ≈ 0.35 GB reads/token at q4k — **⅓ of Qwen3-1.7B's per-token traffic**, same-machine anchor 40 tok/s → **40–80 tok/s class** expected (MoE gather overhead is the unknown); even the floor transcribes a dense page (~1000 tokens) in ~25 s, decode speed flat in document length thanks to constant KV.

### Prototype ladder (smallest end-to-end proof)

- **P0 — decoder-only text smoke** (no vision): `deepseek2-ocr` config + loader mapping + MoE kernels + Q5_0 decoder → greedy decode from a text prompt, crossLabDiff vs HF bf16 reference. *This is the go/no-go kernel milestone.* ~1 wk.
- **P1 — vision hybrid**: DeepEncoder → ONNX fp16 → ORT-web session → projector output → `inputs_embeds` injection → **single-page 1024px OCR on WebGPU**, ≤128-token output (R-SWA ≡ causal, no ring needed). ~1 wk.
- **P2 — long-doc**: ring-buffer KV + batched prefill → multi-page one-shot parse, the constant-KV headline nobody else has in-browser. ~1 wk.
- **P3 — publish**: HF `naklitechie/` (MIT upstream, notice retained) — engine-ready GGUF mirror + ONNX vision graph + demo page; candidate spin-off as its own public repo.

### Caveats / watch items

- The GGUFs require llama.cpp **PR #17400** (unmerged); if upstream lands different tensor naming/arch key, re-convert — our loader keys on the current names. The BF16 GGUF (5.88 GB) in the same repo is the requant fallback.
- Arch is 3 weeks old; watch for Unlimited-OCR v1.x re-releases (Baidu iterates fast) and for transformers.js adding the family upstream (would hand us Path A cheaply — recheck at P1).
- Engine tokenizer: byte-level BPE `Ġ` handling is the already-tracked weak layer; DeepSeek-V3 pre-tokenizer regex needs porting care (OCR outputs are markdown-heavy — spacing bugs will show).
- Vision preprocessing (resize to 1024², normalize mean/std 0.5, tiling for multi-page) lives in JS canvas — mechanical but must match `processor_config.json` exactly.
