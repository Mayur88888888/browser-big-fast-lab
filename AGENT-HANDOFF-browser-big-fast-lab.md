# Agent Handoff — browser-big-fast lab (v2, final)

**Objective:** Run as big a model as possible, as fast as possible, in a browser. Operating theme: AI as a sidecar — the model shares the machine with a real app, gets a polite memory slice (~4–6GB), and is judged on time-to-first-useful-token and tool-calling reliability, not just raw tok/s.
**Supersedes:** AGENT-HANDOFF-diffusion-gemma4-small.md and AGENT-HANDOFF-gemma4-fast-lab.md.
**Parent:** kohra (github.com/NakliTechie/kohra). Operator: Chirag. Agent: Claude Code. Autonomous between gates. Lab → comparison table → decide.

---

## The levers (why these tracks)

1. **Memory ceiling is the binding constraint.** Browser "big" is decided by working q4 on WebGPU, not by decode technique. q4 broken = fp16 ceiling (~3–4B dense). q4 fixed = 8–12B-class.
2. **Sparse MoE is the size play.** Big-total/small-active gives big-model capacity at small-model bandwidth cost. Per-token compute scales with active params.
3. **QAT removes the q4 quality tax.** Google's Gemma 4 QAT checkpoints (June 2026, all sizes + MTP drafters) put q4 within a few points of bf16. q4 becomes the default tier, not a compromise.
4. **MTP is the speed play for dense AR** (1.5–2.2× wall-clock, no quality loss, keeps KV cache, low TTFT — right profile for sidecar chat). Costs ~2GB extra memory — directly eats the sidecar slice; measure, don't assume.
5. **Diffusion is repositioned:** not a speed play at this scale (kohra's own benchmark: AR wins at small scale, diffusion costs quality). Its remaining case is the *edit-in-place / infill* sidecar capability (bidirectional attention), which AR can't do at any speed.

## Tracks (in priority order)

### T0 — q4 on WebGPU (the unblock; gates everything "big")
Status quo: dense q4 `MatMulNBits` miscomputes on WebGPU (Apple GPU + ORT-web 1.26.0); CPU decodes correctly. fp16 ships; q4 parked.
- T0-G1: Sweep newer ORT-web dev builds (the 1.26.0-dev.20260416 build fixed MatMulNBits for the kohra q4 variant — start there) × quant configs (RTN sym, block sizes, accuracy_level) against a known-good CPU reference. Build a tiny correctness harness (logit diff vs CPU) — reusable artifact.
- T0-G2: If ORT-web path stays broken: document precisely, file/locate upstream issue, and declare the q4-browser route Kiln/MLC territory (q4 demonstrably works there — LFM2-8B-A1B precedent). Do not build custom kernels.
- Exit artifact: "q4 on WebGPU: works with config X / blocked, route via MLC" — one paragraph + harness results.

### T1 — LFM2.5-8B-A1B in the browser (the anchor: big + sparse + sidecar-native)
8.3B total / 1.5B active MoE, tool-calling + reasoning post-train (RL), 128K ctx, day-one ONNX, <6GB quantized. Hybrid LIV-conv + 6 GQA layers = minimal KV bloat (sidecar-friendly long context). Lineage already proven in-browser (LFM2-8B-A1B via LocalMind).
- T1-G0 recon: (a) transformers.js / ORT-web arch support for LFM2.5 (vocab doubled to 128K, otherwise LFM2-8B-A1B arch — likely near-supported; check onnx-community). (b) Reasoning-only model — establish whether thinking channel can be suppressed/short-circuited and what it costs in quality; TTFT with vs without CoT is a headline sidecar metric. (c) Quant plan: MoE experts symmetric-only (hard ORT-web rule, lesson already learned on LFM2).
- T1-G1: Runs in browser on WebGPU, q4 (depends on T0 for the dense parts; MoE quant path was already viable for LFM2 — verify).
- T1-G2: Sidecar eval: TTFT, tok/s, memory high-water mark, GPU utilization, ToolCall-15 score (this model's reason for existing — report per-category).
- Scope split: ONNX/JS path = this lab. MLC/WebLLM path = Kiln (exists; don't duplicate).

### T2 — Gemma 4 E4B-QAT + MTP in the browser (the dense quality+speed stack)
The bullseye artifact: E4B QAT checkpoint + Google's QAT'd MTP drafter, q4 ONNX, JS draft-verify speculative loop (kohra pattern: bypass transformers.js generate(), raw forwards, keep KV cache). E4B AR already runs in-browser out of the box — that's the baseline to beat.
- T2-G0 recon: (a) MTP graph shape — extra heads on same checkpoint or separate drafter model? Export strategy follows. (b) **Quantizer-grid eval:** QAT checkpoints ship as `qat-q4_0-unquantized` (weights snapped to q4_0 grid, stored bf16). Naive conversions measurably lose accuracy (Unsloth: 70.2% naive vs 85.6% recovered, 26B). Verify our ONNX RTN-symmetric quant lands on/near the q4_0 grid; eval against bf16 reference, don't assume. Use NVIDIA's NVFP4 checkpoint quality numbers as a second quantized-quality reference point where published. Helpful: q4_0 is symmetric — matches the ORT-web constraint. (NVFP4 itself doesn't port: its speed is native Blackwell FP4 compute, absent on Metal/WebGPU; its quality trick — micro-block FP scaling — is already approximated by existing block-quant schemes. MXFP4 = watch item only.)
- T2-G1: E4B-MTP ONNX export; MTP head outputs verified vs reference in Python ORT.
- T2-G2: JS speculative loop; coherent; acceptance-rate telemetry.
- T2-G3: Bench vs plain AR E4B: wall-clock target ≥1.3×; memory delta of MTP heads measured against the sidecar slice.
- Ships as sibling module (`kohra-mtp.js` working name); kohra.js API frozen.

### T3 — LLaDA-MoE-7B-A1B in kohra (the big-diffusion candidate, zero training)
Promoted from kohra G3. 7B total / 1.4B active, first open MoE diffusion LM, quality ≈ Qwen2.5-3B. The `-Instruct-TD` trajectory-distilled variant attacks step count (the proven speed lever: cost linear in steps).
- T3-G1: ONNX export (MoE symmetric quant; size class proven by LFM2-8B-A1B). Runs in kohra.
- T3-G2: Bench both variants (base + TD) with threshold decoding; **plus the infill test** — edit-in-place / fill-in-middle cases AR can't do, the capability that justifies diffusion in a sidecar.

### T4 — Fine-tune → MTP recon (cheap question, answer it; heavy work gated)
Does fine-tuning kill Google's pre-trained MTP heads (acceptance collapse → speedup gone)?
- T4-G0 (near-free): light LoRA on E4B-QAT (Unsloth, Lightning), measure MTP acceptance pre/post. Secondary: does fine-tuning degrade QAT quant robustness? 
- Outcomes: survives → fine-tuned sidecar models keep the MTP speedup, pipeline = FT → merge → T2 export. Collapses → head re-tune experiment OR park with findings.
- No heavy spend before T2-G2 exists.

### T5 — Gemma A2D diffusion conversion (DEMOTED: only if T1–T3 disappoint)
Original idea (convert dense E2B/E4B to MDLM via dLLM toolkit) serves neither size (dense) nor sidecar speed (kohra's own AR-vs-MDLM benchmark) — and T3 tests the diffusion-capability question for free. Retained as contingency; original gate ladder lives in the superseded handoff. Hackable Diffusion recon folds into T3/T5 only if activated.

## Locked decisions

1. **Effort order: T0 → T1 → T2 → T3 → T4-G0 (cheap, anytime) → T5 only on explicit go.** T0 and the G0 recons can run as one opening batch.
2. **One harness** (extend kohra `web/bench.html`): fixed 10-prompt general set + **ToolCall-15** for tool calling + 5 infill cases. Metrics per mode: TTFT, tok/s, wall-clock, memory high-water, **GPU utilization** (compute-bound vs bandwidth-bound diagnostic — diffusion's payoff scales with compute-rich/bandwidth-poor hardware; if AR shows the GPU idling and diffusion saturates it, the browser sits on diffusion's side of that law), quality side-by-side, ToolCall-15 score. Every gate artifact uses it.
   - **ToolCall-15 adoption** (github.com/stevibe/ToolCall-15, MIT): 15 deterministic scenarios × 5 categories (tool selection, parameter precision, multi-step chains, restraint/refusal, error recovery), temperature 0, mocked tools, 2/1/0 scoring. Vendor `lib/benchmark.ts` (scenario spec + mock handlers + scoring) into the browser harness with attribution — do NOT build an OpenAI-compatible HTTP shim around in-page models; port the spec, keep scoring logic byte-equivalent so scores stay comparable to published runs. Preserve the fixed reference-date anchor (2026-03-20) and up-to-8-turn loop. Restraint/refusal and error recovery are the sidecar-critical categories — report per-category, not just the composite.
3. **Compute budget $500 total** (Lightning). T0/T1/T2/T3 are export+eng (~$0 GPU training). Only T4/T5 train. Beyond cap = escalation.
4. **Targets are QAT checkpoints wherever they exist** (all Gemma work).
5. **Publishing:** HF `naklitechie/`, `{model}-{variant}-ONNX`. Licenses follow upstream: Gemma derivatives = Gemma Terms of Use; LFM2.5 = LFM Open License (verify exact terms at T1-G0); LLaDA-MoE = upstream license (verify). Compliance check in every G0. No blanket Apache 2.0.
6. **12B-dense-in-browser: out of scope** (q4 8–9GB + headroom vs WebGPU buffer limits). 12B/26B-class = desktop (Edge-First L1 rung), and any future big-browser question via Kiln/MLC, not here.
7. **Text-only everywhere; strip/freeze vision-audio towers in exports.**

## Carried constraints (hard rules)

- MoE quantization on ORT-web: **symmetric only**, no zero-point.
- fp16 WebGPU: RMSNorm fused pre-convert or silent zero logits; verify per-arch (Gemma 4, LFM2.5, LLaDA), fuse manually if optimizer lacks the model_type.
- Never ship known-wrong q4. CPU-reference correctness harness before any browser claim.
- MDLM: no KV cache, steps×forward profile — don't fight it. AR+MTP: keep the cache.
- kohra.js public API frozen; new loops = sibling modules.
- No custom WebGPU kernels, no new primitives/infra, no multimodal, no custom training-data pipelines.

## Escalation protocol

Own authority: ORT versions, quant configs, hyperparameters, export tactics, debugging, prompt/test-set composition, library choices. Stop only for: (1) locked-decision conflict, (2) spend beyond cap / new paid dependency, (3) route ambiguity (e.g., MTP graph un-exportable; LFM2.5 arch unsupported and port cost is large; T0 dead-end making the lab pointless on ONNX path).

## End-of-lab deliverable

One comparison table, all modes on the one harness:

| Mode | Model | TTFT | tok/s | wall-clock | mem peak | GPU util | quality | ToolCall-15 | infill | in-browser today? |
|---|---|---|---|---|---|---|---|---|---|---|
| AR baseline | Gemma4 E4B(-QAT) | | | | | | | | n/a | |
| AR+MTP | E4B-QAT+drafter | | | | | | | | n/a | |
| Sparse MoE AR | LFM2.5-8B-A1B | | | | | | | | n/a | |
| Diffusion MoE | LLaDA-MoE (base+TD) | | | | | | | | ✓ | |

That table + one page of findings = the decision input for what ships into LocalMind / the sidecar pattern. No other reports.
