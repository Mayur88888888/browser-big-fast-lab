# deepseek-ocr-spike — Unlimited-OCR in-browser recon artifacts

Reference material for the T7 spike (see `../RESULTS.md` § T7). Findings live in
RESULTS.md; this folder holds the primary sources they were verified against.

- `upstream/` — vendored from [baidu/Unlimited-OCR](https://huggingface.co/baidu/Unlimited-OCR)
  (MIT, © 2026 Baidu): `config.json`, modeling/encoder python, processor config.
  The ground truth for the no-MLA finding (`use_mla: false`) and the R-SWA
  ring-buffer implementation (`modeling_deepseekv2.py:1232` `SlidingWindowLlamaAttention`).
- `lm_header.txt` / `mmproj_header.txt` — full KV metadata + tensor tables of
  [sahilchachra/Unlimited-OCR-GGUF](https://huggingface.co/sahilchachra/Unlimited-OCR-GGUF)
  `Q4_K_M` (155 tensors, arch `deepseek2-ocr`) and `mmproj-F16` (476 tensors,
  SAM+CLIP+projector), parsed byte-level via HTTP Range.
- `gguf_header.py` — the ranged remote-GGUF header parser that produced them
  (reusable: `python3 gguf_header.py <hf-resolve-url>`).

Heavy artifacts (the GGUFs themselves, 2.76 GB) live outside git in the main
checkout at `browser-big-fast-lab/deepseek-ocr-spike/models/`.

Engine work happens in `custom-kernels/` (its own repo), branch `deepseek-ocr`.
