# Vendored third-party sources

## toolcall15-benchmark.ts
Verbatim from [stevibe/ToolCall-15](https://github.com/stevibe/ToolCall-15) `lib/benchmark.ts`
(MIT, see toolcall15-LICENSE), commit `615b1576e2`, vendored 2026-06-12.

Self-contained: `SYSTEM_PROMPT`, `UNIVERSAL_TOOLS` (12 tools), `SCENARIOS` (15 = 5 categories ×3),
`scoreModelResults`, `CATEGORY_LABELS`. The harness (web/toolcall15.html) transpiles this in-browser
(Sucrase) and drives each scenario through an in-browser Qwen3 ORT session via `<tool_call>` parsing,
replacing ToolCall-15's OpenAI-API transport. Scenario scoring is used unmodified.
