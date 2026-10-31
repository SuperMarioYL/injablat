# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-10-31

First public release. The CLI ablate the context a harness injected into a
model-behaviour claim, re-runs the claim bit-exactly, and reports which spans the
claimed behaviour actually needs.

### m1 — replay and manifest

- Claim schema for a model, its injected spans (system, prefill, tool schema,
  few-shot, user), a declarative detector, and sampling parameters.
- Chat template resolution from the claim or the endpoint, with a hard failure
  instead of a guessed template when neither provides one.
- OpenAI-compatible replay against vLLM and llama.cpp, pinned to temperature 0,
  seed 0, and batch 1.
- Manifest with model revision, endpoint build, chat template hash, tokenizer
  identity, prompt token-id hash, request hash, and output hash.
- Determinism gate: each configuration runs at least twice; disagreeing repeats
  report `NONDETERMINISTIC`, exit non-zero, and write no manifest.
- Per-request run records under `reports/<claim>/runs/` carrying the request and
  response bytes plus output sha256.

### m2 — ablation and minimal set

- Deletion sweep with a bare-prompt baseline and a full-injection reference row.
- Prefill prefix ladder through vLLM's `continue_final_message`.
- Declarative detectors: `contains`, `regex`, and `json_field`.
- Greedy minimal sufficient injection search with the full search path preserved
  in the report.
- Verdict card as `report.md`, with `NECESSARY`, `REDUNDANT`, and `UNDECIDED`
  rows plus `survives without prefill`.
- Synthetic offline fixture and a bundled example claim, clearly tagged
  `+fixture` in every card.

### m3 — verify and demo

- `report.html` alongside the Markdown card and `report.json`.
- `injablat verify` re-dispatches the recorded requests and prints `MATCH` or
  `DRIFT` per row; `report.md` embeds the card data so one file is enough.
- Recorded offline demo at `docs/demo.cast`, with the captured commands and key
  outputs in `docs/demo-results.json`.
- Release workflow builds an sdist and wheel on a `v*` tag and attaches them to
  the GitHub release.

### Documentation

- Bilingual README: Chinese primary, full English sibling, cross-linked at the
  top.
- Documented limitations: synthetic example, vLLM-only prefill ablation, a single
  span JSONL format, deletion-only ablation, and greedy minimal sets.
