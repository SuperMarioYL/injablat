**English** | [简体中文](./README.md)

<p align="center">
  <img src="https://readme-typing-svg.demolab.com?font=JetBrains+Mono&weight=600&size=22&pause=1200&color=2F81F7&center=true&vCenter=true&width=920&lines=Reproduce%20is%20not%20attribution;Does%20the%20claim%20survive%20without%20the%20prefill%3F;bit-exact%20manifest%20%C2%B7%20minimal%20sufficient%20injection%20%C2%B7%20verifiable%20verdict%20card" alt="Reproduce is not attribution · Does the claim survive without the prefill?">
</p>

<h1 align="center">Injablat</h1>

<p align="center"><b>Remove each span the harness injected, re-run the claim, and see whether the claimed model behaviour still holds.</b></p>

<p align="center">
  <a href="https://github.com/SuperMarioYL/injablat/actions/workflows/ci.yml"><img src="https://img.shields.io/github/actions/workflow/status/SuperMarioYL/injablat/ci.yml?branch=main&label=ci" alt="CI"></a>
  <img src="https://img.shields.io/badge/python-3.12-3776AB?logo=python&logoColor=white" alt="Python 3.12">
  <img src="https://img.shields.io/badge/license-MIT-2ea44f" alt="MIT">
  <img src="https://img.shields.io/badge/version-0.1.0-6f42c1" alt="v0.1.0">
  <img src="https://img.shields.io/badge/backends-vLLM%20%7C%20llama.cpp-0b7285" alt="vLLM and llama.cpp">
</p>

## What it solves

A public claim says the model completes a partial cURL command. You reproduce it on your own endpoint: with the same system prompt, prefill, and tool schema, it does. But reproduction only shows that the behaviour appeared in some context. It does not answer who made it appear.

Injablat turns that question into a controlled deletion. Pin temperature 0, seed 0, and batch 1; record the model revision, chat template, tokenizer, and request bytes; then remove one injected span at a time and re-run. The output is a verdict card with the minimal sufficient injection and the answer to the question that actually matters: if this span is gone, does the conclusion still hold?

It is built for three kinds of teams:

- **Model safety and compliance teams** who must separate model behaviour from harness injection before a contested conclusion enters a report or a procurement review.
- **Agent platform and harness authors** whose system prompt, prefill, tool schema, and few-shot blocks stack on top of each other, and who need to know which layer is load-bearing.
- **Model vendor security groups** who have to answer "is this the model's own capability?" with a table the other side can re-run.

## Install

Python 3.12 or newer is required. v0.1 is not on PyPI yet, so install from source. The CLI only makes HTTP requests: no CUDA, no model download.

```bash
git clone https://github.com/SuperMarioYL/injablat.git
cd injablat
uv venv --python 3.12
uv pip install -e .
source .venv/bin/activate
injablat --help
```

Without uv, use the standard library venv:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

## Quick start

Start with the offline path to confirm the mechanism. It uses the bundled deterministic fixture, contacts no endpoint, and needs no GPU:

```bash
injablat validate examples/qwen-prefill/claim.yaml --fixture
injablat replay examples/qwen-prefill/claim.yaml --runs 2 --fixture
injablat ablate examples/qwen-prefill/claim.yaml --runs 2 --fixture
injablat report examples/qwen-prefill/claim.yaml
injablat verify reports/qwen-prefill-curl --fixture
```

The fixture answers come from a local simulated endpoint. They demonstrate the ablation mechanism and do not represent the behaviour of any released weights. The card tags the endpoint build with `+fixture` rather than passing the run off as a real measurement.

To measure a real endpoint, start an OpenAI-compatible server:

```bash
vllm serve Qwen/Qwen3-8B-Instruct --port 8000 --seed 0 --max-model-len 8192
```

Then replace `model.endpoint`, `model.revision`, and `chat_template_sha` in `examples/qwen-prefill/claim.yaml` with the values for those weights, keep the claim's own chat template, and run the same commands without `--fixture`. The template travels with the claim. If the claim has no template and the endpoint does not provide one, `validate` exits with an error instead of guessing.

llama.cpp uses the same `/v1/chat/completions` surface: set `backend` to `llamacpp` and point the claim at `llama-server`. Prefill ablation is out of scope for v0.1; see "Limitations and known boundaries".

## Demo

About 70 seconds, from `validate` to `verify`, all real output from the offline fixture:

[`docs/demo.cast`](./docs/demo.cast) · play locally with `asciinema play docs/demo.cast`

```text
claim qwen-prefill-curl
...
manifest sha256:8c5885884957544c421ea15c3a1c227b585c3f6af29d391851517d5e89b46127
runs per configuration 2
determinism DETERMINISTIC (repeat runs are byte-identical for every configuration)

  remove span                   behavior    output sha    role
  (bare prompt)                 NOT_FIRED   7d554cf3...   baseline
  (nothing removed)             FIRED       c73ca629...   reproduces
  prefill[0:2232]               NOT_FIRED   7d554cf3...   necessary
  system                        FIRED       d670c441...   redundant
  tool_schema                   NOT_FIRED   7d554cf3...   necessary
  few_shot[0:2]                 NOT_FIRED   7d554cf3...   necessary
  prefill[0:558]                NOT_FIRED   7d554cf3...   insufficient
  prefill[0:1116]               NOT_FIRED   7d554cf3...   insufficient
  prefill[0:1674]               NOT_FIRED   7d554cf3...   insufficient
  (minimal set)                 FIRED       d670c441...   verified

minimal sufficient injection  prefill[0:2232] + tool_schema + few_shot[0:2]
survives without prefill      NO
```

The card says something specific: the behaviour appears under the full injection; it disappears when the 2,232-token prefill is removed; it remains when the system prompt is removed; the tool schema and few-shot block are necessary too; truncating the prefill to 558, 1,116, or 1,674 tokens is not enough. The final minimal set was dispatched once more and still held.

The commands, exit codes, and key outputs from the same run are recorded in [`docs/demo-results.json`](./docs/demo-results.json). The bundled example text is synthetic, shaped after the public prefill dispute; its token counts come from the fixture's word-piece tokenizer, not from a Qwen tokenizer.

## How it works

### Span, claim, and manifest

A measurement is defined by three objects:

```text
Span    {id, role, text, tokens, truncations?}
        role: system | prefill | tool_schema | few_shot | user
Claim   {id, model, spans, detector, sampling}
Report  {manifest, rows, minimal_sufficient, survives_without, determinism}
```

A `Claim` pins a published statement together with the context that was injected at the time: model name, endpoint, backend, revision, the chat template hash (and optionally the template file itself), each span's text and token count, a declarative detector, and the sampling parameters. Only injected spans are ablated; the user task always stays.

The `Manifest` is the core of verifiability, not decorative metadata. It records the model revision, the endpoint build string (vLLM's `GET /version` or llama.cpp's `build_info` from `GET /props`), the chat template hash, the tokenizer name, the tokenizer probe hash, the prompt token-id hash, the sampling parameters, the request hash of the serialized messages, and the output hash. Prompt token ids come only from the backend's `/tokenize` response, never from the chat response.

Every request is written to `reports/<claim>/runs/<config>.run<N>.json` with the request bytes, response bytes, elapsed time, and output sha256. `report.md`, `report.html`, and `verify` read only those files, so every card traces back to a specific HTTP exchange.

### Re-runs and determinism

Each configuration is dispatched at least twice. If the two `output_sha256` values differ, the tool reports `NONDETERMINISTIC`, exits non-zero, and writes no manifest: without a stable fingerprint there is no card to re-check. It would rather say "cannot decide" than turn noise into a conclusion.

Both backends use `/v1/chat/completions` for inference, but tokenization, chat templates, and prefill are not interchangeable:

| Capability | vLLM | llama.cpp |
| --- | --- | --- |
| tokenize | `POST /tokenize`, accepts prompt or messages | `POST /tokenize`, accepts content and piece options |
| chat template | `GET /tokenizer_info` (server flag required) or supplied by the claim | `GET /props` or supplied by the claim |
| prefill / assistant prefix | native `continue_final_message` | not supported for prefill ablation in v0.1 |
| build identity | `GET /version` | `build_info` in `GET /props` |

The two `/tokenize` request shapes are not compatible, so the backend layer branches explicitly on `backend` instead of pretending one request works on both.

### Declarative detector

There is no model on the decision path. The detector has three kinds:

| kind | matching |
| --- | --- |
| `contains` | output contains the given substring |
| `regex` | output matches the given regular expression |
| `json_field` | parse the output as JSON and read a field |

It answers "did the claimed behaviour occur?", not "is the claim true?". Turning a natural-language claim into a checkable predicate is the user's job; the tool does not ask another LLM to referee.

### Three verdict words

| word | meaning |
| --- | --- |
| `NECESSARY` | removing this span (for prefill: truncating it to 0) makes the behaviour disappear |
| `REDUNDANT` | the behaviour still appears without this span |
| `UNDECIDED` | the configuration's repeat runs disagree, so no conclusion is given |

Each card row also carries a role that says why it was run:

| role | meaning |
| --- | --- |
| `baseline` | the user task with every injected span removed |
| `reproduces` | the full injection; the behaviour appears here |
| `insufficient` | the declared prefill prefix does not yet restore the behaviour |
| `sufficient` | the declared prefill prefix restores the behaviour |
| `verified` | the final set, dispatched once more to confirm it still holds |
| `undecided` | the repeats disagreed; this row carries no conclusion |

### Minimal sufficient injection

The search is greedy: remove one span; if the behaviour disappears, keep the removal, otherwise put it back; then dispatch the final set once more. The request count is O(number of spans), not O(2^n).

The minimal sufficient set is not unique. The greedy search gives one solution, not all solutions. The report keeps every search-path row and the deletion order, so the other side can re-run under a different order; as long as the intersection of two minimal sets is non-empty, the shared spans are still load-bearing.

## Command reference

| command | what it does | common flags |
| --- | --- | --- |
| `injablat validate <claim>` | check the schema, resolve the template, render messages, count each span with `/tokenize` | `--config` |
| `injablat replay <claim>` | re-dispatch, write the manifest, run the determinism gate | `--runs`, `--out` |
| `injablat ablate <claim>` | deletion sweep, prefill ladder, greedy minimal-set search | `--runs`, `--out` |
| `injablat report <claim>` | render `report.md`, `report.html`, and `report.json` from a recorded sweep | `--out` |
| `injablat verify <report>` | re-dispatch the card's recorded requests and print `MATCH` or `DRIFT` per row | `--endpoint`, `--runs` |

`verify` accepts a report directory, `report.json`, or `report.md`. `report.md` embeds the same card data as `report.json`, so the recipient only needs one file.

Exit codes: `0` a verdict was produced; `1` no verdict (nondeterminism or drift); `2` claim, template, or endpoint error.

`--fixture` is the local simulated endpoint for offline walkthroughs and is hidden from `--help`. It calls no real endpoint, and the card tags the endpoint build with `+fixture`.

## Configuration

Runtime settings are read from `configs/injablat.yaml`; `--config` points at another file. Command-line options override file values. Claims themselves never carry timeouts or run counts.

```yaml
backend:
  request_timeout_s: 300
  tokenize_timeout_s: 30

determinism:
  runs: 2
```

Unknown configuration keys are rejected at load time, so a switch that nothing reads cannot silently look active.

## How it differs from eval and trace tools

| | Eval frameworks | Trace replay | Injablat |
| --- | --- | --- | --- |
| starting point | a fixed task or dataset | one recorded execution | one published claim and its injected context |
| unit | aggregate score over tasks | one transcript | a span-removal search per claim |
| question | how does the model do on this task set | can this execution be reproduced | which injected span is necessary, and does the conclusion survive without it |
| output | comparable aggregate metrics | transcript and diff | verdict card, manifest, minimal sufficient injection |

Eval frameworks such as lm-evaluation-harness and Inspect AI do the unglamorous part well: task definitions, scoring, reproducibility. Their unit is a task over a dataset and their output is an aggregate meant for comparison. A per-claim span search has no leaderboard slot; that is an abstraction mismatch, not a missing feature.

A trace release is a real improvement over "trust us": full fidelity, the actual input sequence, cheap to produce. But replay proves the behaviour can be reproduced, not who caused it. The transcript with the prefill removed is one the vendor never published and has little incentive to publish. Injablat does not replace these tools; it handles the step they leave open: attribution.

## Limitations and known boundaries

- **The bundled example is synthetic.** Its text was written for the example, and its token counts come from the fixture tokenizer. It demonstrates the mechanism; it does not represent released weights.
- **The offline fixture is not a measurement.** `--fixture` only validates the flow quickly, and the card says `+fixture`.
- **Prefill ablation is vLLM-only.** It uses `continue_final_message`; llama.cpp is supported for non-prefill deletion sweeps in v0.1.
- **Only one span JSONL format is accepted.** Traces from arbitrary agent frameworks need a manual conversion; there is no automatic importer.
- **Only deletion and prefix truncation.** No rewriting, no synonym substitution, no token perturbation. A behaviour disappearing after a removal shows the span was necessary, not that it alone was sufficient.
- **The detector is declarative.** It does not understand meaning; it runs `contains`, `regex`, or `json_field`. The claim must first be turned into a checkable predicate.
- **The minimal set is not unique.** Greedy order affects the result; the report keeps the search path so it can be checked under another order.
- **No statistical significance.** The default premise is temperature 0 and seed 0; there are no confidence intervals from repeated sampling.
- **Closed endpoints with unpinnable revisions are out of scope.** Without a re-checkable manifest there is no verdict.
- **Single-process CLI with local report directories.** No web panel, server, multi-user login, or queue.
- **No native Windows packaging.** Run it with Python 3.12 on Linux or macOS.
- **Endpoint drift produces DRIFT.** New weights, drivers, batching, or a silent upgrade can change output hashes; that is exactly what `verify` exposes.
- **Token counts can disagree.** When the endpoint tokenizer disagrees with the recorded counts, `validate` prints the difference instead of quietly rewriting the claim.

## Roadmap

v0.1 implements: claim schema, template pinning, vLLM and llama.cpp non-prefill replay, manifest and determinism gate, deletion sweep, the vLLM prefill ladder, greedy minimal sufficient search, `report.md` / `report.html`, `verify` with MATCH / DRIFT, and the offline fixture.

Directions that are not implemented yet (no dates promised):

- Prefill ablation on llama.cpp through a locally rendered native `/v1/completions` request.
- A one-time importer from common agent-framework traces into span JSONL.
- Batch matrices across multiple claims and models.
- Repeated sampling and statistical significance.
- Card signing, archiving, and regression on new weights (team edition).

## Team edition and licensing

The CLI and the ablation engine remain MIT-licensed and free to use. The paid team edition sells deliverability, not a locked command line:

- **Private and on-premises deployment**: adapt to inference endpoints reachable only inside a network, with no data leaving it.
- **Card archiving and signing**: turn reports, manifests, and run records into an auditable evidence chain.
- **Regression on new weights**: re-run the same claim after every weight release and report MATCH / DRIFT over time.
- **Procurement and due-diligence templates**: reviewable report formats for compliance review, filings, and procurement.
- **A single verdict report**: for one contested claim, deliver the card, the manifest, and one `verify` against the counterpart's endpoint. This is the smallest engagement before a team licence.

The team edition is licensed annually; there is no self-serve checkout. Engagements go through a contract, invoice, and corporate payment. For seat counts, scope, and pricing, reach the maintainer through GitHub Issues with a title starting with `[team]`.

## License

MIT. See [LICENSE](./LICENSE).

<p align="center"><sub><a href="./LICENSE">MIT</a> © 2026 SuperMarioYL</sub></p>
