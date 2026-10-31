"""Command line entry point.

Implemented in this build stage:

    validate   schema, template resolution, and per-span token counts
    replay     dispatches the claim, writes the manifest, and runs the
               determinism gate

Not implemented yet (they exit 3 and say so):

    ablate     span removal sweep and minimal sufficient search
    report     report.md / report.html verdict card
    verify     re-dispatch a report's manifest and diff it row by row

Exit codes: 0 success, 1 nondeterministic replay, 2 claim/template/endpoint
error, 3 command not implemented in this stage.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import NoReturn, Optional, Sequence

import typer

from .backend import Backend, BackendError, ChatResult, build_backend
from .claim import SPAN_ROLES, Claim, ClaimError, load_claim
from .manifest import (
    DeterminismResult,
    Manifest,
    build_manifest,
    check_determinism,
    manifest_sha256,
)
from .prompt import (
    ChatTemplate,
    TemplateError,
    build_messages,
    render_prompt,
    resolve_chat_template,
)

EXIT_NONDETERMINISTIC = 1
EXIT_ERROR = 2
EXIT_NOT_IMPLEMENTED = 3

CONFIG_ID_FULL = "full"

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Ablate the context a harness injected into a model-behaviour claim.",
)

_FIXTURE_OPTION = typer.Option(
    False,
    "--fixture",
    hidden=True,
    help="Serve deterministic local answers instead of calling an endpoint (offline checks).",
)


def _die(message: str, code: int = EXIT_ERROR) -> NoReturn:
    typer.echo(message, err=True)
    raise typer.Exit(code)


def _fixture_enabled(flag: bool) -> bool:
    return flag or os.environ.get("INJABLAT_FAKE_BACKEND") == "1"


def _read_claim(path: Path) -> Claim:
    try:
        return load_claim(path)
    except ClaimError as exc:
        _die(str(exc))


def _render(claim: Claim, backend: Backend) -> tuple[ChatTemplate, list[dict[str, str]], str]:
    """Resolve the pinned template and render the prompt for this claim."""
    template = resolve_chat_template(claim, backend)
    messages = build_messages(claim)
    prompt = render_prompt(template, messages, continue_final_message=claim.has_prefill)
    return template, messages, prompt


def _role_counts(claim: Claim, measured: dict[str, int]) -> list[tuple[str, int, int, int]]:
    """(role, span count, recorded tokens, measured tokens) in a stable order."""
    rows = []
    for role in SPAN_ROLES:
        spans = [span for span in claim.spans if span.role == role]
        if not spans:
            continue
        recorded = sum(span.tokens for span in spans)
        seen = sum(measured[span.id] for span in spans)
        rows.append((role, len(spans), recorded, seen))
    return rows


def _print_claim_header(claim: Claim, build: str, template) -> None:
    spec = claim.model
    typer.echo(f"claim {claim.id}")
    typer.echo(f"model {spec.name} @ {spec.endpoint} ({spec.backend}, revision {spec.revision})")
    typer.echo(f"endpoint build {build}")
    typer.echo(f"chat template sha256:{template.sha256} ({template.source})")


@app.command()
def validate(
    claim_path: Path = typer.Argument(
        ..., exists=True, dir_okay=False, readable=True, help="Path to claim.yaml."
    ),
    fixture: bool = _FIXTURE_OPTION,
) -> None:
    """Check the claim, render its prompt, and count each span with /tokenize."""
    claim = _read_claim(claim_path)

    try:
        with build_backend(claim.model, fixture=_fixture_enabled(fixture)) as backend:
            template, messages, prompt = _render(claim, backend)
            measured = {span.id: backend.tokenize(span.text).count for span in claim.spans}
            prompt_tokens = backend.tokenize(prompt).count
            build = backend.identity()
    except (TemplateError, BackendError) as exc:
        _die(str(exc))

    _print_claim_header(claim, build, template)
    injected = sum(1 for span in claim.spans if span.injected)
    typer.echo(f"spans {len(claim.spans)} ({injected} injected, {len(claim.spans) - injected} user prompt)")
    typer.echo("")
    typer.echo(f"  {'role':<12}{'spans':>6}{'recorded':>10}{'measured':>10}")
    for role, count, recorded, seen in _role_counts(claim, measured):
        typer.echo(f"  {role:<12}{count:>6}{recorded:>10}{seen:>10}")

    drifted = [
        f"{span.id} ({span.tokens} recorded, {measured[span.id]} measured)"
        for span in claim.spans
        if measured[span.id] != span.tokens
    ]
    if drifted:
        typer.echo("")
        typer.echo(
            "note: the endpoint tokenizer disagrees with the recorded counts: " + ", ".join(drifted)
        )

    typer.echo("")
    typer.echo(f"rendered {len(messages)} messages -> {prompt_tokens} prompt tokens")
    typer.echo("ok")


@app.command()
def replay(
    claim_path: Path = typer.Argument(
        ..., exists=True, dir_okay=False, readable=True, help="Path to claim.yaml."
    ),
    runs: int = typer.Option(2, "--runs", "-n", min=2, help="Dispatches per configuration."),
    out: Optional[Path] = typer.Option(
        None, "--out", help="Report directory (default: reports/<claim id>)."
    ),
    fixture: bool = _FIXTURE_OPTION,
) -> None:
    """Replay the claim, write its manifest, and fail when repeat runs disagree."""
    claim = _read_claim(claim_path)

    try:
        with build_backend(claim.model, fixture=_fixture_enabled(fixture)) as backend:
            template, messages, prompt = _render(claim, backend)
            prompt_ids = backend.tokenize(prompt).tokens
            build = backend.identity()

            manifests: list[Manifest] = []
            chats: list[ChatResult] = []
            for _ in range(runs):
                chat = backend.chat(
                    messages, claim.sampling, continue_final_message=claim.has_prefill
                )
                manifests.append(
                    build_manifest(
                        claim=claim,
                        config_id=CONFIG_ID_FULL,
                        template=template,
                        endpoint_build=build,
                        prompt_tokens=prompt_ids,
                        chat=chat,
                    )
                )
                chats.append(chat)
    except (TemplateError, BackendError) as exc:
        _die(str(exc))

    verdict = check_determinism(manifests)
    report_dir = out if out is not None else Path("reports") / claim.id
    manifest_path, verdict_path, records = _write_replay(report_dir, manifests, chats, verdict)

    _print_claim_header(claim, build, template)
    sampling = claim.sampling
    typer.echo(
        f"sampling temp {sampling.temperature}  top_p {sampling.top_p}  "
        f"seed {sampling.seed}  max_tokens {sampling.max_tokens}"
    )
    typer.echo(f"prompt {manifests[0].prompt_tokens} tokens sha256:{manifests[0].prompt_token_sha256}")
    if verdict.ok:
        typer.echo(f"manifest sha256:{manifest_sha256(manifests[0])}")
    else:
        # No stable fingerprint exists when the repeats disagree; the per-run
        # hashes stay in replay.json for debugging, not as a pin.
        typer.echo("manifest sha256: unavailable (runs disagree)")
    typer.echo(f"determinism {verdict.verdict.value} ({verdict.detail()})")
    typer.echo(f"wrote {manifest_path}")
    typer.echo(f"wrote {verdict_path} ({records} run records)")

    if not verdict.ok:
        raise typer.Exit(EXIT_NONDETERMINISTIC)


def _write_replay(
    report_dir: Path,
    manifests: Sequence[Manifest],
    chats: Sequence[ChatResult],
    verdict: DeterminismResult,
) -> tuple[Path, Path, int]:
    """Persist the manifest, one record per dispatch, and the gate's verdict."""
    runs_dir = report_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = report_dir / "manifest.json"
    _write_json(manifest_path, manifests[0].model_dump(mode="json"))

    records = []
    for index, (manifest, chat) in enumerate(zip(manifests, chats), start=1):
        record_path = runs_dir / f"{manifest.config_id}.run{index}.json"
        _write_json(
            record_path,
            {
                "config_id": manifest.config_id,
                "run": index,
                "manifest": manifest.model_dump(mode="json"),
                "request_body": chat.request_body,
                "response_body": chat.response_body,
                "elapsed_ms": chat.elapsed_ms,
            },
        )
        records.append({"run": index, "output_sha256": manifest.output_sha256})

    verdict_path = report_dir / "replay.json"
    _write_json(
        verdict_path,
        {
            "claim_id": manifests[0].claim_id,
            "config_id": manifests[0].config_id,
            "manifest_sha256": manifest_sha256(manifests[0]),
            "determinism": verdict.verdict.value,
            "detail": verdict.detail(),
            "runs": records,
        },
    )
    return manifest_path, verdict_path, len(records)


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def _not_implemented(command: str) -> NoReturn:
    typer.echo(f"{command}: not implemented in this build stage", err=True)
    raise typer.Exit(EXIT_NOT_IMPLEMENTED)


@app.command()
def ablate(
    claim_path: Optional[Path] = typer.Argument(None, help="Path to claim.yaml."),
) -> None:
    """Remove spans, truncate the prefill ladder, and search for the minimal set."""
    _not_implemented("ablate")


@app.command()
def report(
    claim_path: Optional[Path] = typer.Argument(None, help="Path to claim.yaml."),
) -> None:
    """Write the verdict card as report.md and report.html."""
    _not_implemented("report")


@app.command()
def verify(
    report_path: Optional[Path] = typer.Argument(None, help="Path to a report or manifest."),
) -> None:
    """Re-dispatch a recorded manifest and print MATCH or DRIFT per row."""
    _not_implemented("verify")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
