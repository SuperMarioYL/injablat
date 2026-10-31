"""Command line entry point.

    validate   schema, template resolution, and per-span token counts
    replay     dispatch the claim, write the manifest, run the determinism gate
    ablate     deletion sweep, prefill ladder, and greedy minimal-set search
    report     render the verdict card as report.md and report.html
    verify     re-dispatch a card's recorded requests and print MATCH or DRIFT

Exit codes: 0 a verdict was produced, 1 nondeterministic / no verdict / drift,
2 claim, template, or endpoint error.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import NoReturn, Optional, Sequence

import typer
from pydantic import ValidationError

from .ablate import AblationError, AblationResult, ablate_claim, load_ablation
from .backend import Backend, BackendError, ChatResult, build_backend
from .claim import SPAN_ROLES, Claim, ClaimError, ModelSpec, load_claim
from .config import ConfigError, RunSettings, load_run_settings
from .manifest import (
    TOKENIZER_PROBE,
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
from .report import extract_embedded_report, write_report
from .verdict import Report, Status, VerifyResult, assemble, verify_report

EXIT_NONDETERMINISTIC = 1
EXIT_ERROR = 2

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

_CONFIG_OPTION = typer.Option(
    None,
    "--config",
    exists=True,
    dir_okay=False,
    readable=True,
    help="Runtime settings file (default: configs/injablat.yaml when present).",
)


def _die(message: str, code: int = EXIT_ERROR) -> NoReturn:
    typer.echo(message, err=True)
    raise typer.Exit(code)


def _fixture_enabled(flag: bool) -> bool:
    active = flag or os.environ.get("INJABLAT_FAKE_BACKEND") == "1"
    if active:
        typer.echo(
            "warning: offline fixture backend is active; no real endpoint will be called",
            err=True,
        )
    return active


def _run_settings(config: Optional[Path]) -> RunSettings:
    path = config
    if path is None:
        default = Path("configs/injablat.yaml")
        path = default if default.is_file() else None
    try:
        return load_run_settings(path)
    except ConfigError as exc:
        _die(str(exc))


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
    config: Optional[Path] = _CONFIG_OPTION,
    fixture: bool = _FIXTURE_OPTION,
) -> None:
    """Check the claim, render its prompt, and count each span with /tokenize."""
    claim = _read_claim(claim_path)
    settings = _run_settings(config)

    try:
        with build_backend(
            claim.model, fixture=_fixture_enabled(fixture), settings=settings
        ) as backend:
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
    typer.echo(f"  {'span':<28}{'role':<12}{'recorded':>10}{'measured':>10}")
    for span in claim.spans:
        typer.echo(
            f"  {span.id:<28}{span.role:<12}{span.tokens:>10}{measured[span.id]:>10}"
        )

    typer.echo("")
    typer.echo("  role totals")
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
    runs: Optional[int] = typer.Option(
        None, "--runs", "-n", help="Dispatches per configuration (default: determinism.runs)."
    ),
    out: Optional[Path] = typer.Option(
        None, "--out", help="Report directory (default: reports/<claim id>)."
    ),
    config: Optional[Path] = _CONFIG_OPTION,
    fixture: bool = _FIXTURE_OPTION,
) -> None:
    """Replay the claim, write its manifest, and fail when repeat runs disagree."""
    claim = _read_claim(claim_path)
    settings = _run_settings(config)
    run_count = runs if runs is not None else settings.runs
    if run_count < 2:
        _die(f"a replay needs at least two runs, got {run_count}")

    try:
        with build_backend(
            claim.model, fixture=_fixture_enabled(fixture), settings=settings
        ) as backend:
            template, messages, prompt = _render(claim, backend)
            prompt_ids = backend.tokenize(prompt).tokens
            probe_ids = backend.tokenize(TOKENIZER_PROBE).tokens
            tokenizer_name = backend.tokenizer_name()
            build = backend.identity()

            manifests: list[Manifest] = []
            chats: list[ChatResult] = []
            usage_warnings: list[str] = []
            for index in range(1, run_count + 1):
                chat = backend.chat(
                    messages, claim.sampling, continue_final_message=claim.has_prefill
                )
                manifests.append(
                    build_manifest(
                        claim=claim,
                        config_id=CONFIG_ID_FULL,
                        template=template,
                        endpoint_build=build,
                        tokenizer_name=tokenizer_name,
                        tokenizer_probe_tokens=probe_ids,
                        prompt_tokens=prompt_ids,
                        chat=chat,
                    )
                )
                chats.append(chat)
                reported = chat.usage.get("prompt_tokens")
                if isinstance(reported, int) and reported != len(prompt_ids):
                    usage_warnings.append(
                        f"run {index}: chat usage.prompt_tokens={reported} but /tokenize "
                        f"counted {len(prompt_ids)}; the counts are a cross-check only"
                    )
    except (TemplateError, BackendError) as exc:
        _die(str(exc))

    for warning in usage_warnings:
        typer.echo(f"warning: {warning}", err=True)

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
    typer.echo(
        f"tokenizer {manifests[0].tokenizer_name} probe sha256:"
        f"{manifests[0].tokenizer_probe_sha256}"
    )
    if verdict.ok:
        typer.echo(f"manifest sha256:{manifest_sha256(manifests[0])}")
    else:
        # No stable fingerprint exists when the repeats disagree; the per-run
        # hashes stay in replay.json for debugging, not as a pin.
        typer.echo("manifest sha256: unavailable (runs disagree)")
    typer.echo(f"determinism {verdict.verdict.value} ({verdict.detail()})")
    if manifest_path is not None:
        typer.echo(f"wrote {manifest_path}")
    else:
        typer.echo("wrote no manifest (a nondeterministic config is not pinned)")
    typer.echo(f"wrote {verdict_path} ({records} run records)")

    if not verdict.ok:
        raise typer.Exit(EXIT_NONDETERMINISTIC)


def _write_replay(
    report_dir: Path,
    manifests: Sequence[Manifest],
    chats: Sequence[ChatResult],
    verdict: DeterminismResult,
) -> tuple[Optional[Path], Path, int]:
    """Persist the manifest (only when pinned), one record per dispatch, and the verdict."""
    runs_dir = report_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = report_dir / "manifest.json"
    if verdict.ok:
        _write_json(manifest_path, manifests[0].model_dump(mode="json"))
    else:
        # Never leave a stale pin behind from an earlier deterministic run.
        if manifest_path.exists():
            manifest_path.unlink()
        manifest_path = None

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
            "pinned": verdict.ok,
            "manifest_sha256": manifest_sha256(manifests[0]) if verdict.ok else None,
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


# -- ablation --------------------------------------------------------------


@app.command()
def ablate(
    claim_path: Path = typer.Argument(
        ..., exists=True, dir_okay=False, readable=True, help="Path to claim.yaml."
    ),
    runs: Optional[int] = typer.Option(
        None, "--runs", "-n", help="Dispatches per configuration (default: determinism.runs)."
    ),
    out: Optional[Path] = typer.Option(
        None, "--out", help="Sweep directory (default: reports/<claim id>)."
    ),
    config: Optional[Path] = _CONFIG_OPTION,
    fixture: bool = _FIXTURE_OPTION,
) -> None:
    """Remove spans, walk the prefill ladder, and search the minimal sufficient set."""
    claim = _read_claim(claim_path)
    settings = _run_settings(config)
    run_count = runs if runs is not None else settings.runs
    if run_count < 2:
        _die(
            f"an ablation needs at least two runs per configuration to test determinism, "
            f"got {run_count}"
        )

    out_dir = out if out is not None else Path("reports") / claim.id
    try:
        with build_backend(
            claim.model, fixture=_fixture_enabled(fixture), settings=settings
        ) as backend:
            result = ablate_claim(claim, backend, run_count)
    except (TemplateError, BackendError) as exc:
        _die(str(exc))

    report = assemble(result.ablation)
    ablation_path, manifest_path, records = _write_ablation(out_dir, result)
    _print_report_header(report)
    typer.echo(f"runs per configuration {report.runs_per_config}")
    typer.echo(f"determinism {report.determinism.value} ({report.determinism_detail})")
    typer.echo("")
    typer.echo(f"  {'remove span':<30}{'behavior':<12}{'output sha':<14}{'role'}")
    for row in report.rows:
        typer.echo(
            f"  {row.label:<30}{row.behavior.value:<12}{_short_sha(row.output_sha256):<14}"
            f"{row.role.value}"
        )
    typer.echo("")
    if report.status is Status.VERDICT:
        typer.echo(f"minimal sufficient injection  {report.minimal_text}")
        prefill = report.survives_without_prefill()
        if prefill is not None:
            typer.echo(f"survives without prefill      {'YES' if prefill else 'NO'}")
    else:
        typer.echo(report.reason)
    typer.echo("")
    typer.echo(f"wrote {ablation_path}")
    if manifest_path is not None:
        typer.echo(f"wrote {manifest_path}")
    typer.echo(f"wrote {records} run records under {out_dir / 'runs'}")

    if report.exit_code != 0:
        raise typer.Exit(report.exit_code)


def _short_sha(sha: Optional[str]) -> str:
    return f"{sha[:8]}..." if sha is not None else "(disagree)"


def _print_report_header(report: Report) -> None:
    model = report.model
    sampling = report.sampling
    typer.echo(f"claim {report.claim_id}")
    typer.echo(
        f"model {model.name} @ {model.endpoint} ({model.backend}, revision {model.revision})"
    )
    typer.echo(f"endpoint build {model.endpoint_build}   tokenizer {model.tokenizer_name}")
    typer.echo(f"chat template sha256:{report.chat_template.sha256} ({report.chat_template.source})")
    typer.echo(
        f"sampling temp {sampling.temperature}  top_p {sampling.top_p}  "
        f"seed {sampling.seed}  max_tokens {sampling.max_tokens}"
    )
    if report.manifest is not None:
        typer.echo(f"manifest sha256:{manifest_sha256(report.manifest)}")
    else:
        typer.echo("manifest sha256: not pinned (runs disagree)")


def _write_ablation(
    directory: Path, result: AblationResult
) -> tuple[Path, Optional[Path], int]:
    runs_dir = directory / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    for config_id, records in result.records.items():
        for record in records:
            _write_json(runs_dir / f"{config_id}.run{record['run']}.json", record)
            count += 1

    ablation_path = directory / "ablation.json"
    _write_json(ablation_path, result.ablation.model_dump(mode="json"))

    manifest_path = directory / "manifest.json"
    if result.ablation.manifest is not None:
        _write_json(manifest_path, result.ablation.manifest.model_dump(mode="json"))
    else:
        if manifest_path.exists():
            manifest_path.unlink()
        manifest_path = None
    return ablation_path, manifest_path, count


# -- report and verify -----------------------------------------------------


@app.command()
def report(
    claim_path: Path = typer.Argument(
        ..., exists=True, dir_okay=False, readable=True, help="Path to claim.yaml."
    ),
    out: Optional[Path] = typer.Option(
        None,
        "--out",
        help="Directory for report.md/html. The sweep is read from here when it holds "
        "one, otherwise from reports/<claim id> (default: reports/<claim id>).",
    ),
) -> None:
    """Render the verdict card from a recorded ablation sweep."""
    claim = _read_claim(claim_path)
    target = out if out is not None else Path("reports") / claim.id
    # The sweep belongs to the claim; --out chooses where the card is written.
    source = target if (target / "ablation.json").is_file() else Path("reports") / claim.id

    try:
        ablation = load_ablation(source, claim)
    except AblationError as exc:
        _die(str(exc))

    card = assemble(ablation)
    markdown_path, html_path, json_path = write_report(card, target)
    typer.echo(f"wrote {markdown_path}")
    typer.echo(f"wrote {html_path}")
    typer.echo(f"wrote {json_path}")
    if card.status is Status.NO_VERDICT:
        typer.echo(card.reason, err=True)
    if card.exit_code != 0:
        raise typer.Exit(card.exit_code)


def _load_report_card(path: Path) -> Report:
    """Read a card from a directory, report.json, or report.md.

    report.md carries the same JSON as report.json in a marked fenced block, so
    the single file a recipient is sent is sufficient to verify it.
    """
    if path.is_dir():
        json_candidate = path / "report.json"
        if json_candidate.is_file():
            return _parse_json_card(json_candidate)
        markdown_candidate = path / "report.md"
        if markdown_candidate.is_file():
            return _parse_markdown_card(markdown_candidate)
        _die(
            f"no report.json or report.md in {path}: pass a report directory, "
            "report.json, or report.md"
        )
    if path.suffix == ".json":
        return _parse_json_card(path)
    return _parse_markdown_card(path)


def _parse_json_card(path: Path) -> Report:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        _die(f"cannot read {path}: {exc}")
    try:
        return Report.model_validate_json(text)
    except ValidationError as exc:
        _die(f"{path}: not a usable report: {exc}")


def _parse_markdown_card(path: Path) -> Report:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        _die(f"cannot read {path}: {exc}")
    payload = extract_embedded_report(text)
    if payload is None:
        _die(
            f"{path}: no embedded card data (report.md written by 'injablat report' "
            "carries it); pass report.json or the report directory instead"
        )
    try:
        return Report.model_validate_json(payload)
    except ValidationError as exc:
        _die(f"{path}: embedded card data is not a usable report: {exc}")


@app.command()
def verify(
    report_path: Path = typer.Argument(
        ...,
        exists=True,
        readable=True,
        help="A report directory, report.json, or report.md with the embedded card data.",
    ),
    endpoint: Optional[str] = typer.Option(
        None, "--endpoint", help="Replay against this endpoint instead of the recorded one."
    ),
    runs: Optional[int] = typer.Option(
        None, "--runs", "-n", help="Dispatches per row (default: the card's own count)."
    ),
    config: Optional[Path] = _CONFIG_OPTION,
    fixture: bool = _FIXTURE_OPTION,
) -> None:
    """Re-dispatch a card's recorded requests and print MATCH or DRIFT per row."""
    card = _load_report_card(report_path)

    spec = ModelSpec.model_validate(
        {
            "name": card.model.name,
            "endpoint": endpoint or card.model.endpoint,
            "backend": card.model.backend,
            "revision": card.model.revision,
            "chat_template_sha": card.model.chat_template_sha,
            "chat_template_path": None,
        }
    )
    run_count = runs if runs is not None else card.runs_per_config
    if run_count < 1:
        _die(f"verify needs at least one dispatch per row, got {run_count}")

    settings = _run_settings(config)
    try:
        with build_backend(
            spec, fixture=_fixture_enabled(fixture), settings=settings
        ) as backend:
            result = verify_report(card, backend, run_count)
    except BackendError as exc:
        _die(str(exc))

    _print_verify(report_path, card, result, run_count)
    if result.exit_code != 0:
        raise typer.Exit(result.exit_code)


def _print_verify(source_path: Path, card: Report, result: VerifyResult, runs: int) -> None:
    model = card.model
    typer.echo(f"verify {source_path}")
    typer.echo(
        f"claim {card.claim_id}   model {model.name} @ {model.endpoint} "
        f"({model.backend}, revision {model.revision})"
    )
    typer.echo(f"runs per row {runs}")
    typer.echo("")
    typer.echo(f"  {'row':<32}{'expected':<14}{'observed':<14}{'status'}")
    for row in result.rows:
        typer.echo(
            f"  {row.label:<32}{_short_sha(row.expected_sha256):<14}"
            f"{_short_sha(row.observed_sha256):<14}{row.status.value}"
        )

    typer.echo("")
    if result.drifted:
        typer.echo(
            f"DRIFT {len(result.drifted)} of {len(result.rows)} rows differ"
        )
        for row in result.drifted:
            typer.echo(
                f"  {row.label}: expected {_short_sha(row.expected_sha256)}, "
                f"observed {_short_sha(row.observed_sha256)}"
                + (f" ({row.note})" if row.note else "")
            )
    elif result.matched:
        typer.echo(
            f"MATCH {len(result.matched)} of {len(result.rows)} rows "
            "(output hashes unchanged)"
        )
    else:
        typer.echo(
            f"UNDECIDED {len(result.undecided)} of {len(result.rows)} rows: "
            "no pinned output hash, so nothing was compared"
        )
    if result.undecided and (result.drifted or result.matched):
        typer.echo(
            f"{len(result.undecided)} row(s) recorded as UNDECIDED were not compared"
        )


def main() -> None:
    app()


if __name__ == "__main__":
    main()
