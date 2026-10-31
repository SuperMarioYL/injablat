"""The verdict card: report.md for reading, report.html for sharing.

Both are renders of one ``Report``. Nothing here re-runs or re-derives evidence;
the rows and hashes come from the recorded sweep, which is why every line of a
card can be traced back to a concrete HTTP exchange in ``runs/``.

The HTML is self-contained: inline CSS, no scripts, no external fonts. It exists
so the card can be attached to a review, and its verification panel names the
exact command that turns it into MATCH or DRIFT.
"""

from __future__ import annotations

import html
import json
from pathlib import Path

from .claim import Detector
from .manifest import manifest_sha256
from .verdict import Report


def write_report(report: Report, directory: Path) -> tuple[Path, Path, Path]:
    """Write report.md, report.html and report.json into ``directory``."""
    directory.mkdir(parents=True, exist_ok=True)
    markdown_path = directory / "report.md"
    html_path = directory / "report.html"
    json_path = directory / "report.json"

    markdown_path.write_text(render_markdown(report), encoding="utf-8")
    html_path.write_text(render_html(report), encoding="utf-8")
    json_path.write_text(
        json.dumps(report.model_dump(mode="json"), indent=2, sort_keys=True, ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )
    return markdown_path, html_path, json_path


# -- shared formatting -----------------------------------------------------


def detector_text(detector: Detector) -> str:
    if detector.kind == "json_field":
        return f"json_field `{detector.path}` == `{detector.expect}`"
    return f"{detector.kind} `{detector.expect}`"


def _short(sha: str | None) -> str:
    if sha is None:
        return "(disagree)"
    return f"{sha[:8]}..."


def _cell(value: str) -> str:
    """Keep a markdown table cell on one line and out of the next column."""
    return " ".join(value.split()).replace("|", "\\|")


def _yn(value: bool | None) -> str:
    if value is True:
        return "YES"
    if value is False:
        return "NO"
    return "UNDECIDED"


def _minimal_text(report: Report) -> str:
    text = report.minimal_text
    return text if text is not None else "(not established)"


def _prefill_question(report: Report) -> str | None:
    value = report.survives_without_prefill()
    if value is None:
        return None
    return _yn(value)


# -- markdown --------------------------------------------------------------


def render_markdown(report: Report) -> str:
    model = report.model
    sampling = report.sampling
    lines: list[str] = [
        f"# Injection ablation — {report.claim_id}",
        "",
        f"Detector: **{detector_text(report.detector)}**",
        "",
        "| field | value |",
        "| --- | --- |",
        f"| model | {_cell(model.name)} |",
        f"| backend | {_cell(model.backend)} |",
        f"| endpoint | {_cell(model.endpoint)} |",
        f"| revision | {_cell(model.revision)} |",
        f"| endpoint build | {_cell(model.endpoint_build)} |",
        f"| tokenizer | {_cell(model.tokenizer_name)} |",
        f"| chat template | `sha256:{report.chat_template.sha256}` ({report.chat_template.source}) |",
        f"| sampling | temperature {sampling.temperature}, top_p {sampling.top_p}, "
        f"seed {sampling.seed}, max_tokens {sampling.max_tokens} |",
        f"| runs per configuration | {report.runs_per_config} |",
        f"| determinism | {report.determinism.value} — {_cell(report.determinism_detail)} |",
    ]
    if report.manifest is not None:
        lines.append(f"| manifest | `sha256:{manifest_sha256(report.manifest)}` |")
    else:
        lines.append("| manifest | not pinned (runs disagree) |")

    lines += [
        "",
        "## Verdict",
        "",
        report.reason + ".",
        "",
        f"minimal sufficient injection  **{_cell(_minimal_text(report))}**",
    ]
    prefill = _prefill_question(report)
    if prefill is not None:
        lines.append(f"survives without prefill      **{prefill}**")
    lines += [
        "",
        "| remove span | behavior | output sha | role |",
        "| --- | --- | --- | --- |",
    ]
    for row in report.rows:
        lines.append(
            f"| {_cell(row.label)} | {row.behavior.value} | {_short(row.output_sha256)} "
            f"| {row.role.value} |"
        )

    lines += [
        "",
        "## What the roles mean",
        "",
        "| role | meaning |",
        "| --- | --- |",
        "| baseline | the user task with every injected span removed |",
        "| reproduces | the full recorded injection: the behaviour is there to explain |",
        "| necessary | removing this span at its turn (for the prefill: truncating it to 0) stops the behaviour |",
        "| redundant | the behaviour still appears without this span at its turn |",
        "| insufficient | a declared prefill prefix that does not yet restore the behaviour |",
        "| sufficient | a declared prefill prefix that does restore the behaviour |",
        "| verified | the final set, dispatched once more before being called minimal |",
        "| undecided | the repeats disagreed; this row carries no conclusion |",
        "",
        "## One solution, not the only solution",
        "",
        "The minimal set above is one solution under this deletion order, not the unique one.",
        "The search deletes one span at a time, so a span that only becomes necessary because",
        "an earlier one was removed stays necessary here. Every tested configuration is listed",
        "below with the spans it kept and removed, so the same claim can be re-run under a",
        "different order and the two cards compared. While the intersection of two minimal sets",
        "is non-empty, the shared spans are still load-bearing.",
        "",
        "Deletion order: " + ", ".join(f"`{span_id}`" for span_id in report.deletion_order) + ".",
        "",
        "## Search path",
        "",
        "| row | tested | kept | removed | prompt tokens |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in report.rows:
        lines.append(
            f"| {_cell(row.label)} | {_cell(row.tested or '—')} "
            f"| {_cell(', '.join(row.keep) or '—')} | {_cell(', '.join(row.remove) or '—')} "
            f"| {row.prompt_tokens} |"
        )

    lines += ["", "## Manifest", ""]
    if report.manifest is not None:
        lines += [
            "```json",
            json.dumps(
                report.manifest.model_dump(mode="json"),
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
            ),
            "```",
        ]
    else:
        lines.append("No manifest is pinned: the full configuration was not reproducible.")

    lines += [
        "",
        "## Reproduce or refute this card",
        "",
        f"    injablat verify reports/{report.claim_id}",
        "",
        "`verify` re-dispatches the request bytes recorded for every row against the same",
        "stack and prints MATCH or DRIFT per row. MATCH means the output hash is unchanged,",
        "not that the claim is endorsed. A single DRIFT row means the endpoint is no longer",
        "the stack this card was measured on, or the weights changed.",
        "",
    ]
    return "\n".join(lines)


# -- html ------------------------------------------------------------------

_CSS = """
:root {
  color-scheme: light;
  --ink: #1c1e21;
  --muted: #5b616e;
  --line: #d8dbe0;
  --panel: #f7f8fa;
  --good: #1c6b3c;
  --good-bg: #e8f4ec;
  --bad: #8a2b1f;
  --bad-bg: #fbebe8;
  --warn: #6b4e12;
  --warn-bg: #f7f0dc;
  --grey: #4a4e57;
  --grey-bg: #eceef1;
}
* { box-sizing: border-box; }
body {
  margin: 0;
  padding: 40px 24px 64px;
  background: #fff;
  color: var(--ink);
  font: 15px/1.55 -apple-system, BlinkMacSystemFont, "Segoe UI", "Helvetica Neue", Arial,
        "Noto Sans SC", sans-serif;
}
main { max-width: 1040px; margin: 0 auto; }
h1 { font-size: 28px; line-height: 1.2; margin: 0 0 6px; letter-spacing: -0.01em; }
h2 { font-size: 17px; margin: 36px 0 10px; padding-bottom: 6px; border-bottom: 1px solid var(--line); }
p { margin: 8px 0; }
code, pre, .sha {
  font-family: "SF Mono", Menlo, Consolas, "Liberation Mono", monospace;
  font-size: 12.5px;
}
code { background: var(--panel); padding: 1px 5px; border-radius: 4px; }
pre {
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 6px;
  padding: 14px 16px;
  overflow-x: auto;
}
.sub { color: var(--muted); margin: 0 0 22px; }
.badge {
  display: inline-block;
  padding: 3px 10px;
  border-radius: 999px;
  font-size: 12px;
  font-weight: 650;
  letter-spacing: 0.04em;
  vertical-align: 3px;
  margin-left: 10px;
}
.badge.VERDICT { background: var(--good-bg); color: var(--good); }
.badge.DOES_NOT_REPRODUCE { background: var(--warn-bg); color: var(--warn); }
.badge.NO_VERDICT { background: var(--grey-bg); color: var(--grey); }
.grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(230px, 1fr));
  gap: 1px;
  background: var(--line);
  border: 1px solid var(--line);
  border-radius: 6px;
  overflow: hidden;
}
.grid div { background: #fff; padding: 10px 13px; }
.grid dt { color: var(--muted); font-size: 11.5px; letter-spacing: 0.05em; text-transform: uppercase; margin: 0 0 3px; }
.grid dd { margin: 0; word-break: break-word; }
.verdict-box {
  border: 1px solid var(--line);
  border-left: 4px solid var(--ink);
  border-radius: 6px;
  padding: 14px 18px;
  margin: 12px 0 18px;
  background: var(--panel);
}
.verdict-box .big { font-size: 17px; font-weight: 650; margin: 4px 0; }
.verdict-box .line { margin: 6px 0; }
table { border-collapse: collapse; width: 100%; }
th, td { text-align: left; padding: 8px 10px; border-bottom: 1px solid var(--line); vertical-align: top; }
th { font-size: 11.5px; letter-spacing: 0.05em; text-transform: uppercase; color: var(--muted); font-weight: 600; }
tbody tr:last-child td { border-bottom: none; }
.wrap { border: 1px solid var(--line); border-radius: 6px; overflow: hidden; }
.b-FIRED { color: var(--good); font-weight: 600; }
.b-NOT_FIRED { color: var(--bad); font-weight: 600; }
.b-UNDECIDED { color: var(--grey); font-weight: 600; }
.r-necessary, .r-reproduces, .r-verified, .r-sufficient { color: var(--good); }
.r-redundant, .r-insufficient { color: var(--warn); }
.r-undecided { color: var(--grey); }
.r-baseline { color: var(--muted); }
.note { color: var(--muted); font-size: 13.5px; }
.verify {
  border: 1px solid var(--line);
  border-radius: 6px;
  padding: 16px 18px;
  margin-top: 10px;
}
.verify .cmd { display: block; margin: 10px 0; }
.verify table { margin-top: 8px; }
.kv { margin: 4px 0; }
footer { margin-top: 40px; color: var(--muted); font-size: 12.5px; }
"""


def _e(value: object) -> str:
    return html.escape(str(value))


def _detector_html(detector: Detector) -> str:
    if detector.kind == "json_field":
        return (
            f'json_field <code>{_e(detector.path or "")}</code> == '
            f"<code>{_e(detector.expect)}</code>"
        )
    return f"{_e(detector.kind)} <code>{_e(detector.expect)}</code>"


def _row_class(role: str) -> str:
    return f"r-{role}"


def _html_rows(report: Report) -> str:
    rows = []
    for row in report.rows:
        rows.append(
            "<tr>"
            f"<td><code>{_e(row.label)}</code></td>"
            f'<td class="b-{_e(row.behavior.value)}">{_e(row.behavior.value)}</td>'
            f'<td class="sha">{_e(_short(row.output_sha256))}</td>'
            f'<td class="{_row_class(row.role.value)}">{_e(row.role.value)}</td>'
            "</tr>"
        )
    return "\n".join(rows)


def _html_search_rows(report: Report) -> str:
    rows = []
    for row in report.rows:
        rows.append(
            "<tr>"
            f"<td><code>{_e(row.label)}</code></td>"
            f"<td>{_e(row.tested or '—')}</td>"
            f"<td>{_e(', '.join(row.keep) or '—')}</td>"
            f"<td>{_e(', '.join(row.remove) or '—')}</td>"
            f"<td>{row.prompt_tokens}</td>"
            "</tr>"
        )
    return "\n".join(rows)


def _html_summary(report: Report) -> str:
    model = report.model
    sampling = report.sampling
    items = [
        ("model", f"{_e(model.name)}"),
        ("backend", _e(model.backend)),
        ("endpoint", f"<code>{_e(model.endpoint)}</code>"),
        ("revision", _e(model.revision)),
        ("endpoint build", _e(model.endpoint_build)),
        ("tokenizer", _e(model.tokenizer_name)),
        ("chat template", f'<span class="sha">sha256:{_e(report.chat_template.sha256)}</span> ({_e(report.chat_template.source)})'),
        (
            "sampling",
            _e(
                f"temperature {sampling.temperature}, top_p {sampling.top_p}, "
                f"seed {sampling.seed}, max_tokens {sampling.max_tokens}"
            ),
        ),
        ("runs per configuration", str(report.runs_per_config)),
        (
            "manifest",
            f'<span class="sha">sha256:{_e(manifest_sha256(report.manifest))}</span>'
            if report.manifest is not None
            else "not pinned (runs disagree)",
        ),
        (
            "determinism",
            f"{_e(report.determinism.value)} — {_e(report.determinism_detail)}",
        ),
    ]
    body = "\n".join(
        f"<div><dt>{_e(key)}</dt><dd>{value}</dd></div>" for key, value in items
    )
    return f'<dl class="grid">{body}</dl>'


def render_html(report: Report) -> str:
    prefill = _prefill_question(report)
    minimal = _minimal_text(report)
    manifest_json = (
        json.dumps(
            report.manifest.model_dump(mode="json"), indent=2, sort_keys=True, ensure_ascii=False
        )
        if report.manifest is not None
        else "No manifest is pinned: the full configuration was not reproducible."
    )
    prefill_line = (
        f'<div class="line">survives without prefill &nbsp;<strong>{_e(prefill)}</strong></div>'
        if prefill is not None
        else ""
    )

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Injection ablation — {_e(report.claim_id)}</title>
<style>{_CSS}</style>
</head>
<body>
<main>
<h1>Injection ablation — {_e(report.claim_id)}<span class="badge {_e(report.status.value)}">{_e(report.status.value)}</span></h1>
<p class="sub">Detector: <strong>{_detector_html(report.detector)}</strong></p>

{_html_summary(report)}

<h2>Verdict</h2>
<div class="verdict-box">
  <div class="note">{_e(report.reason)}</div>
  <div class="big">minimal sufficient injection &nbsp;{_e(minimal)}</div>
  {prefill_line}
</div>

<div class="wrap">
<table>
<thead><tr><th>remove span</th><th>behavior</th><th>output sha</th><th>role</th></tr></thead>
<tbody>
{_html_rows(report)}
</tbody>
</table>
</div>

<h2>What the roles mean</h2>
<div class="wrap">
<table>
<thead><tr><th>role</th><th>meaning</th></tr></thead>
<tbody>
<tr><td class="r-baseline">baseline</td><td>the user task with every injected span removed</td></tr>
<tr><td class="r-reproduces">reproduces</td><td>the full recorded injection: the behaviour is there to explain</td></tr>
<tr><td class="r-necessary">necessary</td><td>removing this span at its turn (for the prefill: truncating it to 0) stops the behaviour</td></tr>
<tr><td class="r-redundant">redundant</td><td>the behaviour still appears without this span at its turn</td></tr>
<tr><td class="r-insufficient">insufficient</td><td>a declared prefill prefix that does not yet restore the behaviour</td></tr>
<tr><td class="r-sufficient">sufficient</td><td>a declared prefill prefix that does restore the behaviour</td></tr>
<tr><td class="r-verified">verified</td><td>the final set, dispatched once more before being called minimal</td></tr>
<tr><td class="r-undecided">undecided</td><td>the repeats disagreed; this row carries no conclusion</td></tr>
</tbody>
</table>
</div>

<h2>One solution, not the only solution</h2>
<p>The minimal set above is one solution under this deletion order, not the unique one.
The search deletes one span at a time, so a span that only becomes necessary because an
earlier one was removed stays necessary here. Every tested configuration is listed below,
so the same claim can be re-run under a different order and the two cards compared. While
the intersection of two minimal sets is non-empty, the shared spans are still load-bearing.</p>
<p class="note">Deletion order: {_e(", ".join(report.deletion_order))}.</p>

<h2>Search path</h2>
<div class="wrap">
<table>
<thead><tr><th>row</th><th>tested</th><th>kept</th><th>removed</th><th>prompt tokens</th></tr></thead>
<tbody>
{_html_search_rows(report)}
</tbody>
</table>
</div>

<h2>Verify this card</h2>
<div class="verify">
  <div class="kv">Another machine re-dispatches the recorded request bytes and compares output hashes row by row.</div>
  <code class="cmd">injablat verify reports/{_e(report.claim_id)}</code>
  <div class="kv"><strong>MATCH</strong> — every pinned output hash is unchanged.</div>
  <div class="kv"><strong>DRIFT</strong> — at least one row's output hash changed; the endpoint is no longer
  the stack this card was measured on, or the weights moved.</div>
  <div class="kv note">A MATCH is not an endorsement of the claim: it only means the ablation is
  reproducible. {len(report.rows)} row(s) are pinned in this card.</div>
</div>

<h2>Manifest</h2>
<pre>{_e(manifest_json)}</pre>

<footer>Verified by request bytes and output hashes, not by trust. Undecided configurations carry no conclusion.</footer>
</main>
</body>
</html>
"""
