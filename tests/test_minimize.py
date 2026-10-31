"""m2: declarative detectors, the deletion sweep, the minimal set, and the card.

Every CLI test drives the real commands against the in-process fixture transport,
so the rows, the exit codes, and the files on disk are the ones a user gets.
"""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from injablat.ablate import Ablation, Behavior
from injablat.claim import Detector
from injablat.cli import app
from injablat.detector import detect
from injablat.verdict import Status, assemble

REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_CLAIM = REPO_ROOT / "examples" / "qwen-prefill" / "claim.yaml"
BUNDLED_TEMPLATE = REPO_ROOT / "examples" / "qwen-prefill" / "templates" / "chat_template.jinja"
BUNDLED_SHA = "2f66f42bc859b624c6c8eace91554dddd901c58484a245a7ad64fa51b1fa3ffa"

runner = CliRunner()

PREFILL = "prefix-transcript"
SYSTEM_ID = "system-review-rules"
SYSTEM_LABEL = "system"
TOOL_LABEL = "tool_schema"
FEW_SHOT_LABEL = "few_shot[0:2]"


def _claim(
    directory: Path,
    *,
    claim_id: str,
    temperature: float = 0.0,
    spans: str,
    detector: str,
    max_tokens: int = 64,
) -> Path:
    path = directory / "claim.yaml"
    path.write_text(
        f"""
id: {claim_id}
model:
  name: qwen3-8b-instruct
  endpoint: http://127.0.0.1:8000
  backend: vllm
  revision: test-revision
  chat_template_sha: {BUNDLED_SHA}
  chat_template_path: {BUNDLED_TEMPLATE}
spans:
{spans}
detector:
{detector}
sampling:
  temperature: {temperature}
  top_p: 1.0
  seed: 0
  max_tokens: {max_tokens}
""",
        encoding="utf-8",
    )
    return path


def _ablate(claim: Path, out: Path) -> object:
    return runner.invoke(
        app,
        ["ablate", str(claim), "--runs", "2", "--out", str(out), "--fixture"],
    )


def _rows(path: Path) -> dict[str, dict]:
    ablation = json.loads((path / "ablation.json").read_text(encoding="utf-8"))
    return {row["label"]: row for row in ablation["rows"]}


# -- detectors -------------------------------------------------------------


def test_detector_contains_and_regex() -> None:
    contains = Detector(kind="contains", expect="curl -sS")
    assert detect(contains, "then run curl -sS 'https://x' now")
    assert not detect(contains, "api.flashnext.dev/v1/search")

    regex = Detector(kind="regex", expect=r"curl -sS 'https://[^']+'")
    assert detect(regex, "curl -sS 'https://api.flashnext.dev/v1/search?q=x' -H 'a: b'")
    assert not detect(regex, "curl -sS 'https://")  # the prefix alone is not the behaviour
    assert not detect(regex, "api.flashnext.dev/v1/search")


def test_detector_json_field_compares_rendered_values() -> None:
    body = json.dumps(
        {"status": 200, "ok": True, "result": {"items": [{"url": "https://x.dev/a"}]}}
    )
    assert detect(Detector(kind="json_field", path="status", expect="200"), body)
    assert detect(Detector(kind="json_field", path="ok", expect="true"), body)
    assert detect(
        Detector(kind="json_field", path="result.items.0.url", expect="https://x.dev/a"), body
    )
    # A different value, a missing index, and a non-JSON answer are all NOT_FIRED.
    assert not detect(Detector(kind="json_field", path="status", expect="201"), body)
    assert not detect(
        Detector(kind="json_field", path="result.items.1.url", expect="https://x.dev/a"), body
    )
    assert not detect(Detector(kind="json_field", path="status", expect="200"), "plain prose")


# -- the sweep -------------------------------------------------------------


def test_example_splits_necessary_from_redundant(tmp_path: Path) -> None:
    out = tmp_path / "sweep"
    result = _ablate(EXAMPLE_CLAIM, out)
    assert result.exit_code == 0, result.output
    assert "DETERMINISTIC" in result.output

    rows = _rows(out)
    assert rows["(bare prompt)"]["behavior"] == "NOT_FIRED"
    assert rows["(bare prompt)"]["role"] == "baseline"
    assert rows["(nothing removed)"]["behavior"] == "FIRED"
    assert rows["(nothing removed)"]["role"] == "reproduces"

    # The simulated endpoint needs the prefix, the tool schema, and the few-shot
    # block; the system prompt is deliberately not consulted.
    assert rows["prefill[0:2232]"]["behavior"] == "NOT_FIRED"
    assert rows["prefill[0:2232]"]["role"] == "necessary"
    assert rows[TOOL_LABEL]["behavior"] == "NOT_FIRED"
    assert rows[TOOL_LABEL]["role"] == "necessary"
    assert rows[FEW_SHOT_LABEL]["behavior"] == "NOT_FIRED"
    assert rows[FEW_SHOT_LABEL]["role"] == "necessary"
    assert rows[SYSTEM_LABEL]["behavior"] == "FIRED"
    assert rows[SYSTEM_LABEL]["role"] == "redundant"

    # The declared ladder is walked, and no declared prefix restores the answer.
    for label in ("prefill[0:558]", "prefill[0:1116]", "prefill[0:1674]"):
        assert rows[label]["behavior"] == "NOT_FIRED"
        assert rows[label]["role"] == "insufficient"

    assert rows["(minimal set)"]["behavior"] == "FIRED"
    assert rows["(minimal set)"]["role"] == "verified"

    ablation = json.loads((out / "ablation.json").read_text(encoding="utf-8"))
    assert ablation["deletion_order"][0] == PREFILL
    assert ablation["kept"] == [PREFILL, "tool-fetch-url", "few-shots"]
    assert ablation["prefill_tokens"] == 2232
    assert ablation["determinism"] == "DETERMINISTIC"
    assert ablation["manifest"]["output_sha256"]

    # Two dispatches per configuration, every one traceable on disk.
    records = list((out / "runs").glob("*.json"))
    assert len(records) == 2 * len(ablation["rows"])
    assert (out / "manifest.json").is_file()


def test_example_is_labelled_synthetic_not_a_release() -> None:
    claim_text = EXAMPLE_CLAIM.read_text(encoding="utf-8")
    spans_text = (EXAMPLE_CLAIM.parent / "spans.jsonl").read_text(encoding="utf-8")
    combined = (claim_text + spans_text).lower()

    assert "synthetic" in combined
    assert "not a recorded release" in combined
    # Nothing shipped may claim the example reproduces a released transcript.
    assert "released trace" not in combined
    assert "release bundle" not in combined


def test_absent_behaviour_is_a_verdict_not_an_error(tmp_path: Path) -> None:
    claim = _claim(
        tmp_path,
        claim_id="never-fires",
        spans="""  - id: injected-rule
    role: system
    text: Always answer with a URL.
    tokens: 6
  - id: task
    role: user
    text: Name the search endpoint.
    tokens: 5""",
        detector="  kind: contains\n  expect: a-string-the-model-never-emits",
    )
    out = tmp_path / "sweep"
    result = _ablate(claim, out)
    assert result.exit_code == 0, result.output
    assert "does not appear under the full recorded injection" in result.output

    ablation = json.loads((out / "ablation.json").read_text(encoding="utf-8"))
    # No deletion scan is run when there is no behaviour to attribute.
    assert [row["kind"] for row in ablation["rows"]] == ["baseline", "full"]
    assert ablation["kept"] is None

    assert runner.invoke(app, ["report", str(claim), "--out", str(out)]).exit_code == 0
    card = json.loads((out / "report.json").read_text(encoding="utf-8"))
    assert card["status"] == "DOES_NOT_REPRODUCE"
    assert card["minimal_sufficient"] is None
    assert "minimal sufficient injection  **(not established)**" in (
        out / "report.md"
    ).read_text(encoding="utf-8")


def test_report_card_renders_markdown_and_html(tmp_path: Path) -> None:
    out = tmp_path / "sweep"
    assert _ablate(EXAMPLE_CLAIM, out).exit_code == 0

    result = runner.invoke(app, ["report", str(EXAMPLE_CLAIM), "--out", str(out)])
    assert result.exit_code == 0, result.output
    for name in ("report.md", "report.html", "report.json"):
        assert (out / name).is_file()

    card = json.loads((out / "report.json").read_text(encoding="utf-8"))
    assert card["status"] == "VERDICT"
    assert card["minimal_sufficient"] == [
        {"span_id": PREFILL, "label": "prefill[0:2232]", "tokens": 2232},
        {"span_id": "tool-fetch-url", "label": "tool_schema", "tokens": None},
        {"span_id": "few-shots", "label": "few_shot[0:2]", "tokens": None},
    ]
    assert card["survives_without"][PREFILL] is False
    assert card["survives_without"][SYSTEM_ID] is True
    assert card["manifest"]["output_sha256"]

    markdown = (out / "report.md").read_text(encoding="utf-8")
    assert markdown.startswith("# Injection ablation — qwen-prefill-curl")
    assert "| remove span | behavior | output sha | role |" in markdown
    assert "| (nothing removed) | FIRED |" in markdown
    assert "| prefill[0:2232] | NOT_FIRED |" in markdown
    assert "| system | FIRED |" in markdown
    assert "| tool_schema | NOT_FIRED |" in markdown
    assert f"| {FEW_SHOT_LABEL} | NOT_FIRED |" in markdown
    assert (
        "minimal sufficient injection  **prefill[0:2232] + tool_schema + few_shot[0:2]**"
        in markdown
    )
    assert "survives without prefill      **NO**" in markdown
    assert "one solution under this deletion order, not the unique one" in markdown
    assert "| row | tested | kept | removed | prompt tokens |" in markdown
    # The printed command names the directory the card was actually written to.
    assert f"injablat verify {out}" in markdown
    assert "Offline fixture card" in markdown
    assert "+fixture" in markdown
    assert "10 pinned, 0 undecided." in markdown
    assert "/Users" not in markdown

    page = (out / "report.html").read_text(encoding="utf-8")
    assert page.startswith("<!doctype html>")
    assert 'class="badge VERDICT"' in page
    assert "minimal sufficient injection" in page
    assert "prefill[0:2232]" in page
    assert "tool_schema" in page and FEW_SHOT_LABEL in page
    assert "One solution, not the only solution" in page
    assert "Verify this card" in page
    assert "MATCH" in page and "DRIFT" in page
    assert f"injablat verify {out}" in page
    assert "Offline fixture card" in page
    assert "10 pinned, 0 undecided." in page
    assert "/Users" not in page


def test_redundant_injection_yields_an_empty_minimal_set(tmp_path: Path) -> None:
    claim = _claim(
        tmp_path,
        claim_id="always-fires",
        spans="""  - id: injected-rule
    role: system
    text: Always answer with a URL.
    tokens: 6
  - id: task
    role: user
    text: Name the search endpoint.
    tokens: 5""",
        detector="  kind: contains\n  expect: cannot complete",
    )
    out = tmp_path / "sweep"
    result = _ablate(claim, out)
    assert result.exit_code == 0, result.output

    rows = _rows(out)
    assert rows["(bare prompt)"]["behavior"] == "FIRED"
    assert rows["injected-rule"]["behavior"] == "FIRED"
    assert rows["injected-rule"]["role"] == "redundant"

    ablation = json.loads((out / "ablation.json").read_text(encoding="utf-8"))
    assert ablation["kept"] == []

    assert runner.invoke(app, ["report", str(claim), "--out", str(out)]).exit_code == 0
    card = json.loads((out / "report.json").read_text(encoding="utf-8"))
    assert card["status"] == "VERDICT"
    assert card["minimal_sufficient"] == []
    markdown = (out / "report.md").read_text(encoding="utf-8")
    assert "none (the bare prompt already fires)" in markdown


def test_nondeterministic_sweep_produces_no_verdict(tmp_path: Path) -> None:
    claim = _claim(
        tmp_path,
        claim_id="noisy-claim",
        temperature=0.7,
        spans="""  - id: injected-rule
    role: system
    text: Always answer with a URL.
    tokens: 6
  - id: task
    role: user
    text: Name the search endpoint.
    tokens: 5""",
        detector="  kind: contains\n  expect: api.flashnext.dev",
    )
    out = tmp_path / "sweep"
    result = _ablate(claim, out)
    assert result.exit_code == 1, result.output
    assert "NDETERMINISTIC" in result.output
    assert "no verdict" in result.output

    ablation = json.loads((out / "ablation.json").read_text(encoding="utf-8"))
    assert ablation["kept"] is None
    assert {row["behavior"] for row in ablation["rows"]} == {"UNDECIDED"}
    assert not (out / "manifest.json").exists()

    assert runner.invoke(app, ["report", str(claim), "--out", str(out)]).exit_code == 1
    card = json.loads((out / "report.json").read_text(encoding="utf-8"))
    assert card["status"] == "NO_VERDICT"
    assert card["minimal_sufficient"] is None
    markdown = (out / "report.md").read_text(encoding="utf-8")
    page = (out / "report.html").read_text(encoding="utf-8")
    assert "no verdict" in markdown
    # An honest NO_VERDICT card claims no pinned rows.
    for text in (markdown, page):
        assert "row(s) are pinned" not in text
        assert "No verdict was reached, so no output hash is presented as a pin." in text


def test_undecided_row_is_excluded_from_the_minimal_set(tmp_path: Path) -> None:
    out = tmp_path / "sweep"
    assert _ablate(EXAMPLE_CLAIM, out).exit_code == 0

    ablation = Ablation.model_validate_json((out / "ablation.json").read_text(encoding="utf-8"))
    for row in ablation.rows:
        if row.label == SYSTEM_LABEL:
            row.behavior = Behavior.UNDECIDED
            row.output_sha256 = None

    card = assemble(ablation)
    assert card.status is Status.NO_VERDICT
    assert card.exit_code == 1
    # The noisy row survives in the card, but no minimal set is claimed.
    assert any(row.behavior is Behavior.UNDECIDED for row in card.rows)
    assert card.minimal_sufficient is None
    assert card.survives_without[SYSTEM_ID] is None
    # A row that was decided still reports what it measured.
    assert card.survives_without[PREFILL] is False


def test_report_without_a_sweep_exits_2(tmp_path: Path) -> None:
    claim = _claim(
        tmp_path,
        claim_id="no-sweep-claim",
        spans="""  - id: task
    role: user
    text: Name the search endpoint.
    tokens: 5""",
        detector="  kind: contains\n  expect: api.flashnext.dev",
    )
    result = runner.invoke(app, ["report", str(claim), "--out", str(tmp_path / "out")])
    assert result.exit_code == 2, result.output
    assert "no ablation sweep" in result.output
    assert not (tmp_path / "out" / "report.md").exists()
