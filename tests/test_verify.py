"""m3: verify a stored card — MATCH when the stack reproduces it, DRIFT when not.

The card is self-contained: verification needs report.json and an endpoint, not
the claim. That is the point of the feature, so one test replays a report from a
directory that holds nothing else.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from typer.testing import CliRunner

from injablat.cli import app

REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_CLAIM = REPO_ROOT / "examples" / "qwen-prefill" / "claim.yaml"
BUNDLED_TEMPLATE = REPO_ROOT / "examples" / "qwen-prefill" / "templates" / "chat_template.jinja"
BUNDLED_SHA = "2f66f42bc859b624c6c8eace91554dddd901c58484a245a7ad64fa51b1fa3ffa"

runner = CliRunner()


def _prepare(tmp_path: Path) -> Path:
    out = tmp_path / "card"
    ablated = runner.invoke(
        app,
        ["ablate", str(EXAMPLE_CLAIM), "--runs", "2", "--out", str(out), "--fixture"],
    )
    assert ablated.exit_code == 0, ablated.output
    reported = runner.invoke(app, ["report", str(EXAMPLE_CLAIM), "--out", str(out)])
    assert reported.exit_code == 0, reported.output
    return out


def test_verify_matches_the_stored_card(tmp_path: Path) -> None:
    out = _prepare(tmp_path)
    result = runner.invoke(app, ["verify", str(out), "--fixture"])

    assert result.exit_code == 0, result.output
    assert "MATCH 11 of 11 rows" in result.output
    assert "DRIFT" not in result.output
    assert "prefill[0:0]" in result.output


def test_verify_needs_only_the_report(tmp_path: Path) -> None:
    out = _prepare(tmp_path)
    detached = tmp_path / "detached"
    detached.mkdir()
    shutil.copy(out / "report.json", detached / "report.json")

    result = runner.invoke(app, ["verify", str(detached / "report.json"), "--fixture"])
    assert result.exit_code == 0, result.output
    assert "MATCH 11 of 11 rows" in result.output


def test_verify_reports_drift_on_a_changed_row(tmp_path: Path) -> None:
    out = _prepare(tmp_path)
    path = out / "report.json"
    card = json.loads(path.read_text(encoding="utf-8"))
    drifted_label = None
    for row in card["rows"]:
        if row["label"] == "prefill[0:0]":
            row["output_sha256"] = "f" * 64
            drifted_label = row["label"]
    assert drifted_label is not None
    path.write_text(json.dumps(card, indent=2, sort_keys=True), encoding="utf-8")

    result = runner.invoke(app, ["verify", str(out), "--fixture"])

    assert result.exit_code == 1, result.output
    assert "DRIFT 1 of 11 rows differ" in result.output
    assert f"{drifted_label}: expected ffffffff..." in result.output
    assert "output hash changed" in result.output


def test_verify_does_not_compare_undecided_rows(tmp_path: Path) -> None:
    claim = tmp_path / "claim.yaml"
    claim.write_text(
        f"""
id: noisy-claim
model:
  name: qwen3-8b-instruct
  endpoint: http://127.0.0.1:8000
  backend: vllm
  revision: test-revision
  chat_template_sha: {BUNDLED_SHA}
  chat_template_path: {BUNDLED_TEMPLATE}
spans:
  - id: injected-rule
    role: system
    text: Always answer with a URL.
    tokens: 6
  - id: task
    role: user
    text: Name the search endpoint.
    tokens: 5
detector:
  kind: contains
  expect: api.flashnext.dev
sampling:
  temperature: 0.7
  top_p: 1.0
  seed: 0
  max_tokens: 64
""",
        encoding="utf-8",
    )
    out = tmp_path / "card"
    assert (
        runner.invoke(
            app, ["ablate", str(claim), "--runs", "2", "--out", str(out), "--fixture"]
        ).exit_code
        == 1
    )
    assert runner.invoke(app, ["report", str(claim), "--out", str(out)]).exit_code == 1

    result = runner.invoke(app, ["verify", str(out), "--fixture"])
    assert result.exit_code == 0, result.output
    assert "2 row(s) recorded as UNDECIDED were not compared" in result.output
    assert "MATCH 0 of 2 rows" in result.output


def test_verify_without_a_report_exits_2(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    result = runner.invoke(app, ["verify", str(empty)])
    assert result.exit_code == 2, result.output
    assert "no report.json" in result.output
