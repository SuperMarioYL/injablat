"""m1 gates: determinism, template pinning, and the no-pin-on-disagreement rule.

Every test drives the real CLI against the in-process fixture transport, so the
exit codes, the artifacts on disk, and the stash of per-run hashes are the same
ones a user would get.
"""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from injablat.cli import app

REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_CLAIM = REPO_ROOT / "examples" / "qwen-prefill" / "claim.yaml"
BUNDLED_TEMPLATE = REPO_ROOT / "examples" / "qwen-prefill" / "templates" / "chat_template.jinja"
BUNDLED_SHA = "2f66f42bc859b624c6c8eace91554dddd901c58484a245a7ad64fa51b1fa3ffa"

runner = CliRunner()


def _write_claim(
    directory: Path,
    *,
    template_path: str,
    template_sha: str,
    temperature: float = 0.0,
) -> Path:
    claim = directory / "claim.yaml"
    claim.write_text(
        f"""
id: gate-check
model:
  name: qwen3-8b-instruct
  endpoint: http://127.0.0.1:8000
  backend: vllm
  revision: test-revision
  chat_template_sha: {template_sha}
  chat_template_path: {template_path}
spans:
  - id: user-task
    role: user
    text: Complete the partial curl command.
    tokens: 6
detector:
  kind: contains
  expect: curl
sampling:
  temperature: {temperature}
  top_p: 1.0
  seed: 0
  max_tokens: 32
""",
        encoding="utf-8",
    )
    return claim


def test_deterministic_replay_pins_manifest(tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        ["replay", str(EXAMPLE_CLAIM), "--runs", "2", "--out", str(tmp_path), "--fixture"],
    )

    assert result.exit_code == 0, result.output
    assert "DETERMINISTIC" in result.output

    replay = json.loads((tmp_path / "replay.json").read_text(encoding="utf-8"))
    assert replay["pinned"] is True
    assert isinstance(replay["manifest_sha256"], str) and len(replay["manifest_sha256"]) == 64
    assert len(replay["runs"]) == 2

    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["tokenizer_name"]
    assert len(manifest["tokenizer_probe_sha256"]) == 64
    assert len(manifest["prompt_token_sha256"]) == 64


def test_nondeterministic_replay_exits_1_and_leaves_no_pin(tmp_path: Path) -> None:
    claim = _write_claim(
        tmp_path,
        template_path=str(BUNDLED_TEMPLATE),
        template_sha=BUNDLED_SHA,
        temperature=0.7,
    )

    result = runner.invoke(
        app,
        ["replay", str(claim), "--runs", "2", "--out", str(tmp_path / "out"), "--fixture"],
    )

    assert result.exit_code == 1, result.output
    assert "NONDETERMINISTIC" in result.output

    replay = json.loads((tmp_path / "out" / "replay.json").read_text(encoding="utf-8"))
    assert replay["pinned"] is False
    assert replay["manifest_sha256"] is None
    assert len(replay["runs"]) == 2
    assert all(run["output_sha256"] for run in replay["runs"])
    assert not (tmp_path / "out" / "manifest.json").exists()


def test_missing_template_exits_2(tmp_path: Path) -> None:
    claim = _write_claim(
        tmp_path,
        template_path="templates/absent.jinja",
        template_sha=BUNDLED_SHA,
    )

    result = runner.invoke(
        app,
        ["replay", str(claim), "--out", str(tmp_path / "out"), "--fixture"],
    )

    assert result.exit_code == 2, result.output
    assert "cannot read chat template" in result.output
    assert not (tmp_path / "out" / "manifest.json").exists()


def test_template_hash_mismatch_exits_2(tmp_path: Path) -> None:
    drifted = tmp_path / "drifted.jinja"
    drifted.write_text("{{ messages }}", encoding="utf-8")
    claim = _write_claim(
        tmp_path,
        template_path=str(drifted),
        template_sha=BUNDLED_SHA,
    )

    result = runner.invoke(
        app,
        ["replay", str(claim), "--out", str(tmp_path / "out"), "--fixture"],
    )

    assert result.exit_code == 2, result.output
    assert "drifted apart" in result.output
    assert not (tmp_path / "out" / "manifest.json").exists()
