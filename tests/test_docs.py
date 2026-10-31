"""Polish gates: the shipped docs, demo record, and release pipeline stay in sync.

These are contract tests for the public surface. They assert that the bilingual
README pair exists and is cross-linked, that the recorded demo is a real
asciicast, that its results record matches the card, and that a tag push still
builds and publishes release assets.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
FOOTER = '<p align="center"><sub><a href="./LICENSE">MIT</a> © 2026 SuperMarioYL</sub></p>'
EMOJI = re.compile("[\U0001f300-\U0001faff\u2600-\u27bf]")


def _read(name: str) -> str:
    return (ROOT / name).read_text(encoding="utf-8")


def test_readmes_are_cross_linked_and_complete() -> None:
    zh = _read("README.md")
    en = _read("README.en.md")

    assert zh.splitlines()[0].strip() == "[English](./README.en.md) | **简体中文**"
    assert en.splitlines()[0].strip() == "**English** | [简体中文](./README.md)"

    for heading in (
        "它解决什么问题",
        "安装",
        "快速开始",
        "演示",
        "工作原理",
        "命令参考",
        "配置",
        "与评测/trace 工具的区别",
        "局限与已知边界",
        "路线图",
        "团队版与授权",
        "LICENSE",
    ):
        assert f"## {heading}" in zh, heading

    for heading in (
        "What it solves",
        "Install",
        "Quick start",
        "Demo",
        "How it works",
        "Command reference",
        "Configuration",
        "How it differs from eval and trace tools",
        "Limitations and known boundaries",
        "Roadmap",
        "Team edition and licensing",
        "License",
    ):
        assert f"## {heading}" in en, heading

    assert "./docs/demo.cast" in zh and "./docs/demo.cast" in en
    assert "./docs/demo-results.json" in zh and "./docs/demo-results.json" in en


def test_readme_footer_is_clean_and_sections_have_no_emoji_icons() -> None:
    for name in ("README.md", "README.en.md"):
        text = _read(name)
        assert text.rstrip().endswith(FOOTER), name
        for line in text.splitlines():
            if line.startswith("#"):
                assert not EMOJI.search(line), (name, line)


def test_demo_cast_is_a_real_asciicast_v2() -> None:
    lines = _read("docs/demo.cast").splitlines()
    header = json.loads(lines[0])
    assert header["version"] == 2
    assert header["width"] >= 100
    assert "command" not in header

    events = [json.loads(line) for line in lines[1:]]
    assert events
    output = "".join(event[2] for event in events if event[1] == "o")

    for command in (
        "injablat validate examples/qwen-prefill/claim.yaml --fixture",
        "injablat replay examples/qwen-prefill/claim.yaml --runs 2 --fixture",
        "injablat ablate examples/qwen-prefill/claim.yaml --runs 2 --fixture",
        "injablat report examples/qwen-prefill/claim.yaml",
        "injablat verify reports/qwen-prefill-curl --fixture",
    ):
        assert command in output, command

    assert "minimal sufficient injection  prefill[0:2232] + tool_schema + few_shot[0:2]" in output
    assert "survives without prefill      NO" in output
    assert "MATCH 10 of 10 rows" in output
    assert "/Users/" not in output and "/tmp/" not in output


def test_demo_results_match_the_recorded_card() -> None:
    result = json.loads(_read("docs/demo-results.json"))
    assert result["product"] == "injablat"
    assert result["version"] == "0.1.0"
    assert result["mode"] == "offline-fixture"

    commands = {entry["id"]: entry for entry in result["commands"]}
    assert set(commands) == {"validate", "replay", "ablate", "report", "verify"}
    assert all(entry["exit_code"] == 0 for entry in commands.values())
    assert commands["validate"]["key_output"]["spans_total"] == 5
    assert commands["replay"]["key_output"]["determinism"] == "DETERMINISTIC"

    ablate = commands["ablate"]["key_output"]
    assert ablate["survives_without_prefill"] is False
    assert ablate["minimal_sufficient"] == ["prefill[0:2232]", "tool_schema", "few_shot[0:2]"]
    assert len(ablate["rows"]) == 10

    verify = commands["verify"]["key_output"]
    assert verify["matched"] == verify["rows"] == 10
    assert result["verdict"]["replay_match"] == {"matched": 10, "rows": 10}


def test_changelog_covers_every_milestone() -> None:
    text = _read("CHANGELOG.md")
    assert "## [0.1.0] - 2026-10-31" in text
    assert "### m1 — replay and manifest" in text
    assert "### m2 — ablation and minimal set" in text
    assert "### m3 — verify and demo" in text


def test_release_workflow_attaches_release_assets() -> None:
    text = _read(".github/workflows/release.yml")
    workflow = yaml.safe_load(text)
    assert isinstance(workflow, dict)
    assert 'tags: ["v*"]' in text
    assert "python -m build" in text
    assert "sha256sum" in text
    assert "softprops/action-gh-release@v2" in text
    assert "files: dist/*" in text
    assert "rm -f dist/checksums.txt" in text
    assert "vars.PYPI_PUBLISH == 'true'" in text
