"""Runtime defaults read from a config file.

Only settings with a reader live here. A claim never carries a timeout or a run
count, so nothing in the config is overridden by a claim; command-line options
are the override.

The file is optional. ``configs/injablat.yaml`` in the working directory is
used when present, and the built-in defaults apply otherwise. Unknown keys are
rejected rather than ignored, so a setting cannot be documented here while
nothing reads it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


class ConfigError(Exception):
    """The config file cannot be read, or names a setting the tool does not use."""


@dataclass(frozen=True)
class RunSettings:
    request_timeout_s: float = 300.0
    tokenize_timeout_s: float = 30.0
    runs: int = 2


_ALLOWED_KEYS = {
    "backend": {"request_timeout_s", "tokenize_timeout_s"},
    "determinism": {"runs"},
}


def _section(raw: dict, name: str, path: Path) -> dict:
    value = raw.get(name)
    if value is None:
        value = {}
    if not isinstance(value, dict):
        raise ConfigError(f"{path}: {name} must be a mapping")
    unknown = sorted(set(value) - _ALLOWED_KEYS[name])
    if unknown:
        raise ConfigError(f"{path}: {name} has no reader for: {', '.join(unknown)}")
    return value


def _positive(value: object, default: float, path: Path, key: str) -> float:
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        raise ConfigError(f"{path}: {key} must be a positive number")
    return float(value)


def load_run_settings(path: Path | None) -> RunSettings:
    """Read ``path`` when given; return the built-in defaults when it is None."""
    if path is None:
        return RunSettings()

    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ConfigError(f"cannot read config {path}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise ConfigError(f"{path}: not valid YAML: {exc}") from exc

    if raw is None:
        return RunSettings()
    if not isinstance(raw, dict):
        raise ConfigError(f"{path}: the config must be a YAML mapping")

    unknown = sorted(set(raw) - set(_ALLOWED_KEYS))
    if unknown:
        raise ConfigError(f"{path}: unsupported top-level settings: {', '.join(unknown)}")

    backend = _section(raw, "backend", path)
    determinism = _section(raw, "determinism", path)

    runs = determinism.get("runs")
    if runs is not None:
        if isinstance(runs, bool) or not isinstance(runs, int) or runs < 2:
            raise ConfigError(f"{path}: determinism.runs must be an integer >= 2")

    defaults = RunSettings()
    return RunSettings(
        request_timeout_s=_positive(
            backend.get("request_timeout_s"),
            defaults.request_timeout_s,
            path,
            "backend.request_timeout_s",
        ),
        tokenize_timeout_s=_positive(
            backend.get("tokenize_timeout_s"),
            defaults.tokenize_timeout_s,
            path,
            "backend.tokenize_timeout_s",
        ),
        runs=runs if runs is not None else defaults.runs,
    )
