"""Declarative behaviour detectors.

The oracle is a predicate over text, never a model: ``contains``, ``regex``, or
one field of a JSON answer. A detector that asked another model whether the
behaviour happened would move the disputed judgement from the claim's author to
a third party, which is exactly what this tool exists to avoid.

``json_field`` compares ``expect`` against the field's value rendered
canonically: strings as themselves, and everything else as compact JSON
(``true``, ``null``, ``20``, ``{"a":1}``). A field path is dotted and may index
lists with integers. An answer that is not JSON, or has no such field, is
NOT_FIRED rather than an error: models answer in prose when they decline, and an
ablation must not abort on that.
"""

from __future__ import annotations

import json
import re
from typing import Any

from .claim import Detector


def detect(detector: Detector, text: str) -> bool:
    """Whether ``text`` contains the behaviour the claim declares."""
    if detector.kind == "contains":
        return detector.expect in text
    if detector.kind == "regex":
        return re.search(detector.expect, text) is not None
    return _json_field(detector, text)


def _json_field(detector: Detector, text: str) -> bool:
    assert detector.path is not None  # guaranteed by Detector validation
    try:
        payload = json.loads(text)
    except ValueError:
        return False
    value = _resolve(payload, detector.path)
    if value is _MISSING:
        return False
    return _render(value) == detector.expect


_MISSING = object()


def _resolve(payload: Any, path: str) -> Any:
    current = payload
    for part in path.split("."):
        if isinstance(current, dict):
            if part not in current:
                return _MISSING
            current = current[part]
        elif isinstance(current, list):
            if not part.isdigit():
                return _MISSING
            index = int(part)
            if index >= len(current):
                return _MISSING
            current = current[index]
        else:
            return _MISSING
    return current


def _render(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
