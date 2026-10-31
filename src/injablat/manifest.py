"""The manifest, and the gate that decides whether a replay is trustworthy.

A manifest is one HTTP exchange reduced to the facts that make it checkable:
which stack answered (revision, build string, template hash), what was asked
(request bytes), what the endpoint tokenized the prompt into, and what came back.
It deliberately carries no timestamps, durations, or hostnames the tool did not
pin, so the same exchange always hashes to the same manifest.

The determinism gate is not optional. Two dispatches of one configuration must
produce the same output hash; when they do not, the honest verdict is
NONDETERMINISTIC and no conclusion follows.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from enum import Enum
from typing import Any, Sequence

from pydantic import BaseModel, ConfigDict

from .backend import ChatResult
from .claim import Claim, Sampling
from .prompt import ChatTemplate


class Manifest(BaseModel):
    """The pinned identity of a single replay request/response pair."""

    model_config = ConfigDict(extra="forbid")

    claim_id: str
    config_id: str
    backend: str
    endpoint: str
    model: str
    model_revision: str
    endpoint_build: str
    chat_template_sha256: str
    chat_template_source: str
    # Who tokenized the prompt: the endpoint's reported tokenizer identity plus
    # the id list this tokenizer produces for a fixed probe string. Two stacks
    # that report the same name but tokenize the probe differently are not the
    # same computation.
    tokenizer_name: str
    tokenizer_probe_sha256: str
    prompt_tokens: int
    prompt_token_sha256: str
    request_sha256: str
    sampling: Sampling
    output_sha256: str


class Determinism(str, Enum):
    DETERMINISTIC = "DETERMINISTIC"
    NONDETERMINISTIC = "NONDETERMINISTIC"


@dataclass(frozen=True)
class DeterminismResult:
    verdict: Determinism
    mismatches: dict[str, tuple[str, ...]]

    @property
    def ok(self) -> bool:
        return self.verdict is Determinism.DETERMINISTIC

    def detail(self) -> str:
        if self.ok:
            return "repeat runs are byte-identical"
        parts = [
            f"{field}={', '.join(values)}" for field, values in sorted(self.mismatches.items())
        ]
        return "repeat runs disagree: " + "; ".join(parts)


_COMPARED_FIELDS = ("request_sha256", "prompt_token_sha256", "output_sha256")

# Tokenized once per replay to fingerprint the tokenizer itself. The text is
# fixed and carries both prose and digits so a substituted vocabulary usually
# changes at least one id.
TOKENIZER_PROBE = "injablat tokenizer probe: pack my box with five dozen liquor jugs. 0123456789"


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def token_ids_sha256(tokens: Sequence[int]) -> str:
    """Hash the id list, not its rendering: ids are what the endpoint reported."""
    encoded = json.dumps(list(tokens), separators=(",", ":")).encode("utf-8")
    return sha256_hex(encoded)


def build_manifest(
    *,
    claim: Claim,
    config_id: str,
    template: ChatTemplate,
    endpoint_build: str,
    tokenizer_name: str,
    tokenizer_probe_tokens: Sequence[int],
    prompt_tokens: Sequence[int],
    chat: ChatResult,
) -> Manifest:
    spec = claim.model
    return Manifest(
        claim_id=claim.id,
        config_id=config_id,
        backend=spec.backend,
        endpoint=spec.endpoint,
        model=spec.name,
        model_revision=spec.revision,
        endpoint_build=endpoint_build,
        chat_template_sha256=template.sha256,
        chat_template_source=template.source,
        tokenizer_name=tokenizer_name,
        tokenizer_probe_sha256=token_ids_sha256(tokenizer_probe_tokens),
        prompt_tokens=len(prompt_tokens),
        prompt_token_sha256=token_ids_sha256(prompt_tokens),
        request_sha256=sha256_hex(chat.request_bytes),
        sampling=claim.sampling,
        output_sha256=sha256_hex(chat.text.encode("utf-8")),
    )


def manifest_sha256(manifest: Manifest) -> str:
    """Stable digest of the manifest itself, used as the headline fingerprint."""
    payload: dict[str, Any] = manifest.model_dump(mode="json")
    canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return sha256_hex(canonical.encode("utf-8"))


def check_determinism(manifests: Sequence[Manifest]) -> DeterminismResult:
    """Compare every repeat of one configuration; any disagreement fails the gate."""
    if len(manifests) < 2:
        raise ValueError("determinism needs at least two manifests")
    mismatches: dict[str, tuple[str, ...]] = {}
    for field in _COMPARED_FIELDS:
        values = {str(getattr(manifest, field)) for manifest in manifests}
        if len(values) > 1:
            mismatches[field] = tuple(sorted(values))
    verdict = Determinism.NONDETERMINISTIC if mismatches else Determinism.DETERMINISTIC
    return DeterminismResult(verdict=verdict, mismatches=mismatches)
