"""The claim file: everything a disputed run was fed, pinned.

A claim is the input contract of the tool. It records the model and its
revision, the backend dialect it ran on, the hash of the chat template that
produced the prompt, the injected spans (system prompt, prefill, tool schema,
few-shot blocks), the declarative behaviour detector, and the sampling
parameters. Everything a replay needs, and nothing a replay could infer.

Spans live either inline under ``spans:`` or in a single JSONL file named by
``spans_file:`` — never both. The JSONL form is what a published trace converts
to: one span per line, so the text stays diffable and greppable.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

SpanRole = Literal["system", "prefill", "tool_schema", "few_shot", "user"]
BackendKind = Literal["vllm", "llamacpp"]
DetectorKind = Literal["contains", "regex", "json_field"]

SPAN_ROLES: tuple[SpanRole, ...] = ("system", "prefill", "tool_schema", "few_shot", "user")

# v0.1 prefill ablation relies on vLLM's continue_final_message. llama.cpp's
# OpenAI-compatible endpoint does not document an equivalent, and the local
# render + /v1/completions workaround is explicitly out of scope.
PREFILL_REQUIRES_VLLM = (
    "prefill ablation is only supported on a vLLM endpoint: it depends on "
    "continue_final_message, which the llama.cpp chat endpoint does not document. "
    "Run this claim against vLLM, or remove the prefill span."
)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


class ClaimError(Exception):
    """The claim file cannot be read, parsed, or is internally inconsistent."""


class Span(BaseModel):
    """One injected block of text, with the token count recorded from the trace."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    role: SpanRole
    text: str = Field(min_length=1)
    # Optional display name for the verdict card. It defaults to the span id and
    # exists because a source may name a block differently from its machine id
    # (for example "few_shot[0:2]" for a few-shot block of two examples).
    label: str | None = Field(default=None, min_length=1)
    # Measured by the tokenizer of the run the claim was recorded from, not by
    # this tool. validate re-measures and reports a difference instead of
    # silently trusting either number.
    tokens: int = Field(ge=0)
    # Token counts the prefix ladder steps through when this span is a prefill.
    truncations: list[int] = Field(default_factory=list)

    @field_validator("id")
    @classmethod
    def _check_id(cls, value: str) -> str:
        if not _ID_RE.match(value):
            raise ValueError("span id must match [A-Za-z0-9][A-Za-z0-9._-]*")
        return value

    @field_validator("truncations")
    @classmethod
    def _check_truncations(cls, values: list[int], info) -> list[int]:
        previous = -1
        for value in values:
            if value < 0:
                raise ValueError("truncation lengths must be >= 0")
            if value <= previous:
                raise ValueError("truncations must be strictly increasing")
            previous = value
        tokens = info.data.get("tokens")
        if tokens is not None and values and values[-1] > tokens:
            raise ValueError("truncations cannot exceed the span's recorded token count")
        return values

    @property
    def injected(self) -> bool:
        """False for the user prompt: it is the task, not something injected."""
        return self.role != "user"


class ModelSpec(BaseModel):
    """Where the disputed run pointed, and the identity it must still have."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(min_length=1)
    endpoint: str
    backend: BackendKind
    revision: str = Field(min_length=1)
    chat_template_sha: str
    # Optional: the template body as shipped with the claim. Preferred over the
    # endpoint's copy because a replayed prompt must be identical, not similar.
    chat_template_path: Path | None = None

    @field_validator("endpoint")
    @classmethod
    def _check_endpoint(cls, value: str) -> str:
        if not value.startswith(("http://", "https://")):
            raise ValueError("endpoint must be an http:// or https:// URL")
        return value.rstrip("/")

    @field_validator("chat_template_sha")
    @classmethod
    def _check_sha(cls, value: str) -> str:
        if not _SHA256_RE.match(value):
            raise ValueError("chat_template_sha must be 64 lowercase hex characters")
        return value


class Detector(BaseModel):
    """A declarative behaviour predicate. No model judges another model here."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: DetectorKind
    expect: str = Field(min_length=1)
    # json_field only: dotted path into the parsed response body that must equal
    # ``expect``. A field name is required to make that kind meaningful.
    path: str | None = None

    @model_validator(mode="after")
    def _check(self) -> "Detector":
        if self.kind == "json_field" and not self.path:
            raise ValueError("detector.path is required when kind is json_field")
        if self.kind != "json_field" and self.path is not None:
            raise ValueError("detector.path is only meaningful when kind is json_field")
        if self.kind == "regex":
            try:
                re.compile(self.expect)
            except re.error as exc:
                raise ValueError(f"detector.expect is not a valid regular expression: {exc}") from exc
        return self


class Sampling(BaseModel):
    """Pinned decoding parameters. A claim with a moving temperature is not replayable."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    temperature: float = Field(default=0.0, ge=0.0)
    top_p: float = Field(default=1.0, gt=0.0, le=1.0)
    seed: int = 0
    max_tokens: int = Field(gt=0)


class Claim(BaseModel):
    """A validated claim, with its spans already resolved."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    description: str | None = None
    model: ModelSpec
    spans: list[Span] = Field(min_length=1)
    detector: Detector
    sampling: Sampling

    @field_validator("id")
    @classmethod
    def _check_id(cls, value: str) -> str:
        if not _ID_RE.match(value):
            raise ValueError("claim id must match [A-Za-z0-9][A-Za-z0-9._-]*")
        return value

    @model_validator(mode="after")
    def _check_spans(self) -> "Claim":
        ids = [span.id for span in self.spans]
        duplicates = {span_id for span_id in ids if ids.count(span_id) > 1}
        if duplicates:
            raise ValueError(f"duplicate span ids: {', '.join(sorted(duplicates))}")

        prefills = [span for span in self.spans if span.role == "prefill"]
        if len(prefills) > 1:
            raise ValueError("a claim has at most one prefill span")
        if prefills:
            if self.spans[-1].role != "prefill":
                raise ValueError("the prefill span must be the last span: it is the assistant prefix")
            if self.model.backend != "vllm":
                raise ValueError(PREFILL_REQUIRES_VLLM)
        return self

    @property
    def has_prefill(self) -> bool:
        return any(span.role == "prefill" for span in self.spans)


def _read_spans(path: Path) -> list[Span]:
    spans: list[Span] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ClaimError(f"cannot read spans file {path}: {exc}") from exc

    for number, line in enumerate(lines, start=1):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            raw = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise ClaimError(f"{path}:{number}: not valid JSON: {exc.msg}") from exc
        if not isinstance(raw, dict):
            raise ClaimError(f"{path}:{number}: each line must be a JSON object")
        try:
            spans.append(Span.model_validate(raw))
        except ValidationError as exc:
            raise ClaimError(f"{path}:{number}: {_format_validation_error(exc)}") from exc

    if not spans:
        raise ClaimError(f"{path}: no spans found")
    return spans


def _format_validation_error(exc: ValidationError) -> str:
    parts = []
    for error in exc.errors():
        location = ".".join(str(item) for item in error["loc"]) or "<root>"
        parts.append(f"{location}: {error['msg']}")
    return "; ".join(parts)


def load_claim(path: Path) -> Claim:
    """Parse and validate a claim file, resolving ``spans_file`` and template paths."""
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ClaimError(f"cannot read claim file {path}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise ClaimError(f"{path}: not valid YAML: {exc}") from exc

    if not isinstance(raw, dict):
        raise ClaimError(f"{path}: the claim file must contain a YAML mapping")

    raw = dict(raw)
    spans_file = raw.pop("spans_file", None)
    inline_spans = raw.get("spans")

    if spans_file is not None and inline_spans is not None:
        raise ClaimError(f"{path}: set either spans or spans_file, not both")
    if spans_file is not None:
        if not isinstance(spans_file, str):
            raise ClaimError(f"{path}: spans_file must be a path string")
        raw["spans"] = [
            span.model_dump(mode="json") for span in _read_spans(path.parent / spans_file)
        ]
    if "spans" not in raw:
        raise ClaimError(f"{path}: a claim needs spans or spans_file")

    model = raw.get("model")
    if isinstance(model, dict):
        model = dict(model)
        template_path = model.get("chat_template_path")
        if template_path is not None:
            if not isinstance(template_path, str):
                raise ClaimError(f"{path}: model.chat_template_path must be a path string")
            model["chat_template_path"] = (path.parent / template_path).resolve()
        raw["model"] = model

    try:
        return Claim.model_validate(raw)
    except ValidationError as exc:
        raise ClaimError(f"{path}: {_format_validation_error(exc)}") from exc
