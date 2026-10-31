"""The deletion sweep, the prefill ladder, and the greedy minimal-set search.

One unit of evidence is a *configuration*: the span ids it keeps plus, when the
prefill is present, the character prefix used in its place. Every configuration
is dispatched ``runs`` times. Two different outputs make the configuration
UNDECIDED and the search stops, because a conclusion drawn from a noisy
configuration is worse than no conclusion.

The search is greedy and therefore order-dependent. It is declared here, not
inferred: the prefill is tested first (it is the span the claim puts on the
table), then the other injected spans in claim order. A deletion that still
fires stays deleted (REDUNDANT); one that stops firing is reverted
(NECESSARY). When the prefill turns out to be necessary, its declared
truncation ladder is walked upward and the shortest declared prefix that still
fires becomes the prefill's contribution. The final set is dispatched once
more before anything is called minimal.

The result is one solution, not the only one. Every row is kept so a reader can
re-run the same claim under a different order and compare the two cards.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Sequence

from pydantic import BaseModel, ConfigDict, ValidationError

from .backend import Backend, canonical_json
from .claim import Claim, Detector, Sampling
from .detector import detect
from .manifest import (
    TOKENIZER_PROBE,
    Determinism,
    Manifest,
    build_manifest,
    check_determinism,
    sha256_hex,
)
from .prompt import (
    ChatTemplate,
    build_messages,
    longest_prefix,
    render_prompt,
    resolve_chat_template,
)


class AblationError(Exception):
    """The sweep cannot be produced, loaded, or does not belong to this claim."""


class Behavior(str, Enum):
    FIRED = "FIRED"
    NOT_FIRED = "NOT_FIRED"
    UNDECIDED = "UNDECIDED"


class Role(str, Enum):
    """What a row contributes. Only the first three judge a span."""

    BASELINE = "baseline"
    REPRODUCES = "reproduces"
    NECESSARY = "necessary"
    REDUNDANT = "redundant"
    UNDECIDED = "undecided"
    INSUFFICIENT = "insufficient"
    SUFFICIENT = "sufficient"
    VERIFIED = "verified"


class ConfigKind(str, Enum):
    BASELINE = "baseline"
    FULL = "full"
    DELETION = "deletion"
    PREFIX = "prefix"
    MINIMAL = "minimal"


class ModelInfo(BaseModel):
    """The model and endpoint identity a card carries. No local path."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    endpoint: str
    backend: str
    revision: str
    chat_template_sha: str
    endpoint_build: str
    tokenizer_name: str


class TemplateInfo(BaseModel):
    """The exact template body, so the card describes the prompt it measured."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    sha256: str
    source: str
    text: str


class Run(BaseModel):
    """One configuration's measured outcome, and the bytes that produced it."""

    model_config = ConfigDict(extra="forbid")

    config_id: str
    label: str
    kind: ConfigKind
    tested: str | None
    behavior: Behavior
    role: Role
    # None when the repeat runs disagreed: no single output hash exists to pin.
    output_sha256: str | None
    observed_output_shas: list[str]
    prompt_tokens: int
    # Tokens of the prefill actually present in this configuration; None when
    # the prefill was removed or the claim has none.
    prefill_tokens: int | None
    keep: list[str]
    remove: list[str]
    # The exact body sent, so verify can replay the exchange without rebuilding
    # it from a claim that may have moved on.
    request_body: dict[str, Any]


class Ablation(BaseModel):
    """Everything the sweep observed, before any judgement is applied."""

    model_config = ConfigDict(extra="forbid")

    claim_id: str
    claim_sha256: str
    model: ModelInfo
    sampling: Sampling
    detector: Detector
    chat_template: TemplateInfo
    prefill_span_id: str | None
    runs_per_config: int
    deletion_order: list[str]
    rows: list[Run]
    # Set only when the search completed and its final set fired. None means no
    # minimal set was established.
    kept: list[str] | None
    prefill_tokens: int | None
    manifest: Manifest | None
    determinism: Determinism
    determinism_detail: str


@dataclass(frozen=True)
class AblationResult:
    ablation: Ablation
    # config_id -> the per-dispatch records written under runs/. Held apart from
    # the ablation so the model stays free of transport chatter.
    records: dict[str, list[dict[str, Any]]]


@dataclass(frozen=True)
class Measurement:
    behavior: Behavior
    output_sha256: str | None
    observed_output_shas: tuple[str, ...]
    prompt_tokens: int
    request_body: dict[str, Any]


class Sweeper:
    """Dispatches configurations under one pinned stack, and caches the facts
    that are the same for all of them (build, tokenizer, probe)."""

    def __init__(
        self,
        claim: Claim,
        backend: Backend,
        template: ChatTemplate,
        runs: int,
    ) -> None:
        self.claim = claim
        self.backend = backend
        self.template = template
        self.runs = runs
        self.endpoint_build = backend.identity()
        self.tokenizer_name = backend.tokenizer_name()
        self.probe_tokens = backend.tokenize(TOKENIZER_PROBE).tokens
        self.records: dict[str, list[dict[str, Any]]] = {}

    def tokenize(self, text: str) -> tuple[int, ...]:
        return self.backend.tokenize(text).tokens

    def measure(
        self, config_id: str, *, keep_ids: Sequence[str], prefill_prefix: str | None
    ) -> Measurement:
        dropped = {span.id for span in self.claim.spans if span.id not in set(keep_ids)}
        messages = build_messages(
            self.claim, without=dropped, prefill_prefix=prefill_prefix
        )
        continues = bool(messages) and messages[-1]["role"] == "assistant"
        prompt = render_prompt(self.template, messages, continue_final_message=continues)
        prompt_ids = self.backend.tokenize(prompt).tokens

        chats = [
            self.backend.chat(
                messages, self.claim.sampling, continue_final_message=continues
            )
            for _ in range(self.runs)
        ]
        manifests = [
            build_manifest(
                claim=self.claim,
                config_id=config_id,
                template=self.template,
                endpoint_build=self.endpoint_build,
                tokenizer_name=self.tokenizer_name,
                tokenizer_probe_tokens=self.probe_tokens,
                prompt_tokens=prompt_ids,
                chat=chat,
            )
            for chat in chats
        ]

        gate = check_determinism(manifests)
        observed = tuple(sorted({manifest.output_sha256 for manifest in manifests}))
        if not gate.ok:
            behavior = Behavior.UNDECIDED
            output_sha = None
        else:
            prefix = messages[-1]["content"] if continues else ""
            fired = detect(self.claim.detector, prefix + chats[0].text)
            behavior = Behavior.FIRED if fired else Behavior.NOT_FIRED
            output_sha = observed[0]

        self.records[config_id] = [
            {
                "config_id": config_id,
                "run": index,
                "manifest": manifest.model_dump(mode="json"),
                "request_body": chat.request_body,
                "response_body": chat.response_body,
                "elapsed_ms": chat.elapsed_ms,
            }
            for index, (manifest, chat) in enumerate(zip(manifests, chats), start=1)
        ]
        return Measurement(
            behavior=behavior,
            output_sha256=output_sha,
            observed_output_shas=observed,
            prompt_tokens=len(prompt_ids),
            request_body=chats[0].request_body,
        )

    def manifest_for(self, config_id: str) -> Manifest | None:
        """The first manifest of a configuration, for use as a pin."""
        records = self.records.get(config_id)
        if not records:
            return None
        return Manifest.model_validate(records[0]["manifest"])


def claim_fingerprint(claim: Claim) -> str:
    """A stable digest of the claim, excluding its local template path.

    The path is where the bytes happen to live on this machine; the bytes
    themselves are pinned by ``chat_template_sha``.
    """
    payload = {
        "id": claim.id,
        "model": {
            "name": claim.model.name,
            "endpoint": claim.model.endpoint,
            "backend": claim.model.backend,
            "revision": claim.model.revision,
            "chat_template_sha": claim.model.chat_template_sha,
        },
        "spans": [span.model_dump(mode="json") for span in claim.spans],
        "detector": claim.detector.model_dump(mode="json"),
        "sampling": claim.sampling.model_dump(mode="json"),
    }
    return sha256_hex(canonical_json(payload))


def ablate_claim(claim: Claim, backend: Backend, runs: int) -> AblationResult:
    """Run the sweep and the greedy search. No verdict is decided here."""
    if runs < 2:
        raise ValueError("an ablation needs at least two runs per configuration")

    template = resolve_chat_template(claim, backend)
    sweeper = Sweeper(claim, backend, template, runs)

    spans = list(claim.spans)
    by_id = {span.id: span for span in spans}
    all_ids = [span.id for span in spans]
    user_ids = [span.id for span in spans if not span.injected]
    injected = [span for span in spans if span.injected]
    prefill = next((span for span in spans if span.role == "prefill"), None)
    prefill_full_tokens = len(sweeper.tokenize(prefill.text)) if prefill is not None else None

    # The prefill is the span the claim is about, so it is tested against the
    # full injection first. The rest follow in claim order.
    order = ([prefill.id] if prefill is not None else []) + [
        span.id for span in injected if span is not prefill
    ]

    def row(
        measurement: Measurement,
        *,
        config_id: str,
        label: str,
        kind: ConfigKind,
        tested: str | None,
        role: Role,
        keep_ids: Sequence[str],
        prefill_tokens: int | None,
    ) -> Run:
        kept_set = set(keep_ids)
        return Run(
            config_id=config_id,
            label=label,
            kind=kind,
            tested=tested,
            behavior=measurement.behavior,
            role=role,
            output_sha256=measurement.output_sha256,
            observed_output_shas=list(measurement.observed_output_shas),
            prompt_tokens=measurement.prompt_tokens,
            prefill_tokens=prefill_tokens,
            keep=[span_id for span_id in all_ids if span_id in kept_set],
            remove=[span_id for span_id in all_ids if span_id not in kept_set],
            request_body=measurement.request_body,
        )

    def kept_prefill_tokens(keep_ids: Sequence[str]) -> int | None:
        if prefill is None or prefill.id not in keep_ids:
            return None
        return prefill_full_tokens

    rows: list[Run] = []
    bare = sweeper.measure("bare", keep_ids=user_ids, prefill_prefix=None)
    rows.append(
        row(
            bare,
            config_id="bare",
            label="(bare prompt)",
            kind=ConfigKind.BASELINE,
            tested=None,
            role=Role.BASELINE,
            keep_ids=user_ids,
            prefill_tokens=None,
        )
    )

    full = sweeper.measure("full", keep_ids=all_ids, prefill_prefix=None)
    rows.append(
        row(
            full,
            config_id="full",
            label="(nothing removed)",
            kind=ConfigKind.FULL,
            tested=None,
            role=Role.REPRODUCES,
            keep_ids=all_ids,
            prefill_tokens=kept_prefill_tokens(all_ids),
        )
    )

    kept: list[str] | None = None
    minimal_prefill: int | None = None
    blocked = bare.behavior is Behavior.UNDECIDED or full.behavior is Behavior.UNDECIDED

    if not blocked and full.behavior is Behavior.FIRED:
        dropped: set[str] = set()

        # -- deletion pass -------------------------------------------------
        for span_id in order:
            span = by_id[span_id]
            keep_ids = [item for item in all_ids if item not in dropped and item != span_id]
            # Name the block being removed the way the claim names it. The
            # fallback keeps older claims readable: a prefill without a label is
            # shown as truncated to zero.
            label = span.label or ("prefill[0:0]" if span.role == "prefill" else span_id)
            measurement = sweeper.measure(f"drop-{span_id}", keep_ids=keep_ids, prefill_prefix=None)
            if measurement.behavior is Behavior.FIRED:
                role = Role.REDUNDANT
            elif measurement.behavior is Behavior.NOT_FIRED:
                role = Role.NECESSARY
            else:
                role = Role.UNDECIDED
            rows.append(
                row(
                    measurement,
                    config_id=f"drop-{span_id}",
                    label=label,
                    kind=ConfigKind.DELETION,
                    tested=span_id,
                    role=role,
                    keep_ids=keep_ids,
                    prefill_tokens=kept_prefill_tokens(keep_ids),
                )
            )
            if measurement.behavior is Behavior.UNDECIDED:
                blocked = True
                break
            if measurement.behavior is Behavior.FIRED:
                dropped.add(span_id)

        # -- prefill ladder ------------------------------------------------
        prefill_prefix: str | None = None
        if not blocked and prefill is not None and prefill.id not in dropped:
            keep_ids = [item for item in all_ids if item not in dropped]
            seen = {prefill.text}
            for budget in prefill.truncations:
                step = longest_prefix(prefill.text, budget, sweeper.tokenize)
                if step.text in seen:
                    # The full body is already the "(nothing removed)" row.
                    continue
                seen.add(step.text)
                measurement = sweeper.measure(
                    f"prefill-{step.tokens}", keep_ids=keep_ids, prefill_prefix=step.text
                )
                if measurement.behavior is Behavior.FIRED:
                    role = Role.SUFFICIENT
                elif measurement.behavior is Behavior.NOT_FIRED:
                    role = Role.INSUFFICIENT
                else:
                    role = Role.UNDECIDED
                rows.append(
                    row(
                        measurement,
                        config_id=f"prefill-{step.tokens}",
                        label=f"prefill[0:{step.tokens}]",
                        kind=ConfigKind.PREFIX,
                        tested=prefill.id,
                        role=role,
                        keep_ids=keep_ids,
                        prefill_tokens=step.tokens,
                    )
                )
                if measurement.behavior is Behavior.UNDECIDED:
                    blocked = True
                    break
                if measurement.behavior is Behavior.FIRED:
                    prefill_prefix = step.text
                    minimal_prefill = step.tokens
                    break
            if not blocked and prefill_prefix is None:
                minimal_prefill = prefill_full_tokens

        # -- verify the final set -----------------------------------------
        if not blocked:
            keep_ids = [item for item in all_ids if item not in dropped]
            verify_messages = build_messages(
                claim,
                without=[item for item in all_ids if item not in set(keep_ids)],
                prefill_prefix=prefill_prefix,
            )
            full_messages = build_messages(claim)
            if verify_messages == full_messages:
                # Nothing was removed: the full row already verifies the set.
                kept = [item for item in order if item not in dropped]
            else:
                measurement = sweeper.measure(
                    "minimal", keep_ids=keep_ids, prefill_prefix=prefill_prefix
                )
                rows.append(
                    row(
                        measurement,
                        config_id="minimal",
                        label="(minimal set)",
                        kind=ConfigKind.MINIMAL,
                        tested=None,
                        role=Role.VERIFIED,
                        keep_ids=keep_ids,
                        prefill_tokens=kept_prefill_tokens(keep_ids)
                        if prefill_prefix is None
                        else minimal_prefill,
                    )
                )
                if measurement.behavior is Behavior.FIRED:
                    kept = [item for item in order if item not in dropped]
                else:
                    blocked = True

    undecided = [item for item in rows if item.behavior is Behavior.UNDECIDED]
    if undecided:
        determinism = Determinism.NONDETERMINISTIC
        detail = "not reproducible across {} runs: {}".format(
            runs, ", ".join(item.label for item in undecided)
        )
    else:
        determinism = Determinism.DETERMINISTIC
        detail = "repeat runs are byte-identical for every configuration"

    manifest = None
    if full.behavior is not Behavior.UNDECIDED:
        manifest = sweeper.manifest_for("full")

    ablation = Ablation(
        claim_id=claim.id,
        claim_sha256=claim_fingerprint(claim),
        model=ModelInfo(
            name=claim.model.name,
            endpoint=claim.model.endpoint,
            backend=claim.model.backend,
            revision=claim.model.revision,
            chat_template_sha=claim.model.chat_template_sha,
            endpoint_build=sweeper.endpoint_build,
            tokenizer_name=sweeper.tokenizer_name,
        ),
        sampling=claim.sampling,
        detector=claim.detector,
        chat_template=TemplateInfo(
            sha256=template.sha256, source=template.source, text=template.text
        ),
        prefill_span_id=prefill.id if prefill is not None else None,
        runs_per_config=runs,
        deletion_order=order,
        rows=rows,
        kept=kept,
        prefill_tokens=(
            minimal_prefill
            if kept is not None and prefill is not None and prefill.id in kept
            else None
        ),
        manifest=manifest,
        determinism=determinism,
        determinism_detail=detail,
    )
    return AblationResult(ablation=ablation, records=sweeper.records)


def load_ablation(directory: Path, claim: Claim) -> Ablation:
    """Read a recorded sweep and check it still belongs to this claim."""
    path = directory / "ablation.json"
    if not path.is_file():
        raise AblationError(
            f"no ablation sweep in {directory}: run 'injablat ablate <claim>' first"
        )
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise AblationError(f"cannot read {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise AblationError(f"{path}: not valid JSON: {exc}") from exc
    try:
        ablation = Ablation.model_validate(raw)
    except ValidationError as exc:
        raise AblationError(f"{path}: not a usable sweep: {exc}") from exc

    if ablation.claim_id != claim.id:
        raise AblationError(
            f"{path} records claim '{ablation.claim_id}', not '{claim.id}'"
        )
    if ablation.claim_sha256 != claim_fingerprint(claim):
        raise AblationError(
            f"{path} was recorded from a different version of claim '{claim.id}'; "
            "the claim changed after the sweep. Re-run 'injablat ablate'."
        )
    return ablation
