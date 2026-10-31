"""The judgement: status, minimal sufficient set, and the re-run comparison.

This module turns a recorded sweep into a card, and decides the two exit codes
that matter: 0 when a verdict was produced, 1 when the repeats disagreed or the
search could not be completed. It also owns ``verify``, which re-dispatches the
exact request bytes recorded for every row and reports MATCH or DRIFT.

Two rules keep the judgement honest:

* a configuration that was not reproducible is UNDECIDED and contributes
  nothing to the minimal set — not even as a span that could have been dropped;
* the minimal set is described as one solution under the recorded deletion
  order. The rows that produced it travel with the card.

``DOES_NOT_REPRODUCE`` is a verdict, not an error: the behaviour was absent
under the full injection, so there is nothing to attribute.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict

from .ablate import Ablation, Behavior, ConfigKind, ModelInfo, Run, TemplateInfo
from .backend import Backend
from .claim import Detector, Sampling
from .manifest import Determinism, Manifest, sha256_hex


class Status(str, Enum):
    VERDICT = "VERDICT"
    DOES_NOT_REPRODUCE = "DOES_NOT_REPRODUCE"
    NO_VERDICT = "NO_VERDICT"


class Contribution(BaseModel):
    """One span's share of the minimal sufficient injection."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    span_id: str
    label: str
    tokens: int | None = None


class Report(BaseModel):
    """The rendered card, and everything verify needs to re-dispatch it."""

    model_config = ConfigDict(extra="forbid")

    claim_id: str
    status: Status
    reason: str
    model: ModelInfo
    sampling: Sampling
    detector: Detector
    chat_template: TemplateInfo
    prefill_span_id: str | None
    runs_per_config: int
    deletion_order: list[str]
    rows: list[Run]
    # None means "not established": no verdict, or nothing to attribute.
    minimal_sufficient: list[Contribution] | None
    survives_without: dict[str, bool | None]
    determinism: Determinism
    determinism_detail: str
    manifest: Manifest | None

    @property
    def exit_code(self) -> int:
        return 1 if self.status is Status.NO_VERDICT else 0

    @property
    def minimal_text(self) -> str | None:
        if self.minimal_sufficient is None:
            return None
        if not self.minimal_sufficient:
            return "none (the bare prompt already fires)"
        return " + ".join(item.label for item in self.minimal_sufficient)

    def survives_without_prefill(self) -> bool | None:
        if self.prefill_span_id is None:
            return None
        return self.survives_without.get(self.prefill_span_id)


def assemble(ablation: Ablation) -> Report:
    """Turn the sweep into a verdict. Any noise means no verdict at all."""
    full = next(row for row in ablation.rows if row.kind is ConfigKind.FULL)
    undecided = [row for row in ablation.rows if row.behavior is Behavior.UNDECIDED]

    if undecided:
        status = Status.NO_VERDICT
        reason = (
            f"no verdict: {len(undecided)} configuration(s) disagreed between "
            f"{ablation.runs_per_config} runs ({', '.join(row.label for row in undecided)}); "
            "a noisy configuration cannot support a conclusion"
        )
    elif full.behavior is Behavior.NOT_FIRED:
        status = Status.DOES_NOT_REPRODUCE
        reason = (
            "the behaviour does not appear under the full recorded injection, "
            "so there is nothing to attribute"
        )
    elif ablation.kept is None:
        status = Status.NO_VERDICT
        reason = "no verdict: the minimal sufficient search did not complete"
    else:
        status = Status.VERDICT
        reason = (
            "the behaviour appears under the full injection and under the minimal set; "
            "the spans left out are redundant under this deletion order"
        )

    minimal: list[Contribution] | None = None
    if status is Status.VERDICT and ablation.kept is not None:
        minimal = [
            _contribution(ablation, span_id) for span_id in ablation.kept
        ]

    survives: dict[str, bool | None] = {}
    for span_id in ablation.deletion_order:
        deletion = next(
            (
                row
                for row in ablation.rows
                if row.kind is ConfigKind.DELETION and row.tested == span_id
            ),
            None,
        )
        if deletion is None:
            survives[span_id] = None
        elif deletion.behavior is Behavior.FIRED:
            survives[span_id] = True
        elif deletion.behavior is Behavior.NOT_FIRED:
            survives[span_id] = False
        else:
            survives[span_id] = None

    return Report(
        claim_id=ablation.claim_id,
        status=status,
        reason=reason,
        model=ablation.model,
        sampling=ablation.sampling,
        detector=ablation.detector,
        chat_template=ablation.chat_template,
        prefill_span_id=ablation.prefill_span_id,
        runs_per_config=ablation.runs_per_config,
        deletion_order=ablation.deletion_order,
        rows=ablation.rows,
        minimal_sufficient=minimal,
        survives_without=survives,
        determinism=ablation.determinism,
        determinism_detail=ablation.determinism_detail,
        manifest=ablation.manifest,
    )


def _contribution(ablation: Ablation, span_id: str) -> Contribution:
    if span_id == ablation.prefill_span_id and ablation.prefill_tokens is not None:
        return Contribution(
            span_id=span_id,
            label=f"prefill[0:{ablation.prefill_tokens}]",
            tokens=ablation.prefill_tokens,
        )
    # Reuse the display label recorded on this span's deletion row, so a block
    # keeps one name throughout the card.
    deletion = next(
        (
            row
            for row in ablation.rows
            if row.kind is ConfigKind.DELETION and row.tested == span_id
        ),
        None,
    )
    return Contribution(
        span_id=span_id,
        label=deletion.label if deletion is not None else span_id,
        tokens=None,
    )


class VerifyStatus(str, Enum):
    MATCH = "MATCH"
    DRIFT = "DRIFT"
    UNDECIDED = "UNDECIDED"


class VerifyRow(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    config_id: str
    label: str
    status: VerifyStatus
    expected_sha256: str | None
    observed_sha256: str | None
    note: str | None = None


class VerifyResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    rows: list[VerifyRow]

    @property
    def drifted(self) -> list[VerifyRow]:
        return [row for row in self.rows if row.status is VerifyStatus.DRIFT]

    @property
    def matched(self) -> list[VerifyRow]:
        return [row for row in self.rows if row.status is VerifyStatus.MATCH]

    @property
    def undecided(self) -> list[VerifyRow]:
        return [row for row in self.rows if row.status is VerifyStatus.UNDECIDED]

    @property
    def exit_code(self) -> int:
        return 1 if self.drifted else 0


def verify_report(report: Report, backend: Backend, runs: int) -> VerifyResult:
    """Replay every recorded request body and compare output hashes."""
    if runs < 1:
        raise ValueError("verify needs at least one dispatch per row")

    rows: list[VerifyRow] = []
    for row in report.rows:
        if row.output_sha256 is None:
            # Recorded as UNDECIDED: there is no single hash to hold the row to.
            rows.append(
                VerifyRow(
                    config_id=row.config_id,
                    label=row.label,
                    status=VerifyStatus.UNDECIDED,
                    expected_sha256=None,
                    observed_sha256=None,
                    note="recorded as UNDECIDED; no output hash to compare",
                )
            )
            continue

        observed = sorted(
            {
                sha256_hex(backend.dispatch(row.request_body).text.encode("utf-8"))
                for _ in range(runs)
            }
        )
        if len(observed) > 1:
            rows.append(
                VerifyRow(
                    config_id=row.config_id,
                    label=row.label,
                    status=VerifyStatus.DRIFT,
                    expected_sha256=row.output_sha256,
                    observed_sha256=observed[0],
                    note=f"re-run is not reproducible ({len(observed)} distinct outputs)",
                )
            )
        elif observed[0] != row.output_sha256:
            rows.append(
                VerifyRow(
                    config_id=row.config_id,
                    label=row.label,
                    status=VerifyStatus.DRIFT,
                    expected_sha256=row.output_sha256,
                    observed_sha256=observed[0],
                    note="output hash changed",
                )
            )
        else:
            rows.append(
                VerifyRow(
                    config_id=row.config_id,
                    label=row.label,
                    status=VerifyStatus.MATCH,
                    expected_sha256=row.output_sha256,
                    observed_sha256=observed[0],
                )
            )
    return VerifyResult(rows=rows)
