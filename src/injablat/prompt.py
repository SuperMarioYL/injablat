"""Chat template resolution, prompt rendering, and the prefill prefix ladder.

The template body is never guessed. Resolution order is fixed:

1. the claim's own ``chat_template_path``;
2. the endpoint (llama.cpp ``GET /props``, vLLM ``GET /tokenizer_info``);
3. refuse.

A pure HTTP client that fell back to a local model cache or downloaded a
template would make the manifest claim a provenance it does not have, so there
is no step 4.

Span roles are mapped onto chat roles deterministically: the injection is text
at a known position, and ablation deletes text. ``system`` and ``tool_schema``
become system messages, ``few_shot`` and ``user`` become user messages, and
``prefill`` becomes the final assistant message the endpoint continues.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Collection, Literal, Sequence

import jinja2
from jinja2 import Environment, Undefined

from .backend import Backend
from .claim import Claim, Span

_MESSAGE_ROLE: dict[str, str] = {
    "system": "system",
    "tool_schema": "system",
    "few_shot": "user",
    "user": "user",
    "prefill": "assistant",
}


class TemplateError(Exception):
    """No usable template was found, or it does not match the pinned hash."""


@dataclass(frozen=True)
class ChatTemplate:
    text: str
    sha256: str
    source: Literal["claim", "endpoint"]


@dataclass(frozen=True)
class PrefixSlice:
    """A character prefix of a span and the token count it actually measured."""

    text: str
    tokens: int


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def resolve_chat_template(claim: Claim, backend: Backend) -> ChatTemplate:
    """Return the template the claim pins, or fail with the reason it could not."""
    spec = claim.model

    if spec.chat_template_path is not None:
        path: Path = spec.chat_template_path
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as exc:
            raise TemplateError(f"cannot read chat template {path}: {exc}") from exc
        digest = sha256_text(text)
        if digest != spec.chat_template_sha:
            raise TemplateError(
                f"{path} hashes to {digest}, but the claim pins chat_template_sha "
                f"{spec.chat_template_sha}. The claim and its template have drifted apart."
            )
        return ChatTemplate(text=text, sha256=digest, source="claim")

    text = backend.chat_template()
    if text is None:
        raise TemplateError(
            f"no chat template available for backend '{backend.kind}' at {spec.endpoint}: "
            f"{backend.TEMPLATE_HINT}, and the claim does not carry a chat_template_path. "
            "Add model.chat_template_path to the claim; this tool is a pure HTTP client and "
            "will not read a local model cache or download one."
        )
    digest = sha256_text(text)
    if digest != spec.chat_template_sha:
        raise TemplateError(
            f"the template served by {spec.endpoint} hashes to {digest}, but the claim pins "
            f"chat_template_sha {spec.chat_template_sha}. The endpoint is not the stack the "
            "claim was recorded on; point the claim at a matching endpoint or ship the exact "
            "template as model.chat_template_path."
        )
    return ChatTemplate(text=text, sha256=digest, source="endpoint")


def build_messages(
    claim: Claim,
    *,
    without: Collection[str] = (),
    prefill_prefix: str | None = None,
) -> list[dict[str, str]]:
    """Compose the chat messages the injection describes, in claim order.

    Ablation is deletion of text, so ``without`` names spans to leave out and
    ``prefill_prefix`` replaces the prefill with a shorter character prefix.
    Both are how a configuration differs from the recorded run.
    """
    dropped = set(without)
    messages: list[dict[str, str]] = []
    for span in claim.spans:
        if span.id in dropped:
            continue
        text = span.text
        if span.role == "prefill" and prefill_prefix is not None:
            text = prefill_prefix
        messages.append({"role": _MESSAGE_ROLE[span.role], "content": text})
    return messages


def render_prompt(
    template: ChatTemplate, messages: Sequence[dict[str, str]], *, continue_final_message: bool
) -> str:
    """Apply the pinned template to the messages and return the prompt text."""
    environment = Environment(autoescape=False, undefined=Undefined, keep_trailing_newline=True)
    try:
        compiled = environment.from_string(template.text)
    except jinja2.TemplateSyntaxError as exc:
        raise TemplateError(f"chat template is not valid Jinja: {exc}") from exc

    try:
        return compiled.render(
            messages=list(messages),
            add_generation_prompt=not continue_final_message,
            tools=[],
            bos_token="",
            eos_token="",
        )
    except (jinja2.TemplateError, AttributeError, KeyError, TypeError) as exc:
        raise TemplateError(f"chat template failed to render: {exc}") from exc


def longest_prefix(
    text: str, max_tokens: int, tokenize: Callable[[str], Sequence[int]]
) -> PrefixSlice:
    """The longest character prefix of ``text`` that fits in ``max_tokens``.

    Truncation is by token budget but the cut must land on a character boundary,
    and only /tokenize knows token boundaries in both dialects. Binary search
    assumes prefix token counts are monotonic in prefix length, which holds for
    the tokenizers this tool targets.
    """
    total = len(tokenize(text))
    if max_tokens >= total:
        return PrefixSlice(text=text, tokens=total)

    low, high = 0, len(text)
    best = PrefixSlice(text="", tokens=0)
    while low <= high:
        middle = (low + high) // 2
        candidate = text[:middle]
        count = len(tokenize(candidate))
        if count <= max_tokens:
            best = PrefixSlice(text=candidate, tokens=count)
            low = middle + 1
        else:
            high = middle - 1
    return best


def truncation_ladder(
    span: Span, tokenize: Callable[[str], Sequence[int]]
) -> list[PrefixSlice]:
    """Every measured step of a prefill span's prefix ladder, ascending."""
    return [longest_prefix(span.text, budget, tokenize) for budget in span.truncations]
