"""Endpoint adapters for the two backends this tool claims to support.

Both vLLM and llama.cpp speak an OpenAI-compatible /v1/chat/completions, but they
disagree on three things that matter here: the body of /tokenize, where the chat
template body can be read, and whether the last assistant message may be a
prefix. So this module keeps two explicit adapters instead of one "unified"
client that would have to lie about at least one of them.

Two rules hold across both adapters:

* prompt token ids come from POST /tokenize and never from the chat response —
  llama.cpp's chat response carries only usage.prompt_tokens, no ids at all;
* the chat request body is serialized once, canonically, and those exact bytes
  are both sent and hashed into the manifest.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import re
import time
import zlib
from dataclasses import dataclass
from typing import Any, ClassVar, Sequence

import httpx

from .claim import BackendKind, ModelSpec, Sampling
from .config import RunSettings


class BackendError(Exception):
    """The endpoint refused, failed, or answered with something unparseable."""


@dataclass(frozen=True)
class TokenizeResult:
    tokens: tuple[int, ...]
    max_model_len: int | None = None

    @property
    def count(self) -> int:
        return len(self.tokens)


@dataclass(frozen=True)
class ChatResult:
    text: str
    request_body: dict[str, Any]
    request_bytes: bytes
    response_body: dict[str, Any]
    response_bytes: bytes
    usage: dict[str, Any]
    elapsed_ms: int


def canonical_json(obj: Any) -> bytes:
    """The one serialization used for both the wire and the hash."""
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _urls(endpoint: str) -> tuple[str, str]:
    """Return (native root, chat completions URL).

    Accepts both ``http://host:8000`` and ``http://host:8000/v1``; native
    endpoints (/tokenize, /version, /props) always live at the root.
    """
    base = endpoint.rstrip("/")
    if base.endswith("/v1"):
        return base[: -len("/v1")], f"{base}/chat/completions"
    return base, f"{base}/v1/chat/completions"


def _snippet(text: str, limit: int = 300) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else f"{text[:limit]}..."


def _token_ids(payload: dict[str, Any], url: str) -> tuple[int, ...]:
    raw = payload.get("tokens")
    if not isinstance(raw, list):
        raise BackendError(f"{url}: response has no 'tokens' list: {_snippet(json.dumps(payload))}")
    ids: list[int] = []
    for item in raw:
        # llama.cpp returns bare ids with with_pieces=false and {id, piece}
        # objects otherwise; accept both rather than depend on a build flag.
        if isinstance(item, dict):
            item = item.get("id")
        if not isinstance(item, int):
            raise BackendError(f"{url}: token entries must be integers, got {item!r}")
        ids.append(item)
    return tuple(ids)


class Backend:
    """Shared transport; subclasses supply the dialect-specific pieces."""

    kind: ClassVar[BackendKind]
    TEMPLATE_HINT: ClassVar[str] = ""

    def __init__(
        self, spec: ModelSpec, client: httpx.Client, settings: RunSettings | None = None
    ) -> None:
        self.spec = spec
        self.settings = settings or RunSettings()
        self.root, self.chat_url = _urls(spec.endpoint)
        self._client = client

    def __enter__(self) -> "Backend":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def close(self) -> None:
        self._client.close()

    # -- dialect hooks -----------------------------------------------------

    def identity(self) -> str:
        """The endpoint's build string, as recorded in the manifest."""
        raise NotImplementedError

    def tokenizer_name(self) -> str:
        """The tokenizer identity this endpoint reports, as recorded in the manifest."""
        raise NotImplementedError

    def chat_template(self) -> str | None:
        """The template body this endpoint serves, or None when it serves none."""
        raise NotImplementedError

    def tokenize(self, text: str) -> TokenizeResult:
        raise NotImplementedError

    def _chat_body(
        self, messages: Sequence[dict[str, str]], sampling: Sampling, continue_final_message: bool
    ) -> dict[str, Any]:
        return {
            "model": self.spec.name,
            "messages": list(messages),
            "temperature": sampling.temperature,
            "top_p": sampling.top_p,
            "seed": sampling.seed,
            "max_tokens": sampling.max_tokens,
            "n": 1,
            "stream": False,
        }

    # -- shared transport --------------------------------------------------

    def _request(
        self,
        method: str,
        url: str,
        *,
        content: bytes | None = None,
        timeout: float,
        allow_missing: bool = False,
    ) -> httpx.Response:
        headers = {"content-type": "application/json"} if content is not None else None
        try:
            response = self._client.request(method, url, content=content, headers=headers, timeout=timeout)
        except httpx.HTTPError as exc:
            raise BackendError(f"{method} {url} failed: {exc}. Is the endpoint running?") from exc
        if allow_missing and response.status_code in (400, 404, 405, 501):
            return response
        if response.status_code >= 400:
            raise BackendError(
                f"{method} {url} returned HTTP {response.status_code}: {_snippet(response.text)}"
            )
        return response

    def _get_json(self, url: str, *, allow_missing: bool = False) -> dict[str, Any] | None:
        response = self._request(
            "GET", url, timeout=self.settings.tokenize_timeout_s, allow_missing=allow_missing
        )
        if allow_missing and response.status_code >= 400:
            return None
        return self._decode(response, url)

    def _post_json(self, url: str, body: dict[str, Any], *, timeout: float) -> dict[str, Any]:
        response = self._request("POST", url, content=canonical_json(body), timeout=timeout)
        return self._decode(response, url)

    @staticmethod
    def _decode(response: httpx.Response, url: str) -> dict[str, Any]:
        try:
            payload = response.json()
        except ValueError as exc:
            raise BackendError(f"{url}: response is not JSON: {_snippet(response.text)}") from exc
        if not isinstance(payload, dict):
            raise BackendError(f"{url}: response must be a JSON object: {_snippet(response.text)}")
        return payload

    def chat(
        self,
        messages: Sequence[dict[str, str]],
        sampling: Sampling,
        *,
        continue_final_message: bool,
    ) -> ChatResult:
        body = self._chat_body(messages, sampling, continue_final_message)
        payload = canonical_json(body)
        started = time.perf_counter()
        response = self._request(
            "POST", self.chat_url, content=payload, timeout=self.settings.request_timeout_s
        )
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        parsed = self._decode(response, self.chat_url)

        choices = parsed.get("choices")
        if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
            raise BackendError(f"{self.chat_url}: response has no choices: {_snippet(response.text)}")
        first = choices[0]
        message = first.get("message")
        text = ""
        if isinstance(message, dict) and isinstance(message.get("content"), str):
            text = message["content"]
        elif isinstance(first.get("text"), str):
            # /v1/completions-shaped answer; kept because proxies normalise it here.
            text = first["text"]

        usage = parsed.get("usage")
        return ChatResult(
            text=text,
            request_body=body,
            request_bytes=payload,
            response_body=parsed,
            response_bytes=response.content,
            usage=usage if isinstance(usage, dict) else {},
            elapsed_ms=elapsed_ms,
        )


class VllmBackend(Backend):
    """vLLM: /tokenize takes {model, prompt}; prefill via continue_final_message."""

    kind: ClassVar[BackendKind] = "vllm"
    TEMPLATE_HINT = (
        "vLLM only serves the template body from GET /tokenizer_info, and only when the "
        "server was started with --enable-tokenizer-info-endpoint"
    )

    def identity(self) -> str:
        payload = self._get_json(f"{self.root}/version")
        version = payload.get("version") if payload else None
        return f"vllm/{version or 'unknown'}"

    def tokenizer_name(self) -> str:
        # vLLM reports the served tokenizer only through /tokenizer_info; the
        # model label in the claim is the honest fallback when it does not.
        payload = self._get_json(f"{self.root}/tokenizer_info", allow_missing=True)
        name = payload.get("model_name") if payload else None
        return name if isinstance(name, str) and name.strip() else self.spec.name

    def chat_template(self) -> str | None:
        payload = self._get_json(f"{self.root}/tokenizer_info", allow_missing=True)
        template = payload.get("chat_template") if payload else None
        return template if isinstance(template, str) and template.strip() else None

    def tokenize(self, text: str) -> TokenizeResult:
        body = {"model": self.spec.name, "prompt": text}
        url = f"{self.root}/tokenize"
        payload = self._post_json(url, body, timeout=self.settings.tokenize_timeout_s)
        max_model_len = payload.get("max_model_len")
        return TokenizeResult(
            tokens=_token_ids(payload, url),
            max_model_len=max_model_len if isinstance(max_model_len, int) else None,
        )

    def _chat_body(
        self, messages: Sequence[dict[str, str]], sampling: Sampling, continue_final_message: bool
    ) -> dict[str, Any]:
        body = super()._chat_body(messages, sampling, continue_final_message)
        body["continue_final_message"] = continue_final_message
        body["add_generation_prompt"] = not continue_final_message
        return body


class LlamaCppBackend(Backend):
    """llama.cpp: /tokenize takes {content, ...}; the template lives in /props."""

    kind: ClassVar[BackendKind] = "llamacpp"
    TEMPLATE_HINT = "llama.cpp serves the template body from GET /props as 'chat_template'"

    def identity(self) -> str:
        payload = self._get_json(f"{self.root}/props") or {}
        build = payload.get("build_info") or payload.get("build_number")
        return f"llama.cpp/{build or 'unknown'}"

    def tokenizer_name(self) -> str:
        payload = self._get_json(f"{self.root}/props") or {}
        path = payload.get("model_path")
        return path if isinstance(path, str) and path.strip() else self.spec.name

    def chat_template(self) -> str | None:
        payload = self._get_json(f"{self.root}/props") or {}
        template = payload.get("chat_template")
        return template if isinstance(template, str) and template.strip() else None

    def tokenize(self, text: str) -> TokenizeResult:
        body = {
            "content": text,
            "add_special": False,
            "parse_special": False,
            "with_pieces": False,
        }
        url = f"{self.root}/tokenize"
        payload = self._post_json(url, body, timeout=self.settings.tokenize_timeout_s)
        return TokenizeResult(tokens=_token_ids(payload, url))


def build_backend(
    spec: ModelSpec, *, fixture: bool = False, settings: RunSettings | None = None
) -> Backend:
    """Create the adapter the claim asks for.

    ``fixture=True`` installs an in-process mock transport: the same adapters,
    the same request shapes, deterministic answers, and no network. It exists so
    the CLI and the test suite can run offline; it is not part of the product.
    """
    client = httpx.Client(timeout=None, transport=fixture_transport() if fixture else None)
    if spec.backend == "vllm":
        return VllmBackend(spec, client, settings)
    return LlamaCppBackend(spec, client, settings)


# -- offline fixture -------------------------------------------------------
#
# A deterministic stand-in for both dialects. Outputs are derived from the
# request bytes, so identical requests give identical answers. A non-zero
# temperature makes the fixture deliberately unstable, which is what lets the
# determinism gate be exercised without a GPU.

FIXTURE_VERSION = "0.6.3.post1+fixture"
FIXTURE_BUILD = "b5123+fixture"
FIXTURE_CHAT_TEMPLATE = (
    "{%- for message in messages %}\n"
    "{{- '<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>\\n' }}\n"
    "{%- endfor %}\n"
    "{%- if add_generation_prompt %}\n"
    "{{- '<|im_start|>assistant\\n' }}\n"
    "{%- endif %}"
)

_FIXTURE_PIECE_RE = re.compile(r"[A-Za-z0-9_]+|[^\sA-Za-z0-9_]")


def fixture_token_pieces(text: str) -> list[str]:
    """Word/punctuation pieces: stable, dialect-independent, and cheap."""
    return _FIXTURE_PIECE_RE.findall(text)


def fixture_token_ids(text: str) -> list[int]:
    return [zlib.crc32(piece.encode("utf-8")) % 60000 + 100 for piece in fixture_token_pieces(text)]


def fixture_transport() -> httpx.MockTransport:
    samples = itertools.count(1)

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path.rstrip("/") or "/"
        body: dict[str, Any] = {}
        if request.content:
            try:
                parsed = json.loads(request.content)
            except ValueError:
                return httpx.Response(400, json={"error": "fixture: request body is not JSON"})
            if isinstance(parsed, dict):
                body = parsed

        if path == "/version":
            return httpx.Response(200, json={"version": FIXTURE_VERSION})
        if path == "/props":
            return httpx.Response(
                200,
                json={
                    "build_info": FIXTURE_BUILD,
                    "model_path": "/fixture/qwen3-8b-instruct.gguf",
                    "chat_template": FIXTURE_CHAT_TEMPLATE,
                },
            )
        if path == "/tokenizer_info":
            return httpx.Response(
                200,
                json={"model_name": body.get("model", "qwen3-8b-instruct"), "chat_template": FIXTURE_CHAT_TEMPLATE},
            )
        if path == "/tokenize":
            if "content" in body:
                ids = fixture_token_ids(str(body["content"]))
                return httpx.Response(200, json={"tokens": ids})
            prompt = str(body.get("prompt", ""))
            ids = fixture_token_ids(prompt)
            return httpx.Response(
                200,
                json={"count": len(ids), "max_model_len": 32768, "tokens": ids},
            )
        if path == "/v1/chat/completions":
            return httpx.Response(200, json=_fixture_chat(body, request.content, next(samples)))
        return httpx.Response(404, json={"error": f"fixture: no route for {path}"})

    return httpx.MockTransport(handler)


def _fixture_chat(body: dict[str, Any], raw_request: bytes, sample: int) -> dict[str, Any]:
    digest = hashlib.sha256(raw_request).hexdigest()[:10]
    text = (
        f"api.flashnext.dev/v1/search?q=qwen+3.8&limit=20&run={digest}' "
        "-H 'accept: application/json'"
    )
    temperature = body.get("temperature", 0)
    if isinstance(temperature, (int, float)) and temperature > 0:
        text = f"{text} # sample {sample}"

    messages = body.get("messages") or []
    prompt_tokens = sum(
        len(fixture_token_ids(str(message.get("content", ""))))
        for message in messages
        if isinstance(message, dict)
    )
    completion_tokens = len(fixture_token_ids(text))
    return {
        "id": "chatcmpl-fixture",
        "object": "chat.completion",
        "created": 0,
        "model": body.get("model", "qwen3-8b-instruct"),
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }
