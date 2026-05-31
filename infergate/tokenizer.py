from __future__ import annotations

import hashlib
import math
import os
from collections.abc import Mapping
from typing import Any

from .schemas import RequestContext


def _header_get(headers: Mapping[str, Any], key: str) -> str | None:
    lower_key = key.lower()
    for item_key, value in headers.items():
        if str(item_key).lower() == lower_key:
            return str(value)
    return None


def _metadata_get(metadata: Mapping[str, Any], key: str) -> Any:
    return metadata.get(key) or metadata.get(key.replace("-", "_"))


def message_content_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "text":
                    parts.append(str(item.get("text", "")))
                elif "text" in item:
                    parts.append(str(item["text"]))
            else:
                parts.append(str(item))
        return "\n".join(parts)
    return str(content)


def extract_prompt_text(payload: Mapping[str, Any]) -> str:
    if "messages" in payload and isinstance(payload["messages"], list):
        chunks: list[str] = []
        for message in payload["messages"]:
            if not isinstance(message, Mapping):
                chunks.append(str(message))
                continue
            role = message.get("role", "user")
            chunks.append(f"{role}: {message_content_to_text(message.get('content'))}")
        return "\n".join(chunks)
    if "prompt" in payload:
        return message_content_to_text(payload["prompt"])
    return ""


def prompt_prefix_hash(prompt_text: str, prefix_chars: int = 4096) -> str:
    prefix = prompt_text[:prefix_chars].encode("utf-8", errors="ignore")
    return hashlib.sha256(prefix).hexdigest()[:24]


class TokenEstimator:
    def __init__(self, tokenizer_id: str | None = None, prefer_hf: bool = False) -> None:
        self.tokenizer_id = tokenizer_id or os.getenv("INFERGATE_TOKENIZER_ID")
        self.tokenizer = None
        self.fallback = True
        self.load_error: str | None = None
        if prefer_hf and self.tokenizer_id:
            try:
                from transformers import AutoTokenizer  # type: ignore

                self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_id)
                self.fallback = False
            except Exception as exc:
                self.tokenizer = None
                self.fallback = True
                self.load_error = str(exc)

    def count(self, text: str) -> int:
        if not text:
            return 1
        if self.tokenizer is not None:
            try:
                return max(1, len(self.tokenizer.encode(text)))
            except Exception:
                self.fallback = True
        return max(1, math.ceil(len(text) / 4))


def get_max_tokens(payload: Mapping[str, Any], default: int = 512) -> int:
    value = payload.get("max_tokens", payload.get("max_completion_tokens", default))
    try:
        return max(1, int(value))
    except (TypeError, ValueError):
        return default


def build_request_context(
    payload: Mapping[str, Any],
    headers: Mapping[str, Any],
    estimator: TokenEstimator,
) -> RequestContext:
    metadata = payload.get("metadata") if isinstance(payload.get("metadata"), Mapping) else {}
    prompt_text = extract_prompt_text(payload)
    prompt_tokens = estimator.count(prompt_text)
    max_tokens = get_max_tokens(payload)
    cache_key = (
        _header_get(headers, "x-cache-key")
        or _metadata_get(metadata, "x-cache-key")
        or _metadata_get(metadata, "cache_key")
        or prompt_prefix_hash(prompt_text)
    )
    deadline_value = _header_get(headers, "x-request-deadline-ms") or _metadata_get(
        metadata, "x-request-deadline-ms"
    )
    utility_value = _header_get(headers, "x-request-utility") or _metadata_get(
        metadata, "x-request-utility"
    )
    step_value = _header_get(headers, "x-session-step") or _metadata_get(metadata, "x-session-step")
    total_steps_value = _header_get(headers, "x-session-total-steps") or _metadata_get(
        metadata, "x-session-total-steps"
    )
    session_id = (
        _header_get(headers, "x-session-id")
        or _metadata_get(metadata, "x-session-id")
        or _metadata_get(metadata, "session_id")
        or "default"
    )
    tenant_id = (
        _header_get(headers, "x-tenant-id")
        or _metadata_get(metadata, "x-tenant-id")
        or _metadata_get(metadata, "tenant_id")
        or session_id
    )

    def parse_int(value: Any, default: int | None = None) -> int | None:
        if value is None:
            return default
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    def parse_float(value: Any, default: float) -> float:
        if value is None:
            return default
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    return RequestContext(
        session_id=str(session_id),
        tenant_id=str(tenant_id),
        session_step=max(0, parse_int(step_value, 0) or 0),
        session_total_steps=max(1, parse_int(total_steps_value, 1) or 1),
        utility=max(0.0, parse_float(utility_value, 1.0)),
        deadline_ms=parse_int(deadline_value, None),
        cache_key=str(cache_key),
        estimated_cost=prompt_tokens + max_tokens,
        prompt_tokens=prompt_tokens,
        max_tokens=max_tokens,
        prompt_text=prompt_text,
        tokenizer_fallback=estimator.fallback,
    )
