"""Shared connection/auth/URL contract for the direct OpenAI-compatible adapter.

The generation backend and the model-catalog reader must resolve credentials,
validate extra headers and build endpoint URLs identically; keeping that logic
here prevents the two paths from drifting or re-introducing a sensitive-header
bypass on the catalog side.
"""

from __future__ import annotations

import os
from typing import Any, Mapping
from urllib.parse import urlsplit, urlunsplit

from openai_compatible_contract import is_sensitive_header
from sync_model_backend import SyncBackendError


def _parsed_http_url(value: object) -> tuple[str, str, str, str]:
    """Return ``(scheme, netloc, path, query)`` for a clean http(s) URL."""

    text = str(value or "").strip()
    if not text:
        raise SyncBackendError(
            "unsupported_capability",
            request_metadata={"reason": "missing_base_url"},
        )
    try:
        parsed = urlsplit(text)
    except ValueError as exc:
        raise SyncBackendError(
            "unsupported_capability",
            request_metadata={"reason": "invalid_base_url"},
        ) from exc
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.netloc
        or parsed.username is not None
        or parsed.password is not None
        or parsed.fragment
    ):
        raise SyncBackendError(
            "unsupported_capability",
            request_metadata={"reason": "invalid_base_url"},
        )
    return parsed.scheme, parsed.netloc, parsed.path, parsed.query


def openai_compatible_endpoint(
    base_url: object,
    suffix: str,
    *,
    explicit_url: object = "",
) -> str:
    """Return a clean endpoint URL under *base_url*.

    ``explicit_url`` (for example a provider ``models_url``) wins when set.
    The configured query string is preserved; fragments are rejected because
    provider endpoints must not carry client-side fragments.
    """

    target = str(explicit_url or "").strip()
    if target:
        scheme, netloc, path, query = _parsed_http_url(target)
        return urlunsplit((scheme, netloc, path.rstrip("/"), query, ""))
    scheme, netloc, path, query = _parsed_http_url(base_url)
    base_path = path.rstrip("/")
    suffix = "/" + str(suffix or "").strip("/")
    if not base_path.casefold().endswith(suffix.casefold()):
        base_path = f"{base_path}{suffix}"
    return urlunsplit((scheme, netloc, base_path, query, ""))


def redacted_endpoint(url: object) -> str:
    """Return an endpoint URL without query/fragment for diagnostics."""

    parsed = urlsplit(str(url or ""))
    return urlunsplit((parsed.scheme, parsed.netloc, parsed.path, "", ""))


class OpenAICompatibleConnection:
    """Resolve credentials, headers and endpoints for one provider connection."""

    def __init__(
        self,
        *,
        provider: str,
        base_url: str,
        models_url: str = "",
        credential_ref: Mapping[str, Any] | None = None,
        extra_headers: Mapping[str, Any] | None = None,
        api_key: str | None = None,
    ) -> None:
        self.provider = str(provider or "openai_compatible")
        self.base_url = str(base_url or "")
        self.models_url = str(models_url or "")
        self.credential_ref = dict(credential_ref or {})
        self.extra_headers = {
            str(key): str(value)
            for key, value in dict(extra_headers or {}).items()
        }
        self.api_key = str(api_key or "").strip() or None

    # ------------------------------------------------------------------
    # Credentials
    # ------------------------------------------------------------------
    def resolve_api_key(self) -> str | None:
        """Resolve the configured credential reference, or ``None`` for none."""

        if self.api_key:
            return self.api_key
        ref = self.credential_ref
        kind = str(ref.get("kind") or "none").strip().lower()
        if kind == "none":
            return None
        if kind == "env":
            name = str(ref.get("name") or ref.get("env_name") or "").strip()
            value = str(os.environ.get(name) or "").strip()
            if not value:
                raise SyncBackendError(
                    "authentication",
                    request_metadata={"reason": "credential_env_missing"},
                )
            return value
        if kind == "keyring":
            name = str(ref.get("name") or self.provider or "").strip()
            # ``litellm_provider_config`` is a credential-reference reader; it
            # does not import the LiteLLM runtime at module import time.  A
            # genuine import failure is still reported as missing_dependency
            # instead of being mislabeled as a bad credential.
            try:
                from litellm_provider_config import load_provider_api_key
            except Exception as exc:
                raise SyncBackendError(
                    "missing_dependency",
                    request_metadata={"reason": "credential_reader_unavailable"},
                ) from exc
            try:
                value = str(load_provider_api_key(name) or "").strip()
            except Exception as exc:
                raise SyncBackendError(
                    "provider_error",
                    request_metadata={"reason": "credential_reader_error"},
                ) from exc
            if not value:
                raise SyncBackendError(
                    "authentication",
                    request_metadata={"reason": "credential_keyring_missing"},
                )
            return value
        raise SyncBackendError(
            "unsupported_capability",
            request_metadata={"reason": "credential_kind_unsupported"},
        )

    # ------------------------------------------------------------------
    # Endpoints / headers
    # ------------------------------------------------------------------
    def chat_completions_url(self) -> str:
        return openai_compatible_endpoint(self.base_url, "/chat/completions")

    def provider_models_url(self) -> str:
        return openai_compatible_endpoint(
            self.base_url,
            "/models",
            explicit_url=self.models_url,
        )

    def request_headers(self, *, json_content: bool = True) -> dict[str, str]:
        headers = {
            "Accept": "application/json",
            "User-Agent": "renpy-translation-lab/openai-compatible",
        }
        if json_content:
            headers["Content-Type"] = "application/json"
        api_key = self.resolve_api_key()
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        for key, value in self.extra_headers.items():
            if not key or not value:
                continue
            if any(marker in key or marker in value for marker in ("\r", "\n")):
                raise SyncBackendError(
                    "unsupported_capability",
                    request_metadata={
                        "provider": self.provider,
                        "reason": "invalid_extra_header",
                    },
                )
            if is_sensitive_header(key, value):
                # Defense in depth: an unvalidated/tolerated config path must
                # not be able to override Authorization or smuggle credentials.
                raise SyncBackendError(
                    "unsupported_capability",
                    request_metadata={
                        "provider": self.provider,
                        "reason": "sensitive_extra_header",
                    },
                )
            try:
                key.encode("ascii")
                value.encode("latin-1")
            except UnicodeEncodeError as exc:
                raise SyncBackendError(
                    "unsupported_capability",
                    request_metadata={
                        "provider": self.provider,
                        "reason": "invalid_extra_header",
                    },
                ) from exc
            headers[key] = value
        return headers
