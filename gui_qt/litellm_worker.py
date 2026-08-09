"""Background workers for operations that may import or call LiteLLM."""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from collections.abc import Mapping
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from PySide6.QtCore import QThread, Signal

from gemini_model_catalog import filter_gemini_generation_config
from litellm_provider_config import (
    LITELLM_CATALOG_URL,
    LITELLM_PYPI_URL,
    CustomLiteLLMProvider,
    build_native_catalog_headers,
    installed_litellm_version,
    latest_compatible_litellm_version,
    models_from_native_catalog_payload,
    models_from_remote_catalog,
    native_catalog_endpoint,
    providers_from_remote_catalog,
)
from litellm_sync_backend import LiteLLMSyncBackend
from sync_model_backend import SyncGenerationRequest
from .user_copy import CUSTOM_LITELLM_PROVIDER_COPY


CONNECTION_TEST_TIMEOUT_SECONDS = 30
# Per-request cap for catalog/version HTTP calls.
CATALOG_TIMEOUT_SECONDS = 20
# Whole model-catalog operation (official + optional LiteLLM fallback).
CATALOG_TOTAL_BUDGET_SECONDS = 35
# Do not start another HTTP hop with less than this remaining.
MIN_REQUEST_TIMEOUT_SECONDS = 3.0

# Completion payloads use this prefix so the GUI can treat cancel as non-error.
CANCELLED_MESSAGE_PREFIX = "已取消"


class OperationCancelled(Exception):
    """User cancelled a LiteLLM network worker."""


class BudgetExhausted(TimeoutError):
    """Shared operation deadline elapsed before the next network hop."""


def _connection_error_message(exc: Exception) -> str:
    category = str(getattr(exc, "category", "provider_error") or "provider_error")
    details = {
        "authentication": "身份验证失败，请检查供应商密钥。",
        "rate_limit": "供应商限流或配额不足，请稍后重试。",
        "service_unavailable": "供应商服务暂时不可用或请求超时。",
        "missing_dependency": "LiteLLM 尚未正确安装。",
        "provider_error": "请求失败，请检查模型、API Base 和网络。",
    }
    return f"连接失败 [{category}]: {details.get(category, details['provider_error'])}"


def _http_error_message(exc: HTTPError, label: str) -> str:
    code = int(getattr(exc, "code", 0) or 0)
    if code in {401, 403}:
        return f"{label} 身份验证失败（HTTP {code}），请检查 API Key。"
    if code == 404:
        return f"{label} 模型列表接口不可用（HTTP 404）。"
    if code == 429:
        return f"{label} 限流（HTTP 429），请稍后重试。"
    return f"{label} 请求失败（HTTP {code}）。"


def is_cancelled_message(message: object) -> bool:
    """Return True when a worker completion payload represents user cancel."""
    text = str(message or "").strip()
    return text.startswith(CANCELLED_MESSAGE_PREFIX)


class _CancellableNetworkWorker(QThread):
    """QThread helper: cooperative cancel, shared deadline, progress messages."""

    # Optional; subclasses that emit progress should declare the same Signal.
    progress = Signal(str)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._cancel = False
        self._active_response: Any = None
        self._deadline: float | None = None
        self._budget_seconds = float(CATALOG_TOTAL_BUDGET_SECONDS)
        # Network I/O already yields the GIL; keep workers low-priority so the
        # settings form stays snappy while a catalog request is in flight.
        self.setPriority(QThread.Priority.LowPriority)

    def request_cancel(self) -> None:
        """Ask the worker to stop; close any in-flight HTTP response if possible."""
        self._cancel = True
        self.requestInterruption()
        self._cancel_active_async_task()
        response = self._active_response
        if response is None:
            return
        try:
            response.close()
        except Exception:
            pass

    def _cancel_active_async_task(self) -> None:
        """Hook for workers whose request runs in an asyncio event loop."""

    def is_cancelled(self) -> bool:
        return bool(self._cancel or self.isInterruptionRequested())

    def _ensure_not_cancelled(self) -> None:
        if self.is_cancelled():
            raise OperationCancelled()

    def _start_budget(self, total_seconds: float) -> None:
        self._budget_seconds = max(0.0, float(total_seconds))
        self._deadline = time.monotonic() + self._budget_seconds

    def _remaining_timeout(self, *, cap: float | None = None) -> float:
        """Return urlopen timeout for the next hop within the shared budget."""
        per_request = float(CATALOG_TIMEOUT_SECONDS if cap is None else cap)
        if self._deadline is None:
            return per_request
        remaining = self._deadline - time.monotonic()
        if remaining < MIN_REQUEST_TIMEOUT_SECONDS:
            raise BudgetExhausted(
                f"联网加载总时限（约 {self._budget_seconds:g} 秒）已用尽"  # noqa: RUF001
            )
        return min(per_request, remaining)

    def _emit_progress(self, message: str) -> None:
        text = str(message or "").strip()
        if not text:
            return
        # progress is declared on the base class so emit is always available.
        self.progress.emit(text)

    def _load_json_url(self, request: Request, *, timeout: float) -> Any:
        """GET JSON with cancel checks; closing the response aborts a stuck read."""
        self._ensure_not_cancelled()
        opened = urlopen(request, timeout=timeout)
        self._active_response = opened
        try:
            # Real urllib responses and test doubles both support context manager.
            with opened as response:
                self._ensure_not_cancelled()
                payload = json.load(response)
                self._ensure_not_cancelled()
                return payload
        except Exception:
            if self.is_cancelled():
                raise OperationCancelled() from None
            raise
        finally:
            try:
                opened.close()
            except Exception:
                pass
            self._active_response = None

    def _load_litellm_catalog(self) -> dict:
        """Load and validate the shared LiteLLM online catalog."""
        request = Request(
            LITELLM_CATALOG_URL,
            headers={"User-Agent": "renpy-translation-lab"},
        )
        catalog = self._load_json_url(request, timeout=self._remaining_timeout())
        if not isinstance(catalog, dict):
            raise ValueError("LiteLLM 官方目录格式无效")
        return catalog


class LiteLLMModelCatalogWorker(_CancellableNetworkWorker):
    """Fetch a provider's text-model list in a cancellable background thread.

    ``custom_providers`` is a snapshot of the custom OpenAI-compatible registry
    (id → :class:`~litellm_provider_config.CustomLiteLLMProvider`) taken at
    construction time. Custom ids route to their configured ``models_url`` with
    the resolved keyring/env key; they never fall back to the LiteLLM online
    subset because user-defined ids do not exist there.
    """

    completed = Signal(object, object, object)
    progress = Signal(str)

    def __init__(
        self,
        provider: str,
        api_key: str = "",
        parent=None,
        custom_providers: Mapping[str, CustomLiteLLMProvider] | None = None,
    ) -> None:
        super().__init__(parent)
        self.provider = str(provider or "").strip().lower()
        self.api_key = str(api_key or "").strip()
        self._custom_providers = (
            dict(custom_providers) if isinstance(custom_providers, Mapping) else {}
        )

    def _fetch_litellm_catalog(self) -> tuple[str, ...]:
        catalog = self._load_litellm_catalog()
        models = models_from_remote_catalog(self.provider, catalog)
        if not models:
            raise ValueError(f"LiteLLM 目录中没有 {self.provider} 文本模型")
        return models

    def _fetch_native_catalog(self) -> tuple[tuple[str, ...], str]:
        endpoint = native_catalog_endpoint(self.provider, self._custom_providers)
        if endpoint is None:
            raise ValueError(f"未配置 {self.provider} 官方模型列表")
        api_key = self.api_key
        custom = self._custom_providers.get(self.provider)
        if not api_key and custom is not None and custom.api_key_env:
            # Same explicit env fallback as the request backend: never fall
            # back to OPENAI_API_KEY for a third-party endpoint.
            api_key = str(os.environ.get(custom.api_key_env) or "").strip()
        if endpoint.require_key and not api_key:
            raise ValueError(
                CUSTOM_LITELLM_PROVIDER_COPY["worker_missing_key"].format(
                    label=endpoint.label
                )
            )

        headers = build_native_catalog_headers(endpoint, api_key)
        request = Request(endpoint.url, headers=headers)
        try:
            payload = self._load_json_url(request, timeout=self._remaining_timeout())
        except OperationCancelled:
            raise
        except BudgetExhausted:
            raise
        except HTTPError as exc:
            raise RuntimeError(_http_error_message(exc, endpoint.label)) from exc
        except URLError as exc:
            raise RuntimeError(f"{endpoint.label} 网络错误：{exc.reason}") from exc

        models = models_from_native_catalog_payload(endpoint, payload)
        if not models:
            raise ValueError(f"{endpoint.label} 未返回可用文本模型")
        return models, endpoint.source

    def run(self) -> None:
        online_errors: list[str] = []
        self._start_budget(CATALOG_TOTAL_BUDGET_SECONDS)
        try:
            self._ensure_not_cancelled()
            endpoint = native_catalog_endpoint(self.provider, self._custom_providers)
            if endpoint is not None:
                # Prefer each provider's own live catalog. LiteLLM's pricing table
                # is only a subset and lags behind official model releases.
                self._emit_progress(f"正在请求 {endpoint.label} 官方模型列表…")
                try:
                    models, source = self._fetch_native_catalog()
                    self._ensure_not_cancelled()
                    self.completed.emit(models, source, None)
                    return
                except OperationCancelled:
                    raise
                except BudgetExhausted as budget_exc:
                    online_errors.append(f"{endpoint.label}：{budget_exc}")
                    raise RuntimeError("；".join(online_errors)) from budget_exc
                except Exception as native_exc:
                    online_errors.append(f"{endpoint.label}：{native_exc}")
                    self._ensure_not_cancelled()
                    if self.provider in self._custom_providers:
                        # LiteLLM's online subset has no entry for user-defined
                        # ids, so the fallback would always fail with a
                        # misleading "provider not found" error.
                        raise RuntimeError("；".join(online_errors)) from native_exc
                    # Bail before claiming "正在回退" if the shared budget is gone.
                    try:
                        self._remaining_timeout()
                    except BudgetExhausted as budget_exc:
                        online_errors.append(f"LiteLLM：{budget_exc}")
                        raise RuntimeError("；".join(online_errors)) from budget_exc
                    self._emit_progress(
                        f"{endpoint.label} 官方列表失败，正在改用 LiteLLM 子集目录…"
                    )
                    try:
                        models = self._fetch_litellm_catalog()
                        self._ensure_not_cancelled()
                        warning = (
                            f"{endpoint.label} 官方列表失败，已改用 LiteLLM 子集目录："
                            f"{native_exc}"
                        )
                        self.completed.emit(models, "online", warning)
                        return
                    except OperationCancelled:
                        raise
                    except BudgetExhausted as budget_exc:
                        online_errors.append(f"LiteLLM：{budget_exc}")
                        raise RuntimeError("；".join(online_errors)) from budget_exc
                    except Exception as litellm_exc:
                        online_errors.append(f"LiteLLM：{litellm_exc}")
                        raise RuntimeError("；".join(online_errors)) from litellm_exc

            self._emit_progress("正在请求 LiteLLM 在线模型目录…")
            models = self._fetch_litellm_catalog()
            self._ensure_not_cancelled()
            self.completed.emit(models, "online", None)
        except OperationCancelled:
            self.completed.emit((), "", f"{CANCELLED_MESSAGE_PREFIX}模型列表加载。")
        except Exception as exc:
            if self.is_cancelled():
                self.completed.emit((), "", f"{CANCELLED_MESSAGE_PREFIX}模型列表加载。")
            else:
                self.completed.emit((), "", f"联网加载模型失败：{exc}")


class LiteLLMProviderCatalogWorker(_CancellableNetworkWorker):
    """Fetch the current LiteLLM provider catalog only on explicit user action."""

    completed = Signal(object, object, object)
    progress = Signal(str)

    def run(self) -> None:
        self._start_budget(CATALOG_TIMEOUT_SECONDS)
        try:
            self._ensure_not_cancelled()
            self._emit_progress("正在请求 LiteLLM 官方供应商目录…")
            catalog = self._load_litellm_catalog()
            providers = providers_from_remote_catalog(catalog)
            if not providers:
                raise ValueError("LiteLLM 官方目录未返回供应商")
            self._ensure_not_cancelled()
            self.completed.emit(providers, "online", None)
        except OperationCancelled:
            self.completed.emit((), "", f"{CANCELLED_MESSAGE_PREFIX}供应商列表加载。")
        except Exception as exc:
            if self.is_cancelled():
                self.completed.emit((), "", f"{CANCELLED_MESSAGE_PREFIX}供应商列表加载。")
            else:
                self.completed.emit((), "", f"联网加载供应商失败：{exc}")


class LiteLLMVersionWorker(_CancellableNetworkWorker):
    completed = Signal(str, str, str, str, object)
    progress = Signal(str)

    def run(self) -> None:
        installed = ""
        self._start_budget(CATALOG_TIMEOUT_SECONDS)
        try:
            installed = installed_litellm_version()
            self._ensure_not_cancelled()
            self._emit_progress("正在查询 PyPI 上的 LiteLLM 版本…")
            request = Request(
                LITELLM_PYPI_URL,
                headers={"User-Agent": "renpy-translation-lab"},
            )
            payload = self._load_json_url(request, timeout=self._remaining_timeout())
            latest = str(payload.get("info", {}).get("version", "")).strip()
            if not latest:
                raise ValueError("PyPI 未返回最新版本")
            releases = payload.get("releases", {})
            if not isinstance(releases, dict):
                raise ValueError("PyPI 未返回版本兼容信息")
            compatible = latest_compatible_litellm_version(releases, sys.version_info[:3])
            requires_python = str(payload.get("info", {}).get("requires_python") or "")
            self._ensure_not_cancelled()
            self.completed.emit(installed, latest, compatible, requires_python, None)
        except OperationCancelled:
            self.completed.emit(
                installed, "", "", "", f"{CANCELLED_MESSAGE_PREFIX}版本检查。"
            )
        except Exception as exc:
            if self.is_cancelled():
                self.completed.emit(
                    installed, "", "", "", f"{CANCELLED_MESSAGE_PREFIX}版本检查。"
                )
            else:
                self.completed.emit(installed, "", "", "", str(exc))


class LiteLLMConnectionTestWorker(_CancellableNetworkWorker):
    """Send one minimal completion request to verify a provider connection.

    ``custom_providers`` is snapshotted at construction and forwarded to
    :class:`~litellm_sync_backend.LiteLLMSyncBackend` so custom ids get the same
    ``openai/<model>`` + ``api_base`` rewrite and credential resolution as
    production sync requests.
    """

    completed = Signal(bool, str)
    progress = Signal(str)

    def __init__(
        self,
        model: str,
        api_key: str = "",
        parent=None,
        custom_providers: Mapping[str, CustomLiteLLMProvider] | None = None,
    ) -> None:
        super().__init__(parent)
        self.model = model
        self.api_key = api_key
        self._custom_providers = (
            dict(custom_providers) if isinstance(custom_providers, Mapping) else {}
        )
        self._async_loop: asyncio.AbstractEventLoop | None = None
        self._async_task: asyncio.Task | None = None

    def _cancel_active_async_task(self) -> None:
        loop = self._async_loop
        task = self._async_task
        if loop is None or task is None or task.done():
            return
        try:
            loop.call_soon_threadsafe(task.cancel)
        except RuntimeError:
            # The event loop may be closing at the same time as the GUI cancel.
            pass

    async def _generate_async(self):
        self._ensure_not_cancelled()
        self._emit_progress("正在发起最小连接测试请求…")
        return await LiteLLMSyncBackend(
            api_key=self.api_key or None,
            custom_providers=self._custom_providers,
        ).generate_async(
            SyncGenerationRequest(
                model=self.model,
                contents="Reply with OK.",
                config=filter_gemini_generation_config(
                    self.model,
                    {
                        "max_output_tokens": 8,
                        "temperature": 0,
                        "timeout": CONNECTION_TEST_TIMEOUT_SECONDS,
                    },
                ),
            )
        )

    def run(self) -> None:
        if self.is_cancelled():
            self.completed.emit(False, f"{CANCELLED_MESSAGE_PREFIX}连接测试。")
            return
        loop = asyncio.new_event_loop()
        self._async_loop = loop
        asyncio.set_event_loop(loop)
        task = loop.create_task(self._generate_async())
        self._async_task = task
        try:
            result = loop.run_until_complete(task)
            if self.is_cancelled():
                self.completed.emit(False, f"{CANCELLED_MESSAGE_PREFIX}连接测试。")
                return
            text = result.response_text.strip()
            self.completed.emit(True, f"连接成功。模型返回：{text[:80] or '（空响应）'}")
        except asyncio.CancelledError:
            self.completed.emit(False, f"{CANCELLED_MESSAGE_PREFIX}连接测试。")
        except OperationCancelled:
            self.completed.emit(False, f"{CANCELLED_MESSAGE_PREFIX}连接测试。")
        except Exception as exc:
            if self.is_cancelled():
                self.completed.emit(False, f"{CANCELLED_MESSAGE_PREFIX}连接测试。")
            else:
                self.completed.emit(False, _connection_error_message(exc))
        finally:
            self._async_task = None
            self._async_loop = None
            try:
                pending = asyncio.all_tasks(loop)
                for pending_task in pending:
                    pending_task.cancel()
                if pending:
                    loop.run_until_complete(
                        asyncio.gather(*pending, return_exceptions=True)
                    )
                loop.run_until_complete(loop.shutdown_asyncgens())
            except Exception:
                pass
            finally:
                asyncio.set_event_loop(None)
                loop.close()
